import os
import torch
import pandas as pd
from PIL import Image
import json
import re
import gc
import random
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

# -------------------- Configuration --------------------
TEST_CSV_PATH = "../FBHM/test.csv"
TRAIN_CSV_PATH = "../FBHM/train.csv"          # path to training samples (500 samples)
BASE_IMAGE_DIR = "../FBHM"
MODEL_REPO_ID = "pbhaskar/qwen3-vl-8b-sft-mami-15-epochs-earlystop-2"
# MODEL_REPO_ID = "pbhaskar/qwen3-vl-8b-sft-fhm-15-epochs-earlystop"
OUTPUT_TXT = "./32_shot_qwen3_sft_mami_15_epochs_earlystop_2_inference_on_FBHM_test_output.txt"
# OUTPUT_TXT = "./32_shot_qwen3_sft_MAMI_15_epochs_earlystop_inference_on_FBHM_test_output.txt"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.001

# Few‑shot settings
NUM_SHOTS = 32
RANDOM_SEED = 42

HF_TOKEN = ""
login(token=HF_TOKEN)

# -------------------------------------------------------

class QwenVLInference:
    def __init__(self, model_repo_id, few_shot_examples):
        """
        few_shot_examples : list of dicts, each with keys:
            'image' : PIL Image
            'text'  : extracted text (string)
            'label' : string, either "hateful" or "not-hateful"
        """
        print(f"Loading model from {model_repo_id}...")
        self.processor = AutoProcessor.from_pretrained(
            model_repo_id,
            trust_remote_code=True,
            do_image_splitting=False
        )
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        self.device = self.model.device
        print(f"Model loaded on device: {self.device}")

        # Store few‑shot examples
        self.few_shot_examples = few_shot_examples
        print(f"Loaded {len(self.few_shot_examples)} few-shot examples.")

    def safe_image_load(self, image_path):
        """Safely load image with error handling"""
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            return Image.new('RGB', (224, 224), color='black')

    def _build_few_shot_messages(self):
        """Build the conversation messages for the few-shot examples."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Consider the following definitions.\n"
                    "1. hateful - a direct or indirect attack on people based on characteristics, including ethnicity, race, "
                    "nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, "
                    "and disability or disease. Attack is defined as violent or dehumanizing (comparing people to non-human "
                    "things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking "
                    "hate crime is also considered hateful.\n"
                    "2. not-hateful - an attack which is not hateful and follows social norms.\n"
                    "Classify the above meme as hateful or not-hateful based on the above definitions considering the image "
                    "and the extracted text from the image delimited by three backticks.\n"
                    "Provide your answer in the format: hateful or not-hateful."
                )
            }
        ]

        # Add each few‑shot example as a user‑assistant exchange
        for ex in self.few_shot_examples:
            # User turn
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text", "text": f"Extracted text: ```{ex['text']}```\nProvide your answer in the format: hateful or not-hateful."}
                ]
            })
            # Assistant turn (only the label)
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex["label"]}]
            })

        return messages

    def predict(self, meme_data, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        """Generate prediction for a single meme using few‑shot examples."""
        # Resolve image path for test meme
        img_rel_path = meme_data['img']
        if img_rel_path.startswith("F") and "/" in img_rel_path:
            folder = img_rel_path.split("/")[0]
            img_name = img_rel_path.split("/")[-1]
            img_path = os.path.join(BASE_IMAGE_DIR, folder, "memes", img_name)
        else:
            img_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)

        if not os.path.exists(img_path):
            base_name = os.path.splitext(img_path)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if os.path.exists(base_name + ext):
                    img_path = base_name + ext
                    break

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path} (original: {img_rel_path})")
            return [f"Error: Image file not found: {img_path}"]

        test_image = self.safe_image_load(img_path)
        test_text = meme_data["text"]

        # Build the full conversation: few‑shot messages + final test query
        messages = self._build_few_shot_messages()
        # Add the final test user message
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": f"Extracted text: ```{test_text}```\nProvide your answer in the format: hateful or not-hateful."}
            ]
        })

        # Apply chat template and generate
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Collect all images that appear in the conversation (for the processor)
        # The easiest way: collect images from the few‑shot examples and the test image.
        all_images = [ex["image"] for ex in self.few_shot_examples] + [test_image]

        inputs = self.processor(
            text=[text],
            images=all_images,          # processor will match images to their positions in the text
            padding=True,
            return_tensors="pt",
            do_image_splitting=False
        ).to(self.device)

        generated_texts = ""
        try:
            with torch.no_grad():
                # generated_ids = self.model.generate(
                #     **inputs,
                #     max_new_tokens=max_new_tokens,
                #     temperature=temperature,
                #     do_sample=temperature > 0,
                #     pad_token_id=self.processor.tokenizer.pad_token_id,
                #     eos_token_id=self.processor.tokenizer.eos_token_id
                # )


                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
                
                new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
                
                generated_texts = self.processor.batch_decode(
                    new_tokens,
                    skip_special_tokens=True
                )
            
            # generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"Error during generation: {e}")
            generated_texts = [f"Error: {e}"]

        return generated_texts


def load_few_shot_examples(train_csv_path, num_shots, seed):
    """
    Load training data, randomly sample `num_shots` examples with the given seed,
    and return a list of dicts with 'image' (PIL), 'text', 'label'.
    """
    random.seed(seed)
    train_df = pd.read_csv(train_csv_path)

    # Ensure there are enough samples
    if len(train_df) < num_shots:
        print(f"Warning: Training set has only {len(train_df)} samples, using all.")
        num_shots = len(train_df)

    # Random sample indices
    sample_indices = random.sample(range(len(train_df)), num_shots)
    sampled_rows = train_df.iloc[sample_indices]

    examples = []
    for _, row in sampled_rows.iterrows():
        img_rel_path = row['img']
        # Resolve path (same logic as in predict)
        if img_rel_path.startswith("F") and "/" in img_rel_path:
            folder = img_rel_path.split("/")[0]
            img_name = img_rel_path.split("/")[-1]
            img_path = os.path.join(BASE_IMAGE_DIR, folder, "memes", img_name)
        else:
            img_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)

        # Try different extensions if necessary
        if not os.path.exists(img_path):
            base_name = os.path.splitext(img_path)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if os.path.exists(base_name + ext):
                    img_path = base_name + ext
                    break

        if not os.path.exists(img_path):
            print(f"Warning: Few-shot image not found: {img_path}, skipping this example.")
            continue

        image = Image.open(img_path).convert('RGB')
        text = row['text']
        # Convert label (0/1) to string
        label_val = row['label']
        label_str = "hateful" if label_val == 1 else "not-hateful"

        examples.append({
            'image': image,
            'text': text,
            'label': label_str
        })

    print(f"Loaded {len(examples)} valid few-shot examples.")
    return examples


def main():
    # Load training samples and select few‑shot examples
    print("\nLoading training data and sampling few‑shot examples...")
    few_shot_examples = load_few_shot_examples(TRAIN_CSV_PATH, NUM_SHOTS, RANDOM_SEED)

    # Initialize the inference model with the few‑shot examples
    inference_model = QwenVLInference(MODEL_REPO_ID, few_shot_examples)

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_data = test_df.to_dict('records')

    # Run predictions and log to text file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
        for index, meme_data in tqdm(enumerate(test_data), total=len(test_data), desc="Processing memes"):
            meme_series = pd.Series(meme_data)
            meme_series.name = index

            # Write meme data as JSON
            meme_dict = meme_series.to_dict()
            # Add full image path for reference
            img_rel_path = meme_dict['img']
            if img_rel_path.startswith("F") and "/" in img_rel_path:
                folder = img_rel_path.split("/")[0]
                img_name = img_rel_path.split("/")[-1]
                full_path = os.path.join(BASE_IMAGE_DIR, folder, "memes", img_name)
            else:
                full_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)
            meme_dict['full_image_path'] = full_path

            output_file.write(json.dumps(meme_dict, default=str))
            output_file.write('\n----------\n')

            # Run inference
            generated_texts = inference_model.predict(
                meme_series,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )

            for text in generated_texts:
                formatted_text = re.sub(r'\bassistant\b', 'Assistant:', text, flags=re.IGNORECASE)
                output_file.write(formatted_text)
                print(formatted_text)
                output_file.write('\n')

            output_file.write("\n##########\n")

            # Optional garbage collection
            if (index + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\nOutput file saved to '{OUTPUT_TXT}'")
    print(f"\nProcessed {len(test_data)} memes with {NUM_SHOTS}‑shot in‑context learning (seed={RANDOM_SEED}).")


if __name__ == "__main__":
    main()