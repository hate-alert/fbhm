import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # adjust as needed

import torch
import pandas as pd
from PIL import Image
import json
import re
import gc
import random
from tqdm import tqdm
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

from transformers import LlavaForConditionalGeneration, AutoProcessor

# ========== Configuration ==========
TEST_CSV_PATH = "../FBHM/test.csv"                         # FBHM test set
TRAIN_CSV_PATH = "../FBHM/train.csv"                       # FBHM training samples
BASE_IMAGE_DIR_FBHM = "../FBHM"                            # base directory for FBHM images

# MODEL_REPO_ID = "mistral-community/pixtral-12b"
MODEL_REPO_ID = "DakshJogchand/pixtral-12b-sft-fhm-15-epochs-earlystop-3"
OUTPUT_TXT = "./32_shot_pixtral_sft_fhm_15_epochs_earlystop_3_inference_on_FBHM_test_output.txt"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.001

# Few‑shot settings
NUM_SHOTS = 32
RANDOM_SEED = 42

HF_TOKEN = ""
login(token=HF_TOKEN)

# ========== Utility ==========
def load_few_shot_examples(train_csv_path, num_shots, seed):
    """
    Load FBHM training data, randomly sample `num_shots` examples with the given seed,
    and return a list of dicts with 'image' (PIL), 'text', 'label'.
    """
    random.seed(seed)
    train_df = pd.read_csv(train_csv_path)

    if len(train_df) < num_shots:
        print(f"Warning: Training set has only {len(train_df)} samples, using all.")
        num_shots = len(train_df)

    sample_indices = random.sample(range(len(train_df)), num_shots)
    sampled_rows = train_df.iloc[sample_indices]

    examples = []
    for _, row in sampled_rows.iterrows():
        img_rel_path = row['img']
        # Resolve FBHM image path (e.g., "FXXXXX/memes/YYYYY.jpg")
        if img_rel_path.startswith("F") and "/" in img_rel_path:
            folder = img_rel_path.split("/")[0]
            img_name = img_rel_path.split("/")[-1]
            img_path = os.path.join(BASE_IMAGE_DIR_FBHM, folder, "memes", img_name)
        else:
            img_path = os.path.join(BASE_IMAGE_DIR_FBHM, img_rel_path)

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
        label_val = row['label']
        label_str = "hateful" if label_val == 1 else "not-hateful"

        examples.append({
            'image': image,
            'text': text,
            'label': label_str
        })

    print(f"Loaded {len(examples)} valid few-shot examples from FBHM train.")
    return examples

def load_test_data(test_csv_path):
    """Load FBHM test CSV and return list of dicts."""
    df = pd.read_csv(test_csv_path)
    return df.to_dict('records')

# ========== Inference Class with Few‑Shot ==========
class PixtralVLInference:
    def __init__(self, model_repo_id, few_shot_examples):
        """
        few_shot_examples : list of dicts with keys 'image', 'text', 'label'
        """
        print(f"Loading Pixtral model from {model_repo_id} with Transformers...")
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_repo_id)
        
        self.system_prompt = (
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
        
        self.few_shot_examples = few_shot_examples
        print(f"Model loaded. {len(self.few_shot_examples)} few-shot examples will be used in each query.")
    
    def _build_messages(self, test_image, test_text):
        """
        Construct the full conversation messages including few‑shot examples and the test query.
        Returns a tuple (messages, images_list) where images_list is a list of PIL images
        in the order they appear (first all few‑shot images, then test image).
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        images_list = []

        # Add few‑shot exchanges
        for ex in self.few_shot_examples:
            # User turn with image and text
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Extracted text: ```{ex['text']}```\nProvide your answer in the format: hateful or not-hateful."},
                    {"type": "image"}   # image placeholder
                ]
            })
            images_list.append(ex["image"])

            # Assistant turn (only the label)
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex["label"]}]
            })

        # Final test user message
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Extracted text: ```{test_text}```\nProvide your answer in the format: hateful or not-hateful."},
                {"type": "image"}
            ]
        })
        images_list.append(test_image)

        return messages, images_list

    def predict(self, test_sample):
        """
        test_sample: dict with keys 'img', 'text', 'label' (label optional)
        Returns generated text (list with one element).
        """
        # Resolve FBHM test image path
        img_rel_path = test_sample['img']
        if img_rel_path.startswith("F") and "/" in img_rel_path:
            folder = img_rel_path.split("/")[0]
            img_name = img_rel_path.split("/")[-1]
            img_path = os.path.join(BASE_IMAGE_DIR_FBHM, folder, "memes", img_name)
        else:
            img_path = os.path.join(BASE_IMAGE_DIR_FBHM, img_rel_path)

        if not os.path.exists(img_path):
            base_name = os.path.splitext(img_path)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if os.path.exists(base_name + ext):
                    img_path = base_name + ext
                    break

        if not os.path.exists(img_path):
            print(f"Warning: Test image not found: {img_path}")
            return [f"Error: Image file not found: {img_path}"]

        try:
            test_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return [f"Error: Failed to load image - {e}"]

        test_text = test_sample["text"]

        # Build conversation messages and collect images
        messages, images_list = self._build_messages(test_image, test_text)

        try:
            # Apply chat template to get prompt string with image tokens
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            # Tokenize with the actual images (list of PIL images)
            inputs = self.processor(
                text=prompt,
                images=images_list,          # list of images in the same order as placeholders
                return_tensors="pt"
            )

            # Move inputs to model device and cast to model dtype
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            inputs = {
                k: v.to(self.model.dtype) if torch.is_floating_point(v) else v
                for k, v in inputs.items()
            }

            gen_kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": TEMPERATURE > 0,
                "temperature": TEMPERATURE if TEMPERATURE > 0 else None,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            if gen_kwargs["temperature"] is None:
                del gen_kwargs["temperature"]

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Extract only newly generated tokens
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

            return [generated_text]

        except Exception as e:
            print(f"Error during generation for image {img_rel_path}: {e}")
            return [f"Error: {e}"]

# ========== Main ==========
def main():
    # Load FBHM few‑shot examples from train set
    print("\nLoading FBHM training data and sampling few‑shot examples...")
    few_shot_examples = load_few_shot_examples(TRAIN_CSV_PATH, NUM_SHOTS, RANDOM_SEED)

    # Initialize inference model with few‑shot examples
    inference_model = PixtralVLInference(MODEL_REPO_ID, few_shot_examples)

    # Load FBHM test data
    print("\nLoading FBHM test data...")
    test_data = load_test_data(TEST_CSV_PATH)
    print(f"Loaded {len(test_data)} test samples.")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
        for idx, sample in tqdm(enumerate(test_data), total=len(test_data), desc="Processing memes"):
            # Write sample data as JSON
            output_file.write(json.dumps(sample, default=str))
            output_file.write('\n----------\n')

            # Run inference
            generated_texts = inference_model.predict(sample)
            print(generated_texts)

            # Write the assistant's response(s)
            for text in generated_texts:
                formatted_text = re.sub(r'\bassistant\b', 'Assistant:', text, flags=re.IGNORECASE)
                output_file.write(formatted_text + '\n')

            output_file.write("\n##########\n")

            # Periodic cleanup
            if (idx + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\nOutput file saved to '{OUTPUT_TXT}'")
    print(f"Processed {len(test_data)} memes with {NUM_SHOTS}‑shot in‑context learning (seed={RANDOM_SEED}).")

if __name__ == "__main__":
    main()