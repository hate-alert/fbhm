import os
import torch
import pandas as pd
from PIL import Image
import json
import re
import gc
import random
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
TEST_TSV_PATH = "/home/du1/21CS30035/mami-dataset/test.tsv"
IMAGE_DIR_MAMI = "/home/du1/21CS30035/mami-dataset/images"
TRAIN_CSV_PATH = "../FBHM/train.csv"                       # FBHM training samples
BASE_IMAGE_DIR_FBHM = "../FBHM"                            # base for FBHM images

MODEL_REPO_ID = "OpenGVLab/InternVL3_5-8B-HF"
# Uncomment the following lines to use a LoRA‑adapted model
# ADAPTER_REPO_ID = "pbhaskar/internvl3_5-8b-normal-then-FBHM-3-epochs"
# ADAPTER_SUBDIR  = "best"
OUTPUT_TXT = "./32_shot_internvl_normal_inference_on_MAMI_test_output.txt"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.001

# Few‑shot settings
NUM_SHOTS = 32
RANDOM_SEED = 42

HF_TOKEN = ""
# login(token=HF_TOKEN)

# MAMI‑specific definitions
DEFINITION_MISOGYNISTIC = (
    "a meme is misogynous if it conceptually describes "
    "an offensive, sexist or hateful scene (weak or strong, implicitly or explicitly) "
    "having as target a woman or a group of women. Misogyny can be expressed "
    "in the form of shaming, stereotype, objectification and/or violence."
)
DEFINITION_NOT_MISOGYNISTIC = (
    "a meme that does not express any form of hate against women."
)

# ========== Utility: Load Few‑Shot Examples from FBHM ==========
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

def load_mami_test_data(tsv_path):
    """Load MAMI test TSV and return list of dicts."""
    df = pd.read_csv(tsv_path, sep='\t')
    data = []
    for _, row in df.iterrows():
        data.append({
            'file_name': row['file_name'],
            'label': int(row['label']),
            'text': row['text'],
            'shaming': int(row['shaming']),
            'stereotype': int(row['stereotype']),
            'objectification': int(row['objectification']),
            'violence': int(row['violence'])
        })
    return data

# ========== Inference Class with Few‑Shot ==========
class InternVLInference:
    def __init__(self, model_repo_id, few_shot_examples, adapter_repo_id=None):
        """
        few_shot_examples : list of dicts with keys 'image', 'text', 'label'
        adapter_repo_id   : optional Hugging Face repo ID for LoRA adapter
        """
        print(f"Loading base model from {model_repo_id}...")
        self.processor = AutoProcessor.from_pretrained(
            model_repo_id,
            trust_remote_code=True,
            do_image_splitting=False
        )
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Optionally load LoRA adapter
        if adapter_repo_id:
            print(f"Loading LoRA adapter from {adapter_repo_id}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_repo_id,
                subfolder=ADAPTER_SUBDIR,
                is_trainable=False
            ).eval()
            print("LoRA adapter attached.")
        
        self.model.eval()
        self.device = self.model.device
        print(f"Model loaded on device: {self.device}")

        # Store few‑shot examples
        self.few_shot_examples = few_shot_examples
        print(f"Model ready. {len(self.few_shot_examples)} few-shot examples will be used.")

    def safe_image_load(self, image_path):
        """Safely load image with error handling"""
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', (224, 224), color='black')

    def _build_messages(self, test_image, test_text):
        """
        Construct the full conversation messages including few‑shot examples and the test query.
        Returns a tuple (messages, images_list) where images_list is a list of PIL images
        in the order they appear (first all few‑shot images, then test image).
        """
        messages = []
        images_list = []

        # System prompt with MAMI definitions
        system_prompt = (
            "Consider the following definitions.\n"
            f"1. misogynistic - {DEFINITION_MISOGYNISTIC}\n"
            f"2. not-misogynistic - {DEFINITION_NOT_MISOGYNISTIC}\n\n"
            "Classify the above meme as misogynistic or not-misogynistic based on these definitions, "
            "considering the image and the extracted text from the image delimited by three backticks.\n"
            "Provide your answer in the format: misogynistic or not-misogynistic."
        )
        messages.append({"role": "system", "content": system_prompt})

        # Add few‑shot exchanges (using FBHM examples, but the user question asks for misogynistic/not-misogynistic)
        for ex in self.few_shot_examples:
            # User turn with image and text – note: we ask for misogynistic/not-misogynistic, even though the label in assistant is hateful/not-hateful
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text", "text": f"Extracted text: ```{ex['text']}```\nProvide your answer in the format: misogynistic or not-misogynistic."}
                ]
            })
            images_list.append(ex["image"])

            # Assistant turn (only the label, which is "hateful" or "not-hateful")
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex["label"]}]
            })

        # Final test user message
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": f"Extracted text: ```{test_text}```\nProvide your answer in the format: misogynistic or not-misogynistic."}
            ]
        })
        images_list.append(test_image)

        return messages, images_list

    def predict(self, test_sample):
        """
        test_sample: dict from MAMI test with keys 'file_name', 'text', 'label', etc.
        Returns generated text (list with one element).
        """
        # Resolve MAMI test image path
        img_filename = test_sample['file_name']
        img_path = os.path.join(IMAGE_DIR_MAMI, img_filename)

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
            test_image = self.safe_image_load(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return [f"Error: Failed to load image - {e}"]

        test_text = test_sample["text"]

        # Build conversation messages and collect images
        messages, images_list = self._build_messages(test_image, test_text)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize with all images
        inputs = self.processor(
            text=[text],
            images=images_list,          # list of PIL images in correct order
            padding=True,
            return_tensors="pt",
            do_image_splitting=False
        ).to(self.device)

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=TEMPERATURE > 0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            # Decode only the newly generated tokens
            input_length = inputs['input_ids'].shape[1]
            new_tokens = generated_ids[:, input_length:]
            generated_texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True)

            return generated_texts

        except Exception as e:
            print(f"Error during generation for file {img_filename}: {e}")
            return [f"Error: {e}"]

# ========== Main ==========
def main():
    # Load FBHM few‑shot examples from train set
    print("\nLoading FBHM training data and sampling few‑shot examples...")
    few_shot_examples = load_few_shot_examples(TRAIN_CSV_PATH, NUM_SHOTS, RANDOM_SEED)

    # Initialize inference model (optionally with adapter)
    inference_model = InternVLInference(
        MODEL_REPO_ID,
        few_shot_examples,
        # adapter_repo_id=ADAPTER_REPO_ID   # uncomment if using adapter
    )

    # Load MAMI test data
    print("\nLoading MAMI test data...")
    test_data = load_mami_test_data(TEST_TSV_PATH)
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