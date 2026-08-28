import os
import base64
import pandas as pd
import json
from tqdm import tqdm
from openai import AzureOpenAI
from PIL import Image
import io
import time
import random

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
TEST_TSV_PATH = "/home/du1/21CS30035/mami-dataset/test.tsv"
IMAGE_DIR_MAMI = "/home/du1/21CS30035/mami-dataset/images"
TRAIN_CSV_PATH = "../FBHM/train.csv"                 # FBHM training samples
BASE_IMAGE_DIR_FBHM = os.path.abspath("../FBHM")     # base for FBHM images

OUTPUT_TXT = "./32_shot_gpt5mini_inference_on_MAMI_test_output.txt"

AZURE_ENDPOINT = "https://hate-vlm.openai.azure.com/"
AZURE_API_KEY = ""                      # Replace with your actual key
AZURE_DEPLOYMENT = "gpt-5-mini"          # GPT‑5‑mini deployment name
AZURE_API_VERSION = "2024-12-01-preview"

NUM_SHOTS = 32
RANDOM_SEED = 42

MAX_TOKENS = 5000                        # Increased for reasoning model
# No temperature for GPT‑5‑mini (or we omit it)
# ------------------------------------------------------------------------------

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

SYSTEM_PROMPT = (
    "Consider the following definitions.\n"
    f"1. misogynistic - {DEFINITION_MISOGYNISTIC}\n"
    f"2. not-misogynistic - {DEFINITION_NOT_MISOGYNISTIC}\n\n"
    "Classify the above meme as misogynistic or not-misogynistic based on these definitions, "
    "considering the image and the extracted text from the image delimited by three backticks.\n"
    "Provide your answer in the format: misogynistic or not-misogynistic."
)

def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def resolve_fbhm_image_path(rel_path):
    """Resolve FBHM image path (used for few‑shot examples)."""
    if rel_path.startswith("F") and "/" in rel_path:
        folder = rel_path.split("/")[0]
        img_name = rel_path.split("/")[-1]
        path = os.path.join(BASE_IMAGE_DIR_FBHM, folder, "memes", img_name)
    else:
        path = os.path.join(BASE_IMAGE_DIR_FBHM, rel_path)

    if not os.path.exists(path):
        base_name = os.path.splitext(path)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if os.path.exists(base_name + ext):
                path = base_name + ext
                break
    return path if os.path.exists(path) else None

def resolve_mami_image_path(file_name):
    """Resolve MAMI image path for test images."""
    img_path = os.path.join(IMAGE_DIR_MAMI, file_name)
    if not os.path.exists(img_path):
        base_name = os.path.splitext(img_path)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if os.path.exists(base_name + ext):
                img_path = base_name + ext
                break
    return img_path if os.path.exists(img_path) else None

def load_fixed_few_shot_examples(train_csv_path, num_shots, seed):
    """
    Load all valid FBHM training examples, then select a fixed set using random.sample.
    """
    train_df = pd.read_csv(train_csv_path)
    valid_examples = []

    for _, row in train_df.iterrows():
        img_path = resolve_fbhm_image_path(row['img'])
        if img_path is None:
            continue
        image_base64 = encode_image_to_base64(img_path)
        if image_base64 is None:
            continue
        label_str = "hateful" if row['label'] == 1 else "not-hateful"
        valid_examples.append({
            'image_base64': image_base64,
            'text': row['text'],
            'label': label_str
        })

    if len(valid_examples) < num_shots:
        print(f"Warning: Only {len(valid_examples)} valid examples found, using all.")
        return valid_examples

    random.seed(seed)
    sampled = random.sample(valid_examples, num_shots)
    print(f"Selected {len(sampled)} fixed few‑shot examples (seed={seed}).")
    return sampled

def load_mami_test_data(tsv_path):
    """Load MAMI test TSV and return list of dicts with required fields."""
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

class GPT5MiniInference:
    def __init__(self, endpoint, api_key, deployment, api_version, few_shot_examples):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        self.deployment = deployment
        self.system_prompt = SYSTEM_PROMPT
        self.few_shot_examples = few_shot_examples

    def predict(self, sample):
        """
        sample: dict from MAMI test with keys: file_name, text, label, etc.
        """
        img_path = resolve_mami_image_path(sample['file_name'])
        if img_path is None:
            return f"Error: Image not found: {sample['file_name']}"

        test_image_base64 = encode_image_to_base64(img_path)
        if test_image_base64 is None:
            return "Error: Failed to encode test image."

        extracted_text = sample["text"]

        # Build messages with the fixed few‑shot set
        messages = [{"role": "system", "content": self.system_prompt}]

        for ex in self.few_shot_examples:
            user_content = [
                {"type": "text", "text": f"Extracted text: ```{ex['text']}```\nProvide your answer in the format: misogynistic or not-misogynistic."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex['image_base64']}", "detail": "low"}}
            ]
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex['label']})  # FBHM labels: hateful/not-hateful

        # Test query
        test_user_content = [
            {"type": "text", "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: misogynistic or not-misogynistic."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{test_image_base64}", "detail": "low"}}
        ]
        messages.append({"role": "user", "content": test_user_content})

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.deployment,
                max_completion_tokens=MAX_TOKENS,
                # temperature is omitted for GPT‑5‑mini
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed: {e}")
            return f"Error: {e}"

def main():
    # Load the fixed set of few‑shot examples from FBHM train
    few_shot_examples = load_fixed_few_shot_examples(TRAIN_CSV_PATH, NUM_SHOTS, RANDOM_SEED)

    inferencer = GPT5MiniInference(
        endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        few_shot_examples=few_shot_examples
    )

    print(f"Loading MAMI test data from {TEST_TSV_PATH}...")
    test_data = load_mami_test_data(TEST_TSV_PATH)
    print(f"Loaded {len(test_data)} test samples.")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(tqdm(test_data, desc="Processing memes")):
            # Write sample data as JSON (include resolved image path for reference)
            img_path = resolve_mami_image_path(sample['file_name'])
            sample_with_path = sample.copy()
            sample_with_path['full_image_path'] = img_path
            out_f.write(json.dumps(sample_with_path, default=str))
            out_f.write('\n----------\n')

            # Run inference
            answer = inferencer.predict(sample)
            print(answer)
            out_f.write(answer)
            out_f.write("\n##########\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            time.sleep(0.5)

    print(f"\nOutput saved to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()