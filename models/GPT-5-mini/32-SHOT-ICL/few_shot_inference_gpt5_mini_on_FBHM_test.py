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
TEST_CSV_PATH = "../FBHM/test.csv"
TRAIN_CSV_PATH = "../FBHM/train.csv"
BASE_IMAGE_DIR = os.path.abspath("../FBHM")
OUTPUT_TXT = "./32_shot_gpt5mini_inference_on_FBHM_test_output.txt"

AZURE_ENDPOINT = "https://hate-vlm.openai.azure.com/"
AZURE_API_KEY = ""                      # Replace with your actual key
AZURE_DEPLOYMENT = "gpt-5-mini"          # Deployment name for GPT‑5‑mini
AZURE_API_VERSION = "2024-12-01-preview"

NUM_SHOTS = 32
RANDOM_SEED = 42

MAX_TOKENS = 5000
TEMPERATURE = 0.001
# ------------------------------------------------------------------------------

SYSTEM_PROMPT = (
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

def resolve_image_path(rel_path):
    if rel_path.startswith("F") and "/" in rel_path:
        folder = rel_path.split("/")[0]
        img_name = rel_path.split("/")[-1]
        path = os.path.join(BASE_IMAGE_DIR, folder, "memes", img_name)
    else:
        path = os.path.join(BASE_IMAGE_DIR, rel_path)

    if not os.path.exists(path):
        base_name = os.path.splitext(path)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if os.path.exists(base_name + ext):
                path = base_name + ext
                break
    return path if os.path.exists(path) else None

def load_fixed_few_shot_examples(train_csv_path, num_shots, seed):
    """
    Load all valid training examples, then select a fixed set using random.sample.
    """
    train_df = pd.read_csv(train_csv_path)
    valid_examples = []

    for _, row in train_df.iterrows():
        img_path = resolve_image_path(row['img'])
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

    def predict(self, meme_data):
        img_path = resolve_image_path(meme_data['img'])
        if img_path is None:
            return f"Error: Image not found: {meme_data['img']}"

        test_image_base64 = encode_image_to_base64(img_path)
        if test_image_base64 is None:
            return "Error: Failed to encode test image."

        extracted_text = meme_data["text"]

        # Build messages with the fixed few‑shot set
        messages = [{"role": "system", "content": self.system_prompt}]

        for ex in self.few_shot_examples:
            user_content = [
                {"type": "text", "text": f"Extracted text: ```{ex['text']}```\nProvide your answer in the format: hateful or not-hateful."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex['image_base64']}", "detail": "low"}}
            ]
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex['label']})

        # Test query
        test_user_content = [
            {"type": "text", "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: hateful or not-hateful."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{test_image_base64}", "detail": "low"}}
        ]
        messages.append({"role": "user", "content": test_user_content})

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.deployment,
                max_completion_tokens=MAX_TOKENS,
                # temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed: {e}")
            return f"Error: {e}"

def main():
    # Load the fixed set of few‑shot examples
    few_shot_examples = load_fixed_few_shot_examples(TRAIN_CSV_PATH, NUM_SHOTS, RANDOM_SEED)

    inferencer = GPT5MiniInference(
        endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        few_shot_examples=few_shot_examples
    )

    print("Loading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_data = test_df.to_dict('records')

    with open(OUTPUT_TXT, "w", encoding="utf-8") as out_f:
        for idx, meme_row in enumerate(tqdm(test_data, desc="Processing memes")):
            out_f.write(json.dumps(meme_row, default=str))
            out_f.write('\n----------\n')

            answer = inferencer.predict(meme_row)
            print(answer)
            out_f.write(answer)
            out_f.write("\n##########\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            time.sleep(0.5)

    print(f"\nOutput saved to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()