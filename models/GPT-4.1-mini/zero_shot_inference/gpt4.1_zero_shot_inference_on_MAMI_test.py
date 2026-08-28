import os
import base64
import pandas as pd
import json
import time
import re
from tqdm import tqdm
from openai import AzureOpenAI
from PIL import Image
import io

# ------------------------------------------------------------------------------
# Configuration – update paths and API key
# ------------------------------------------------------------------------------
TEST_TSV_PATH = "/home/du1/21CS30035/mami-dataset/test.tsv"
IMAGE_DIR = "/home/du1/21CS30035/mami-dataset/images"
OUTPUT_TXT = "./gpt4.1mini_zero_shot_inference_on_MAMI_test_output.txt"

# Azure OpenAI settings
AZURE_ENDPOINT = "https://hate-vlm.openai.azure.com/"
AZURE_API_KEY = ""                      # <-- Replace with your actual key
AZURE_DEPLOYMENT = "gpt-4.1-mini"        # Your deployment name
AZURE_API_VERSION = "2024-12-01-preview"

# Generation parameters
MAX_TOKENS = 50
TEMPERATURE = 0.001                      # Near‑deterministic output
# ------------------------------------------------------------------------------

# MAMI‑specific system prompt
SYSTEM_PROMPT = (
    "Consider the following definitions.\n"
    "1. misogynistic - a meme is misogynous if it conceptually describes "
    "an offensive, sexist or hateful scene (weak or strong, implicitly or explicitly) "
    "having as target a woman or a group of women. Misogyny can be expressed "
    "in the form of shaming, stereotype, objectification and/or violence.\n"
    "2. not-misogynistic - a meme that does not express any form of hate against women.\n"
    "Classify the above meme as misogynistic or not-misogynistic based on the above definitions considering the image "
    "and the extracted text from the image delimited by three backticks.\n"
    "Provide your answer in the format: misogynistic or not-misogynistic."
)

def encode_image_to_base64(image_path):
    """Open an image with PIL and return a base64 string (JPEG format)."""
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

def load_tsv(file_path):
    """Load TSV file into pandas DataFrame."""
    return pd.read_csv(file_path, sep='\t')

class GPT4MiniInference:
    def __init__(self, endpoint, api_key, deployment, api_version, base_image_dir):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        self.deployment = deployment
        self.base_image_dir = base_image_dir
        self.system_prompt = SYSTEM_PROMPT

    def predict(self, sample):
        """
        sample: dict with keys 'file_name', 'label', 'text' (and possibly others)
        Returns generated answer string.
        """
        # Construct full image path
        img_filename = sample['file_name']
        img_path = os.path.join(self.base_image_dir, img_filename)

        if not os.path.exists(img_path):
            return f"Error: Image not found: {img_path}"

        # Encode image to base64
        image_base64 = encode_image_to_base64(img_path)
        if image_base64 is None:
            return "Error: Failed to encode image."

        extracted_text = sample["text"]

        # Build message list for Azure OpenAI
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: misogynistic or not-misogynistic."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.deployment,
                max_completion_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            answer = response.choices[0].message.content.strip()
            # Optional: post-process to ensure clean output (remove extra punctuation, lowercase, etc.)
            # but we'll keep as is.
            return answer
        except Exception as e:
            print(f"API call failed: {e}")
            return f"Error: {e}"

def main():
    # Initialize inference object
    inferencer = GPT4MiniInference(
        endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        base_image_dir=IMAGE_DIR
    )

    print("Loading MAMI test data...")
    test_df = load_tsv(TEST_TSV_PATH)
    test_data = test_df.to_dict('records')
    print(f"Loaded {len(test_data)} samples.")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(tqdm(test_data, desc="Processing memes")):
            # Enrich sample with full image path and a string label for clarity
            img_filename = sample['file_name']
            full_image_path = os.path.join(IMAGE_DIR, img_filename)
            sample_with_path = sample.copy()
            sample_with_path['full_image_path'] = full_image_path
            sample_with_path['label_str'] = 'misogynistic' if sample['label'] == 1 else 'not-misogynistic'

            # Write JSON block
            out_f.write(json.dumps(sample_with_path, default=str))
            out_f.write('\n----------\n')

            # Write the user prompt (for reference)
            user_prompt = (
                f"User: {SYSTEM_PROMPT}\n\n"
                f"Extracted text: ```{sample['text']}```\n"
                f"Provide your answer in the format: misogynistic or not-misogynistic.\n\n"
                f"Assistant:"
            )
            out_f.write(user_prompt + '\n')

            # Run inference
            answer = inferencer.predict(sample)
            print(answer)                # (optional) show in console
            out_f.write(answer + '\n')
            out_f.write("\n##########\n")

            # Small delay to avoid hitting rate limits
            time.sleep(0.5)

    print(f"\nOutput saved to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()