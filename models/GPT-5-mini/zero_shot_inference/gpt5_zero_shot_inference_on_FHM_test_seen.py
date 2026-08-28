import os
import base64
import json
import time
from tqdm import tqdm
from openai import AzureOpenAI
from PIL import Image
import io

# ------------------------------------------------------------------------------
# Configuration – update paths and API key
# ------------------------------------------------------------------------------
TEST_JSONL_PATH = "/home/du1/21CS30035/FacebookHatefulMemes/hateful_memes/test_seen.jsonl"
IMAGE_DIR = "/home/du1/21CS30035/FacebookHatefulMemes/hateful_memes/img"
OUTPUT_TXT = "./gpt5mini_zero_shot_inference_on_FHM_test_seen_output.txt"

# Azure OpenAI settings
AZURE_ENDPOINT = "https://hate-vlm.openai.azure.com/"
AZURE_API_KEY = ""                      # <-- Replace with your actual key
AZURE_DEPLOYMENT = "gpt-5-mini"          # Your deployment name
AZURE_API_VERSION = "2024-12-01-preview"

# Generation parameters
MAX_TOKENS = 500
TEMPERATURE = 0.001                      # Low temperature for deterministic output
# ------------------------------------------------------------------------------

# System prompt (same as used for Pixtral)
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

def load_jsonl(file_path):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

class GPT5MiniInference:
    def __init__(self, endpoint, api_key, deployment, api_version):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        self.deployment = deployment
        self.system_prompt = SYSTEM_PROMPT

    def predict(self, sample):
        """
        sample: dict with keys 'id', 'img', 'label', 'text'
        Returns generated answer string.
        """
        # Construct full image path: IMAGE_DIR / basename(sample['img'])
        img_rel = sample['img']                     # e.g. "img/12345.png"
        img_filename = os.path.basename(img_rel)    # "12345.png"
        img_path = os.path.join(IMAGE_DIR, img_filename)

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
                        "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: hateful or not-hateful."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"      # Use low detail to save tokens
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
                # temperature=TEMPERATURE,        # Uncomment if you want to set temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed: {e}")
            return f"Error: {e}"

def main():
    # Initialize inference object
    inferencer = GPT5MiniInference(
        endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION
    )

    print("Loading test_seen data...")
    test_data = load_jsonl(TEST_JSONL_PATH)
    print(f"Loaded {len(test_data)} samples.")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(tqdm(test_data, desc="Processing memes")):
            # Enrich sample with full image path for logging
            img_filename = os.path.basename(sample['img'])
            full_image_path = os.path.join(IMAGE_DIR, img_filename)
            sample_with_path = sample.copy()
            sample_with_path['full_image_path'] = full_image_path

            # Write JSON block
            out_f.write(json.dumps(sample_with_path, default=str))
            out_f.write('\n----------\n')

            # Write the user prompt (for reference)
            user_prompt = (
                f"User: {SYSTEM_PROMPT}\n\n"
                f"Extracted text: ```{sample['text']}```\n"
                f"Provide your answer in the format: hateful or not-hateful.\n\n"
                f"Assistant:"
            )
            out_f.write(user_prompt + '\n')

            # Run inference
            answer = inferencer.predict(sample)
            print(answer)                     # (optional) show in console
            out_f.write(answer + '\n')
            out_f.write("\n##########\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            # Small delay to avoid hitting rate limits
            time.sleep(0.5)

    print(f"\nOutput saved to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()