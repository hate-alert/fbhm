import os
import base64
import pandas as pd
import json
from tqdm import tqdm
from openai import AzureOpenAI
from PIL import Image
import io
import time

# ------------------------------------------------------------------------------
# Configuration (update paths and keys as needed)
# ------------------------------------------------------------------------------
TEST_CSV_PATH = "../FBHM/test.csv"
BASE_IMAGE_DIR = os.path.abspath("../FBHM")
OUTPUT_TXT = "./gpt4.1mini_zero_shot_inference_on_FBHM_test_output.txt"

# Azure OpenAI settings
AZURE_ENDPOINT = "https://hate-vlm.openai.azure.com/"
AZURE_API_KEY = ""                      # Replace with your actual key
AZURE_DEPLOYMENT = "gpt-4.1-mini"           # Deployment name
AZURE_API_VERSION = "2024-12-01-preview"

# Generation parameters
MAX_TOKENS = 50
TEMPERATURE = 0.001                           # 0 for deterministic output
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
            # Convert to RGB if necessary (e.g., PNG with transparency)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        # Return a transparent 1x1 pixel as fallback (or raise)
        return None

class GPT4MiniInference:
    def __init__(self, endpoint, api_key, deployment, api_version):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        self.deployment = deployment
        self.system_prompt = SYSTEM_PROMPT

    def predict(self, meme_data):
        # Locate the image file (same logic as original)
        img_rel_path = meme_data['img']
        if img_rel_path.startswith("F") and "/" in img_rel_path:
            folder = img_rel_path.split("/")[0]
            img_name = img_rel_path.split("/")[-1]
            img_path = os.path.join(BASE_IMAGE_DIR, folder, "memes", img_name)
        else:
            img_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)

        if not os.path.exists(img_path):
            base_name = os.path.splitext(img_path)[0]
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if os.path.exists(base_name + ext):
                    img_path = base_name + ext
                    found = True
                    break
            if not found:
                return f"Error: Image not found: {img_rel_path}"

        # Encode image to base64
        image_base64 = encode_image_to_base64(img_path)
        if image_base64 is None:
            return "Error: Failed to encode image."

        extracted_text = meme_data["text"]

        # Build the message list
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
                # top_p=1.0,
                # frequency_penalty=0.0,
                # presence_penalty=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed: {e}")
            return f"Error: {e}"

def main():
    # Initialize inference object
    inferencer = GPT4MiniInference(
        endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION
    )

    print("Loading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_data = test_df.to_dict('records')

    with open(OUTPUT_TXT, "w", encoding="utf-8") as out_f:
        for idx, meme_row in enumerate(tqdm(test_data, desc="Processing memes")):
            # Build a dictionary with full image path for logging (similar to original)
            meme_dict = meme_row.copy()
            img_rel_path = meme_dict['img']
            if img_rel_path.startswith("F") and "/" in img_rel_path:
                folder = img_rel_path.split("/")[0]
                img_name = img_rel_path.split("/")[-1]
                full_path = os.path.join(BASE_IMAGE_DIR, folder, "memes", img_name)
            else:
                full_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)
            meme_dict['full_image_path'] = full_path

            # Write JSON block
            out_f.write(json.dumps(meme_dict, default=str))
            out_f.write('\n----------\n')

            # Write the user prompt (for reference)
            user_prompt = (
                f"User: {SYSTEM_PROMPT}\n\n"
                f"Extracted text: ```{meme_row['text']}```\n"
                f"Provide your answer in the format: hateful or not-hateful.\n\n"
                f"Assistant: "
            )
            out_f.write(user_prompt + '\n')

            # Run inference
            answer = inferencer.predict(meme_row)
            print(answer)
            out_f.write(answer + '\n')
            out_f.write("\n##########\n")

            # Optional: small delay to avoid hitting rate limits
            time.sleep(0.5)

    print(f"\nOutput saved to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()