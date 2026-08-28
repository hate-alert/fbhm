import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd
from PIL import Image
import json
import re
import gc
from tqdm import tqdm
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

from transformers import LlavaForConditionalGeneration, AutoProcessor

# Configuration
TEST_CSV_PATH = "../FBHM/test.csv"
BASE_IMAGE_DIR = os.path.abspath("../FBHM")
# MODEL_REPO_ID = "mistral-community/pixtral-12b"
MODEL_REPO_ID = "pbhaskar/pixtral-12b-sft-fhm-then-FBHM-3-epochs"
# OUTPUT_TXT = "./pixtral_sft_MAMI_15_epochs_earlystop_2_zero_shot_inference_on_FBHM_test_output.txt"
OUTPUT_TXT = "./pixtral_sft_fhm_then_FBHM_3_epochs_zero_shot_inference_on_FBHM_test_output.txt"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.001

HF_TOKEN = ""
login(token=HF_TOKEN)

class PixtralVLInference:
    def __init__(self, model_repo_id):
        print(f"Loading Pixtral model from {model_repo_id} with Transformers...")
        
        # Load model with automatic device mapping (supports multiple GPUs)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_repo_id)
        
        # Store system prompt for later use
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
        
        print("Model loaded successfully.")
    
    def safe_image_load(self, image_path):
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', (224, 224), color='black')
    
    def predict(self, meme_data, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        img_rel_path = meme_data['img']
        
        # Build absolute image path (same logic as before)
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
        
        # Load image using PIL
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return [f"Error: Failed to load image - {e}"]
        
        extracted_text = meme_data["text"]
        
        # Build conversation messages (without actual images, just placeholders)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: hateful or not-hateful."},
                    {"type": "image"}   # image placeholder
                ]
            }
        ]
        
        try:
            # Apply chat template to get the prompt string with image tokens (e.g., [IMG])
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # Tokenize with the actual image (returns float32 tensors by default)
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            )
            
            # Move inputs to model device and cast to model dtype (bfloat16)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Convert all floating point tensors (like pixel_values) to model's dtype
            inputs = {
                k: v.to(self.model.dtype) if torch.is_floating_point(v) else v
                for k, v in inputs.items()
            }
            
            # Prepare generation parameters
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else None,
                "pad_token_id": self.processor.tokenizer.eos_token_id,  # suppress warning
            }
            if gen_kwargs["temperature"] is None:
                del gen_kwargs["temperature"]
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract only the newly generated tokens (exclude the input prompt)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            return [generated_text]
        
        except Exception as e:
            print(f"Error during generation for index {getattr(meme_data, 'name', 'unknown')}: {e}")
            return [f"Error: {e}"]

def main():
    inference_model = PixtralVLInference(MODEL_REPO_ID)
    
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_data = test_df.to_dict('records')
    
    with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
        for index, meme_data in tqdm(enumerate(test_data), total=len(test_data), desc="Processing memes"):
            meme_series = pd.Series(meme_data)
            meme_series.name = index
            
            # Build JSON with full image path (for logging)
            meme_dict = meme_series.to_dict()
            img_rel_path = meme_dict['img']
            if img_rel_path.startswith("F") and "/" in img_rel_path:
                folder = img_rel_path.split("/")[0]
                img_name = img_rel_path.split("/")[-1]
                full_path = os.path.join(BASE_IMAGE_DIR, folder, "memes", img_name)
            else:
                full_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)
            meme_dict['full_image_path'] = full_path
            
            # Write JSON block
            output_file.write(json.dumps(meme_dict, default=str))
            output_file.write('\n----------\n')
            
            # Write the user prompt (for reference)
            user_prompt = (
                f"User: {inference_model.system_prompt}\n\n"
                f"Extracted text: ```{meme_data['text']}```\n"
                f"Provide your answer in the format: hateful or not-hateful.\n\n"
                f"Assistant:"
            )
            output_file.write(user_prompt + '\n')
            
            # Run inference
            generated_texts = inference_model.predict(meme_series)
            
            print(generated_texts)
            
            for text in generated_texts:
                output_file.write(text)
                output_file.write('\n')
            
            output_file.write("\n##########\n")
            
            # Periodic cleanup
            if (index + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    print(f"\nOutput file saved to '{OUTPUT_TXT}'")
    print(f"\nProcessed {len(test_data)} memes")

if __name__ == "__main__":
    main()
    