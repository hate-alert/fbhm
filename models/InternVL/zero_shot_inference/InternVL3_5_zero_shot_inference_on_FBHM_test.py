import os
import torch
import pandas as pd
from PIL import Image
import json
import re
import gc
from tqdm import tqdm
from transformers import AutoProcessor,AutoModelForImageTextToText
from peft import PeftModel, PeftConfig
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

# Configuration
TEST_CSV_PATH = "../FBHM/test.csv"
BASE_IMAGE_DIR = "../FBHM"
MODEL_REPO_ID = "OpenGVLab/InternVL3_5-8B-HF"
ADAPTER_REPO_ID = "pbhaskar/internvl3_5-8b-normal-then-FBHM-3-epochs"
# ADAPTER_SUBDIR  = "best"
OUTPUT_TXT = "./InternVL3_5_normal_then_FBHM_3_epochs_inference_on_FBHM_test_output.txt"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.001

HF_TOKEN = ""
login(token=HF_TOKEN)

class InternVLInference:
    def __init__(self, model_repo_id):
        print(f"Loading model from {model_repo_id}...")
        # print(f"Loading adapter from {ADAPTER_REPO_ID}/{ADAPTER_SUBDIR}...")
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
        self.model.eval()
        
        self.model=PeftModel.from_pretrained(
            self.model,
            ADAPTER_REPO_ID,
            # subfolder=ADAPTER_SUBDIR,
            is_trainable=False,
        ).eval()

        self.device = self.model.device
        print("LoRA attached?", any("lora" in n.lower() for n, _ in self.model.named_modules()))
        print(f"Model loaded on device: {self.device}")
    
    def safe_image_load(self, image_path):
        """Safely load image with error handling"""
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def predict(self, meme_data, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        """Generate prediction for a single meme"""
        # Extract image path and handle FBHM folder structure
        img_rel_path = meme_data['img']
        
        # Handle different path formats
        if img_rel_path.startswith("F") and "/" in img_rel_path:
            # Format like "F1/memes/0061.jpg"
            folder = img_rel_path.split("/")[0]  # e.g., "F1"
            img_name = img_rel_path.split("/")[-1]  # e.g., "0061.jpg"
            img_path = os.path.join(BASE_IMAGE_DIR, folder, "memes", img_name)
        else:
            # Try direct path
            img_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)
        
        # Check if file exists with various extensions
        if not os.path.exists(img_path):
            # Try with different extensions
            base_name = os.path.splitext(img_path)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if os.path.exists(base_name + ext):
                    img_path = base_name + ext
                    break
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path} (original: {img_rel_path})")
            return [f"Error: Image file not found: {img_path}"]
        
        image = self.safe_image_load(img_path)
        extracted_text = meme_data["text"]
        
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
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: hateful or not-hateful."}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
            do_image_splitting=False
        ).to(self.device)

        generated_texts = ""
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    temperature=temperature, 
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
        except Exception as e:
            print(f"Error during generation for index {meme_data.name}: {e}")
            generated_texts = [f"Error: {e}"]
        
        return generated_texts

def main():
    inference_model = InternVLInference(MODEL_REPO_ID)
    
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    test_data = test_df.to_dict('records')
    
    # Run predictions and log to text file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
        for index, meme_data in tqdm(enumerate(test_data), total=len(test_data), desc="Processing memes"):
            meme_series = pd.Series(meme_data)
            meme_series.name = index  # Set the index as name
            
            meme_dict = meme_series.to_dict()
            
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
            generated_texts = inference_model.predict(meme_series, 
                                                    max_new_tokens=MAX_NEW_TOKENS, 
                                                    temperature=TEMPERATURE)

            print(generated_texts)

            for text in generated_texts:
                # Replace 'assistant' (case-insensitive) with 'Assistant:'
                formatted_text = re.sub(r'\bassistant\b', 'Assistant:', text, flags=re.IGNORECASE)
                output_file.write(formatted_text)
                output_file.write('\n')
            
            output_file.write("\n##########\n")
            
            # garbage collection for long runs
            if (index + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    print(f"\nOutput file saved to '{OUTPUT_TXT}'")
    
    print(f"\nProcessed {len(test_data)} memes")
    print(f"Output format:")
    print("  1. Meme data in JSON format")
    print("  2. Separator: ----------")
    print("  3. Generated response(s)")
    print("  4. Separator: ##########")

if __name__ == "__main__":
    main()
