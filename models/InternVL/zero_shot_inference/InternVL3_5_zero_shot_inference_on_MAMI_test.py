import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import pandas as pd
from PIL import Image
import json
import re
import gc
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, PeftConfig
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

TEST_TSV_PATH = "/home/du1/21CS30035/mami-dataset/test.tsv"
IMAGE_DIR = "/home/du1/21CS30035/mami-dataset/images"
MODEL_REPO_ID = "OpenGVLab/InternVL3_5-8B-HF"
ADAPTER_REPO_ID = "pbhaskar/internvl3_5-8b-normal-then-FBHM-3-epochs"
# ADAPTER_SUBDIR  = "best"
# MODEL_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_TXT = "./InternVL3_5_normal_then_FBHM_3_epochs_inference_on_MAMI_test_output.txt"
# OUTPUT_TXT = "normal_qwen3_misogynistic_inference_on_MAMI_test_output.txt"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.001

HF_TOKEN = ""
login(token=HF_TOKEN)

# Define the specific definitions
definition_misogynistic = (
    "a meme is misogynous if it conceptually describes "
    "an offensive, sexist or hateful scene (weak or strong, implicitly or explicitly) "
    "having as target a woman or a group of women. Misogyny can be expressed "
    "in the form of shaming, stereotype, objectification and/or violence."
)

definition_not_misogynistic = (
    "a meme that does not express any form of hate against women."
)

class InternVL3Inference:
    def __init__(self, model_repo_id):
        print(f"Loading model from {model_repo_id}...")
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
        # Extract image path from MAMI data
        img_filename = meme_data['file_name']
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        # Check if file exists with various extensions
        if not os.path.exists(img_path):
            # Try with different extensions if the file doesn't exist
            base_name = os.path.splitext(img_path)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if os.path.exists(base_name + ext):
                    img_path = base_name + ext
                    break
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return [f"Error: Image file not found: {img_path}"]
        
        image = self.safe_image_load(img_path)
        extracted_text = meme_data["text"]
        
        messages = [
            {
                "role": "system", 
                "content": (
                    f"Consider the following definitions.\n"
                    f"1. misogynistic - {definition_misogynistic}\n"
                    f"2. not-misogynistic - {definition_not_misogynistic}\n"
                    f"Classify the above meme as misogynistic or not-misogynistic based on the above definitions considering the image "
                    f"and the extracted text from the image delimited by three backticks.\n"
                    f"Provide your answer in the format: misogynistic or not-misogynistic."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: misogynistic or not-misogynistic."}
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
            print(f"Error during generation for file {meme_data.get('file_name', 'unknown')}: {e}")
            generated_texts = [f"Error: {e}"]
        
        return generated_texts

def load_mami_tsv(file_path):
    """Load MAMI dataset from TSV file"""
    df = pd.read_csv(file_path, sep='\t')
    
    # Convert to list of dictionaries
    data = []
    for _, row in df.iterrows():
        data.append(row.to_dict())
    
    return data

def main():
    inference_model = InternVL3Inference(MODEL_REPO_ID)
    
    print("\nLoading MAMI test data from TSV...")
    test_data = load_mami_tsv(TEST_TSV_PATH)
    
    # Run predictions and log to text file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
        for index, meme_data in tqdm(enumerate(test_data), total=len(test_data), desc="Processing memes"):
            output_data = {
                'file_name': meme_data['file_name'],
                'label': int(meme_data['label']),
                'text': meme_data['text'],
                'shaming': int(meme_data['shaming']),
                'stereotype': int(meme_data['stereotype']),
                'objectification': int(meme_data['objectification']),
                'violence': int(meme_data['violence'])
            }
            output_file.write(json.dumps(output_data))
            output_file.write('\n----------\n')
            
            # Run inference
            generated_texts = inference_model.predict(meme_data, 
                                                    max_new_tokens=MAX_NEW_TOKENS, 
                                                    temperature=TEMPERATURE)

            print(f"File: {meme_data['file_name']}, Generated: {generated_texts}")

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
    
    # Print summary statistics
    print(f"\nProcessed {len(test_data)} memes")
    print(f"Output format:")
    print("  1. Meme data in JSON format")
    print("  2. Separator: ----------")
    print("  3. Generated response(s)")
    print("  4. Separator: ##########")

if __name__ == "__main__":
    main()
