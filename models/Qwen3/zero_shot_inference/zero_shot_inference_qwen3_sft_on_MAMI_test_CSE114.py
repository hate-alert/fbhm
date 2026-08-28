import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import torch
import pandas as pd
from PIL import Image
import json
import re
import gc
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

# Updated Configuration for MAMI dataset
TEST_TSV_PATH = "/home/du1/21CS30035/mami-dataset/test.tsv"
IMAGE_DIR = "/home/du1/21CS30035/mami-dataset/images"
MODEL_REPO_ID = "pbhaskar/qwen3-vl-8b-then-FBHM-3-epochs"
# MODEL_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_TXT = "./qwen3_normal_then_FBHM_3_epochs_zero_shot_inference_on_MAMI_test_output.txt"
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

class QwenVLInference:
    def __init__(self, model_repo_id):
        print(f"Loading model from {model_repo_id}...")
        self.processor = AutoProcessor.from_pretrained(
            model_repo_id,
            trust_remote_code=True,
            do_image_splitting=False
        )
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        self.device = self.model.device
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
        
        # Load image
        image = self.safe_image_load(img_path)
        extracted_text = meme_data["text"]
        
        # Use the provided definitions for misogynistic/not-misogynistic
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

        # Apply Qwen3-VL chat template
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
    # Initialize the inference model
    inference_model = QwenVLInference(MODEL_REPO_ID)
    
    # Load test data from TSV file
    print("\nLoading MAMI test data from TSV...")
    test_data = load_mami_tsv(TEST_TSV_PATH)
    
    # Run predictions and log to text file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
        for index, meme_data in tqdm(enumerate(test_data), total=len(test_data), desc="Processing memes"):
            # Write meme data as JSON (with some processing to handle non-serializable types)
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
            
            # Optional garbage collection for long runs
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
    
    # Create CSV with predictions for easier evaluation
    print("\nCreating summary CSV file...")
    results = []
    with open(OUTPUT_TXT, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = content.split('##########\n')
    for section in sections:
        if section.strip():
            lines = section.strip().split('\n')
            if len(lines) >= 3:
                try:
                    data_line = lines[0]
                    prediction_line = lines[2] if len(lines) > 2 else ""
                    
                    # Parse the JSON data
                    data = json.loads(data_line)
                    
                    # Extract prediction - look for misogynistic/not-misogynistic
                    prediction = "ERROR"
                    prediction_lower = prediction_line.lower()
                    
                    # Check for both spellings (misogynous/misogynistic)
                    if ("misogynistic" in prediction_lower and "not-misogynistic" not in prediction_lower) or \
                       ("misogynous" in prediction_lower and "not-misogynous" not in prediction_lower):
                        prediction = "misogynistic"
                    elif ("not-misogynistic" in prediction_lower) or ("not-misogynous" in prediction_lower):
                        prediction = "not-misogynistic"
                    
                    # Map to binary label for MAMI (0=not-misogynistic, 1=misogynistic)
                    binary_pred = 1 if prediction == "misogynistic" else (0 if prediction == "not-misogynistic" else -1)
                    
                    results.append({
                        'file_name': data['file_name'],
                        'true_label': data['label'],
                        'prediction': prediction,
                        'binary_prediction': binary_pred,
                        'text': data['text']
                    })
                except Exception as e:
                    print(f"Error parsing section: {e}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_output = OUTPUT_TXT.replace('.txt', '_predictions.csv')
    results_df.to_csv(csv_output, index=False)
    print(f"Predictions CSV saved to: {csv_output}")
    
    # Calculate and print accuracy
    if len(results) > 0:
        correct = sum(1 for r in results if r['true_label'] == r['binary_prediction'] and r['binary_prediction'] != -1)
        total = sum(1 for r in results if r['binary_prediction'] != -1)
        if total > 0:
            accuracy = correct / total * 100
            print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()