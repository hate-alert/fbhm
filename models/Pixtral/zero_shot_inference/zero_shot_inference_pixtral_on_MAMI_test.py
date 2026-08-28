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

# ========== Configuration ==========
TEST_TSV_PATH = "/home/du1/21CS30035/mami-dataset/test.tsv"
IMAGE_DIR = "/home/du1/21CS30035/mami-dataset/images"
# MODEL_REPO_ID = "mistral-community/pixtral-12b"
MODEL_REPO_ID = "pbhaskar/pixtral-12b-sft-fhm-then-FBHM-3-epochs"
OUTPUT_TXT = "./pixtral_sft_fhm_then_FBHM_3_epochs_zero_shot_inference_on_MAMI_test_output.txt"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.001

HF_TOKEN = ""
login(token=HF_TOKEN)

# ========== Utility ==========
def load_tsv(file_path):
    """Load TSV file into pandas DataFrame."""
    return pd.read_csv(file_path, sep='\t')

# ========== Inference Class ==========
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
        
        # MAMI-specific system prompt
        self.system_prompt = (
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
        
        print("Model loaded successfully.")
    
    def predict(self, sample):
        """
        sample: dict with keys 'file_name', 'label', 'text' (and possibly others)
        Returns generated text (list with one element).
        """
        # Build full image path
        img_filename = sample['file_name']
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return [f"Error: Image file not found: {img_path}"]
        
        # Load image using PIL
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return [f"Error: Failed to load image - {e}"]
        
        extracted_text = sample["text"]
        
        # Build conversation messages (without actual images, just placeholders)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: misogynistic or not-misogynistic."},
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
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": TEMPERATURE > 0,
                "temperature": TEMPERATURE if TEMPERATURE > 0 else None,
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
            print(f"Error during generation for file {sample.get('file_name', 'unknown')}: {e}")
            return [f"Error: {e}"]

# ========== Main ==========
def main():
    inference_model = PixtralVLInference(MODEL_REPO_ID)
    
    print("\nLoading MAMI test data...")
    test_df = load_tsv(TEST_TSV_PATH)
    # Keep only necessary columns if desired, but we'll use all for JSON output
    # Convert to list of dicts
    test_data = test_df.to_dict('records')
    print(f"Loaded {len(test_data)} samples.")
    
    with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
        for idx, sample in tqdm(enumerate(test_data), total=len(test_data), desc="Processing memes"):
            # Enrich sample with full image path and string label
            img_filename = sample['file_name']
            full_image_path = os.path.join(IMAGE_DIR, img_filename)
            sample_with_path = sample.copy()
            sample_with_path['full_image_path'] = full_image_path
            # Convert numeric label to string for clarity
            sample_with_path['label_str'] = 'misogynistic' if sample['label'] == 1 else 'not-misogynistic'
            
            # Write JSON block
            output_file.write(json.dumps(sample_with_path, default=str))
            output_file.write('\n----------\n')
            
            # Write the user prompt (for reference)
            user_prompt = (
                f"User: {inference_model.system_prompt}\n\n"
                f"Extracted text: ```{sample['text']}```\n"
                f"Provide your answer in the format: misogynistic or not-misogynistic.\n\n"
                f"Assistant:"
            )
            output_file.write(user_prompt + '\n')
            
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
    print(f"Processed {len(test_data)} memes.")

if __name__ == "__main__":
    main()