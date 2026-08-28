#!/usr/bin/env python3
"""
LIVE (Learnable In-Context Vector) for Qwen3-VL Hate Meme Classification
Implementation based on: "LIVE: Learnable In-Context Vector for Visual Question Answering"

Modified to evaluate on FHM test_seen.jsonl with proper ID handling.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import json
import re
import gc
from tqdm import tqdm
import warnings
from typing import List, Dict, Tuple, Optional
import random

warnings.filterwarnings('ignore')

from huggingface_hub import login
HF_TOKEN = ""
login(token=HF_TOKEN)

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_linear_schedule_with_warmup

# ===================== CONFIGURATION (modified for FHM) =====================
TEST_JSONL_PATH = "/home/du1/21CS30035/FacebookHatefulMemes/hateful_memes/test_seen.jsonl"
TRAIN_CSV_PATH = "../FBHM/train.csv"                     # unchanged (if needed)
BASE_IMAGE_DIR = "/home/du1/21CS30035/FacebookHatefulMemes/hateful_memes"   # parent of "img"
MODEL_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
# MODEL_REPO_ID = "pbhaskar/qwen3-vl-8b-sft-mami-15-epochs-earlystop-2"
LIVE_OUTPUT_PATH = "./learnable_ICVs_new_qwen3_normal_500_samples_random_15_epochs.pt"
OUTPUT_TXT = "./learnable_ICVs_new_qwen3_normal_500_samples_random_15_epochs_intervention_on_FHM_test_seen.txt"

MAX_NEW_TOKENS = 50
TEMPERATURE = 0.001
NUM_TRAIN_SAMPLES = 500      # only relevant if retraining
NUM_SHOTS_TRAIN = 32
BATCH_SIZE = 2
LEARNING_RATE_V = 1e-3
LEARNING_RATE_ALPHA = 1e-2
LAMBDA = 0.5
EPOCHS = 15

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

# Helper functions (unchanged)
def check_if_string_contains_a_list_of_words(test_string, word_list):
    test_string_lower = test_string.lower()
    return any(word in test_string_lower for word in word_list)

def extract_assistant_response(generated_text):
    formatted_text = re.sub(r'\bassistant\b', 'Assistant:', generated_text, flags=re.IGNORECASE)
    parts = formatted_text.split("Assistant:", 1)
    if len(parts) == 2:
        verdict = parts[1].strip()
        if verdict.endswith('.'):
            verdict = verdict[:-1]
        return verdict
    else:
        return formatted_text.strip()

def extract_label_from_response(response):
    ambiguous_cases = [
        "hateful or not-hateful",
        "hateful or not hateful",
        "hateful or not a hateful",
        "not-hateful or hateful",
        "not hateful or hateful",
        "not a hateful or hateful"
    ]
    positive_cases = ["hateful"]
    negative_cases = ["not hateful", "not-hateful", "not a hateful"]
    
    if check_if_string_contains_a_list_of_words(response, ambiguous_cases):
        return None
    
    if check_if_string_contains_a_list_of_words(response, negative_cases):
        return 0
    
    if check_if_string_contains_a_list_of_words(response, positive_cases):
        return 1
    
    return None

def calculate_metrics(true_labels, pred_labels):
    if not true_labels or not pred_labels:
        return {
            'accuracy': 0.0,
            'precision_0': 0.0, 'recall_0': 0.0, 'f1_0': 0.0,
            'precision_1': 0.0, 'recall_1': 0.0, 'f1_1': 0.0,
            'macro_f1': 0.0,
            'support_0': 0, 'support_1': 0
        }
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    tp = np.sum((true_labels == 1) & (pred_labels == 1))
    tn = np.sum((true_labels == 0) & (pred_labels == 0))
    fp = np.sum((true_labels == 0) & (pred_labels == 1))
    fn = np.sum((true_labels == 1) & (pred_labels == 0))
    
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
    
    macro_f1 = (f1_0 + f1_1) / 2
    accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision_0': precision_0, 'recall_0': recall_0, 'f1_0': f1_0,
        'precision_1': precision_1, 'recall_1': recall_1, 'f1_1': f1_1,
        'macro_f1': macro_f1,
        'support_0': tn + fp,
        'support_1': tp + fn
    }

# ===================== LIVE Core Classes (unchanged) =====================
class LIVEHook:
    def __init__(self, v_l, alpha_l):
        self.v_l = v_l
        self.alpha_l = alpha_l

    def __call__(self, module, inputs, output):
        if self.v_l is None:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        dtype = hidden.dtype
        device = hidden.device

        v = self.v_l.to(device=device, dtype=dtype)
        alpha = self.alpha_l.to(device=device, dtype=dtype)

        orig = hidden
        hidden = hidden + alpha * v

        original_norm = torch.norm(orig, dim=-1, keepdim=True)
        updated_norm = torch.norm(hidden, dim=-1, keepdim=True)
        hidden = hidden * (original_norm / (updated_norm + 1e-8))

        if rest is not None:
            return (hidden, *rest)
        return hidden

class LIVEVectors(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.V = nn.Parameter(torch.randn(num_layers, hidden_size) * 0.01)
        self.alpha = nn.Parameter(torch.ones(num_layers) * 0.1)
        
    def forward(self):
        return self.V, self.alpha
    
    def get_layer_params(self, layer_idx: int):
        if layer_idx < self.num_layers:
            return self.V[layer_idx], self.alpha[layer_idx]
        else:
            raise IndexError(f"Layer index {layer_idx} out of bounds (0-{self.num_layers-1})")

class LIVEQwenVLInference:
    def __init__(self, model_repo_id, train_data=None):
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

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        self.device = self.model.device
        self.hooks = []
        
        self.num_layers = self.model.config.text_config.num_hidden_layers
        self.hidden_size = self.model.config.text_config.hidden_size
        
        self.live_vectors = LIVEVectors(self.num_layers, self.hidden_size).to(self.device)
        
        if train_data:
            self.train_live(train_data)
        
        print(f"Model loaded on device: {self.device}")
        print(f"Number of layers: {self.num_layers}, Hidden size: {self.hidden_size}")
        print(f"LIVE parameters: {sum(p.numel() for p in self.live_vectors.parameters() if p.requires_grad)}")
    
    def safe_image_load(self, image_path):
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', (224, 224), color='black')
    
    # ================= MODIFIED get_image_path to support both FBHM and FHM =================
    def get_image_path(self, img_rel_path):
        """
        Construct full image path.
        For FBHM: path like "F12345/memes/img.jpg"
        For FHM:  path like "img/16395.png"
        """
        # If it's already an absolute path, return it
        if os.path.isabs(img_rel_path):
            return img_rel_path

        # FHM style: starts with "img/"
        if img_rel_path.startswith("img/"):
            return os.path.join(BASE_IMAGE_DIR, img_rel_path)   # e.g. BASE_IMAGE_DIR + "/img/16395.png"

        # FBHM style (with folder like "F...")
        if img_rel_path.startswith("F") and "/" in img_rel_path:
            folder = img_rel_path.split("/")[0]
            img_name = img_rel_path.split("/")[-1]
            candidate = os.path.join(BASE_IMAGE_DIR, folder, "memes", img_name)
            if os.path.exists(candidate):
                return candidate

        # Fallback: just join with BASE_IMAGE_DIR
        candidate = os.path.join(BASE_IMAGE_DIR, img_rel_path)
        if os.path.exists(candidate):
            return candidate

        # Try common extensions
        base_name = os.path.splitext(candidate)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if os.path.exists(base_name + ext):
                return base_name + ext

        return candidate  # return the original guess even if missing

    # ===== All other methods remain exactly as in the original script =====
    def create_icl_prompt(self, demonstrations: List[Dict], query: Dict, include_answer_for_query: bool = False):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for demo in demonstrations:
            img_path = self.get_image_path(demo['img'])
            if not os.path.exists(img_path):
                continue
            
            image = self.safe_image_load(img_path)
            extracted_text = demo["text"]
            label = demo.get('label', 0)
            answer = "hateful" if label == 1 else "not-hateful"
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: hateful or not-hateful."}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": answer
            })
        
        img_path = self.get_image_path(query['img'])
        if not os.path.exists(img_path):
            return None, None
        
        query_image = self.safe_image_load(img_path)
        extracted_text = query["text"]
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": query_image},
                {"type": "text", "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: hateful or not-hateful."}
            ]
        })
        
        if include_answer_for_query:
            label = query.get('label', 0)
            answer = "hateful" if label == 1 else "not-hateful"
            messages.append({
                "role": "assistant",
                "content": answer
            })
        
        return messages, query_image

    def get_output_distribution(self, inputs, images, use_live=False):
        if use_live:
            self.setup_live_hooks()
            outputs = self.model(**inputs, use_cache=False, return_dict=True)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=False, return_dict=True)

        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        if use_live:
            self.remove_hooks()

        return probs

    def setup_live_hooks(self):
        self.remove_hooks()
        for layer_idx in range(self.num_layers):
            v_l, alpha_l = self.live_vectors.get_layer_params(layer_idx)
            module = self.model.model.language_model.layers[layer_idx]
            hook = LIVEHook(v_l, alpha_l)
            handle = module.register_forward_hook(hook)
            self.hooks.append(handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def train_live(self, train_data, num_samples=NUM_TRAIN_SAMPLES, num_shots=NUM_SHOTS_TRAIN):
        # ... (unchanged, kept for completeness)
        print("Training not executed in this run; using pre‑trained LIVE vectors.")

    def save_live(self, path):
        torch.save({
            'V': self.live_vectors.V.detach().cpu(),
            'alpha': self.live_vectors.alpha.detach().cpu(),
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size
        }, path)

    def load_live(self, path):
        if os.path.exists(path):
            live_data = torch.load(path, map_location='cpu')
            self.live_vectors.V.data.copy_(live_data['V'].to(self.device))
            self.live_vectors.alpha.data.copy_(live_data['alpha'].to(self.device))
            print(f"Loaded LIVE vectors from {path}")
            return True
        else:
            print(f"Warning: LIVE path not found: {path}")
            return False

    def predict_with_live(self, meme_data, alpha_multiplier=1.0,
                         max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        img_rel_path = meme_data['img']
        img_path = self.get_image_path(img_rel_path)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return [f"Error: Image file not found: {img_path}"]

        image = self.safe_image_load(img_path)
        extracted_text = meme_data["text"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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

        original_alphas = self.live_vectors.alpha.clone()
        if alpha_multiplier != 1.0:
            self.live_vectors.alpha.data = original_alphas * alpha_multiplier

        self.setup_live_hooks()

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
            print(f"Error during generation: {e}")
            generated_texts = [f"Error: {e}"]

        if alpha_multiplier != 1.0:
            self.live_vectors.alpha.data = original_alphas

        self.remove_hooks()
        return generated_texts

    def predict_baseline(self, meme_data, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        img_rel_path = meme_data['img']
        img_path = self.get_image_path(img_rel_path)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return [f"Error: Image file not found: {img_path}"]

        image = self.safe_image_load(img_path)
        extracted_text = meme_data["text"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
            print(f"Error during generation: {e}")
            generated_texts = [f"Error: {e}"]

        return generated_texts

# ===================== Data loading for FHM JSONL =====================
def load_fhm_jsonl(file_path):
    """Load FHM dataset from JSONL (test_seen.jsonl format)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Ensure all required fields are present
            data.append({
                'id': item['id'],          # keep original string id
                'img': item['img'],         # e.g. "img/16395.png"
                'text': item['text'],
                'label': int(item['label']) # 0/1
            })
    return data

def load_fbhm_csv(file_path):
    """Keep original for training if needed"""
    df = pd.read_csv(file_path)
    data = []
    for idx, row in df.iterrows():
        data.append({
            'id': idx,
            'img': row['img'],
            'text': row['text'],
            'label': int(row['label'])
        })
    return data

# ===================== Main =====================
def main():
    print("="*80)
    print("LIVE (Learnable In-Context Vector) FOR HATE MEME CLASSIFICATION")
    print("Evaluating on FHM test_seen.jsonl")
    print("="*80)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load test data from JSONL
    print(f"\nLoading test data from: {TEST_JSONL_PATH}")
    test_data = load_fhm_jsonl(TEST_JSONL_PATH)
    print(f"Loaded {len(test_data)} test samples")
    # Show first example to verify id
    print(f"First sample id: {test_data[0]['id']}")

    # Initialize model and load pre‑trained LIVE vectors
    print("\nInitializing model...")
    inference_model = LIVEQwenVLInference(model_repo_id=MODEL_REPO_ID, train_data=None)
    success = inference_model.load_live(LIVE_OUTPUT_PATH)
    if not success:
        print("Exiting because LIVE vectors could not be loaded.")
        return

    # ========== Tuning alpha multiplier on a subset (optional) ==========
    tuning_subset = test_data   # use full test set for tuning (or a subset)
    print(f"\nTuning alpha multiplier on {len(tuning_subset)} samples...")

    # Baseline evaluation
    print("\n" + "="*60)
    print("BASELINE EVALUATION (No Intervention)")
    print("="*60)

    true_labels = []
    pred_labels = []
    ambiguous_count = 0

    for meme_data in tqdm(tuning_subset, desc="Baseline"):
        generated_texts = inference_model.predict_baseline(
            meme_data,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )
        if generated_texts and not generated_texts[0].startswith("Error:"):
            assistant_response = extract_assistant_response(generated_texts[0])
            pred_label = extract_label_from_response(assistant_response)
            true_label = meme_data.get('label', 0)

            if pred_label is not None:
                true_labels.append(true_label)
                pred_labels.append(pred_label)
            else:
                ambiguous_count += 1
        else:
            ambiguous_count += 1

    baseline_metrics = calculate_metrics(true_labels, pred_labels)
    baseline_macro_f1 = baseline_metrics['macro_f1']
    baseline_accuracy = baseline_metrics['accuracy']
    baseline_non_ambiguous_rate = len(true_labels) / len(tuning_subset) if len(tuning_subset) > 0 else 0

    print(f"Baseline Macro F1: {baseline_macro_f1:.4f} ({baseline_macro_f1:.2%})")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy:.2%})")
    print(f"Baseline Non-ambiguous responses: {len(true_labels)}/{len(tuning_subset)} ({baseline_non_ambiguous_rate:.2%})")

    # LIVE intervention tuning
    print("\n" + "="*60)
    print("LIVE INTERVENTION TUNING (Alpha Multiplier)")
    print("="*60)

    # Adjust this list as needed
    alpha_multipliers = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    # alpha_multipliers = [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5]

    best_macro_f1 = baseline_macro_f1
    best_alpha_multiplier = 0.5
    best_non_ambiguous_rate = baseline_non_ambiguous_rate
    results = []

    for alpha_mult in alpha_multipliers:
        print(f"\nTesting LIVE with alpha multiplier={alpha_mult}...")

        true_labels = []
        pred_labels = []
        ambiguous_count = 0

        for meme_data in tqdm(tuning_subset, desc=f"Alpha={alpha_mult}", leave=False):
            generated_texts = inference_model.predict_with_live(
                meme_data,
                alpha_multiplier=alpha_mult,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )

            print(generated_texts)

            if generated_texts and not generated_texts[0].startswith("Error:"):
                assistant_response = extract_assistant_response(generated_texts[0])
                pred_label = extract_label_from_response(assistant_response)
                true_label = meme_data.get('label', 0)

                if pred_label is not None:
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)
                else:
                    ambiguous_count += 1
            else:
                ambiguous_count += 1

        metrics = calculate_metrics(true_labels, pred_labels)
        macro_f1 = metrics['macro_f1']
        accuracy = metrics['accuracy']
        non_ambiguous_rate = len(true_labels) / len(tuning_subset) if len(tuning_subset) > 0 else 0

        result = {
            'alpha_multiplier': alpha_mult,
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'f1_0': metrics['f1_0'],
            'f1_1': metrics['f1_1'],
            'non_ambiguous_rate': non_ambiguous_rate,
            'total_parsed': len(true_labels)
        }
        results.append(result)

        print(f"  Macro F1: {macro_f1:.4f} ({macro_f1:.2%})")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy:.2%})")
        print(f"  Non-ambiguous responses: {len(true_labels)}/{len(tuning_subset)} ({non_ambiguous_rate:.2%})")

        if macro_f1 > best_macro_f1 and non_ambiguous_rate >= 0.9:
            best_macro_f1 = macro_f1
            best_alpha_multiplier = alpha_mult
            best_non_ambiguous_rate = non_ambiguous_rate

    print(f"\nBest alpha multiplier: {best_alpha_multiplier}")
    print(f"Best Macro F1: {best_macro_f1:.4f} ({best_macro_f1:.2%})")
    print(f"Improvement over baseline: {best_macro_f1 - baseline_macro_f1:.4f}")

    # ========== Final evaluation on full test set with best alpha ==========
    print("\n" + "="*80)
    print("FULL TEST SET EVALUATION")
    print("="*80)

    if best_alpha_multiplier != 1.0:
        print(f"Using LIVE intervention with alpha multiplier={best_alpha_multiplier}")
        use_live = True
    else:
        print("Using baseline (no intervention)")
        use_live = False

    with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
        output_file.write(f"=== LIVE INTERVENTION CONFIGURATION ===\n")
        output_file.write(f"Model: {MODEL_REPO_ID}\n")
        output_file.write(f"LIVE Source: {LIVE_OUTPUT_PATH}\n")
        output_file.write(f"Test data: {TEST_JSONL_PATH}\n")
        output_file.write(f"Number of training samples (original): {NUM_TRAIN_SAMPLES}\n")
        output_file.write(f"Training shots: {NUM_SHOTS_TRAIN}\n")
        output_file.write(f"Epochs: {EPOCHS}\n")
        if use_live:
            output_file.write(f"Intervention: LIVE with alpha multiplier={best_alpha_multiplier}\n")
        else:
            output_file.write(f"Intervention: Baseline (No LIVE)\n")
        output_file.write("="*50 + "\n\n")

        all_true_labels = []
        all_pred_labels = []
        ambiguous_count = 0

        for index, meme_data in tqdm(enumerate(test_data), total=len(test_data), desc="Processing test set"):
            if use_live:
                generated_texts = inference_model.predict_with_live(
                    meme_data,
                    alpha_multiplier=best_alpha_multiplier,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE
                )
            else:
                generated_texts = inference_model.predict_baseline(
                    meme_data,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE
                )

            # Write the full sample info (including the original id) to output
            output_file.write(json.dumps(meme_data, default=str))
            output_file.write('\n----------\n')

            for text in generated_texts:
                formatted_text = re.sub(r'\bassistant\b', 'Assistant:', text, flags=re.IGNORECASE)
                output_file.write(formatted_text)
                output_file.write('\n')

            # Extract prediction
            if generated_texts and not generated_texts[0].startswith("Error:"):
                assistant_response = extract_assistant_response(generated_texts[0])
                pred_label = extract_label_from_response(assistant_response)
                true_label = meme_data.get('label', 0)

                if pred_label is not None:
                    all_true_labels.append(true_label)
                    all_pred_labels.append(pred_label)
                else:
                    ambiguous_count += 1
            else:
                ambiguous_count += 1

            output_file.write("\n##########\n")

            if (index + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        final_metrics = calculate_metrics(all_true_labels, all_pred_labels)

        output_file.write(f"\n=== FINAL METRICS ===\n")
        output_file.write(f"Total samples: {len(test_data)}\n")
        output_file.write(f"Parsed samples: {len(all_true_labels)}\n")
        output_file.write(f"Ambiguous responses: {ambiguous_count}\n")
        output_file.write(f"Non-ambiguous rate: {len(all_true_labels)/len(test_data):.2%}\n")
        output_file.write(f"Accuracy: {final_metrics['accuracy']:.4f}\n")
        output_file.write(f"Macro F1: {final_metrics['macro_f1']:.4f}\n")
        output_file.write(f"F1 (Class 0 - not-hateful): {final_metrics['f1_0']:.4f}\n")
        output_file.write(f"F1 (Class 1 - hateful): {final_metrics['f1_1']:.4f}\n")

        print(f"\n=== FINAL RESULTS ===")
        print(f"Total samples: {len(test_data)}")
        print(f"Parsed samples: {len(all_true_labels)}")
        print(f"Ambiguous responses: {ambiguous_count}")
        print(f"Non-ambiguous rate: {len(all_true_labels)/len(test_data):.2%}")
        print(f"Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Macro F1: {final_metrics['macro_f1']:.4f}")
        print(f"F1 (Class 0 - not-hateful): {final_metrics['f1_0']:.4f}")
        print(f"F1 (Class 1 - hateful): {final_metrics['f1_1']:.4f}")

    print(f"\nOutput saved to '{OUTPUT_TXT}'")

if __name__ == "__main__":
    main()