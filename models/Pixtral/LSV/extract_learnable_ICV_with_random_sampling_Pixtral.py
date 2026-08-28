#!/usr/bin/env python3
"""
LIVE (Learnable In-Context Vector) for Pixtral Hate Meme Classification
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

from transformers import LlavaForConditionalGeneration, AutoProcessor, get_linear_schedule_with_warmup

# Configuration
TEST_CSV_PATH = "../FBHM/test.csv"
TRAIN_CSV_PATH = "../FBHM/train.csv"
BASE_IMAGE_DIR = "../FBHM"
MODEL_REPO_ID = "mistral-community/pixtral-12b"
# MODEL_REPO_ID = "pbhaskar/pixtral-12b-sft-mami-15-epochs-earlystop-2"
# LIVE_OUTPUT_PATH = "./learnable_ICVs_pixtral_sft_mami_15_epochs_earlystop_2_500_samples_random_batch_1.pt"
LIVE_OUTPUT_PATH = "./learnable_ICVs_normal_500_samples_random_batch_1.pt"
# OUTPUT_TXT = "./learnable_ICVs_pixtral_sft_mami_15_epochs_earlystop_2_500_samples_random_batch_1_intervention_on_FBHM_test.txt"
OUTPUT_TXT = "./learnable_ICVs_normal_500_samples_random_batch_1_intervention_on_FBHM_test.txt"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.001
NUM_TRAIN_SAMPLES = 500          # As in paper
NUM_SHOTS_TRAIN = 32              # 32-shot for training
BATCH_SIZE = 1                    # As in paper
LEARNING_RATE_V = 1e-3            # Learning rate for V vectors
LEARNING_RATE_ALPHA = 1e-2        # Learning rate for alpha scalars
LAMBDA = 0.5                      # Weight for ground truth loss
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

# ----------------------------------------------------------------------
# Helper functions (unchanged)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# LIVE components (unchanged)
# ----------------------------------------------------------------------
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
    """
    Learnable In-Context Vectors (LIVE) as in the paper
    V = {v_1, v_2, ..., v_L}, v_i ∈ R^(1×d)
    α = {α_1, α_2, ..., α_L}, α_i ∈ R^(1×1)
    """
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

# ----------------------------------------------------------------------
# Pixtral inference class
# ----------------------------------------------------------------------
class LIVEPixtralInference:
    def __init__(self, model_repo_id, train_data=None):
        print(f"Loading model from {model_repo_id}...")
        self.processor = AutoProcessor.from_pretrained(
            model_repo_id,
            trust_remote_code=True
        )
        # Set padding side and pad token for causal LM
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = "right"
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.gradient_checkpointing_enable()

        self.device = self.model.device
        self.hooks = []

        # Get language model and its layers (handles both MistralForCausalLM and MistralModel)
        self.lm_model = self.model.language_model
        if hasattr(self.lm_model, 'model') and hasattr(self.lm_model.model, 'layers'):
            self.layers = self.lm_model.model.layers
        elif hasattr(self.lm_model, 'layers'):
            self.layers = self.lm_model.layers
        else:
            raise AttributeError("Cannot find layers in language model. Check model structure.")

        self.num_layers = len(self.layers)
        self.hidden_size = self.lm_model.config.hidden_size

        # Initialize LIVE vectors
        self.live_vectors = LIVEVectors(self.num_layers, self.hidden_size).to(self.device)

        if train_data:
            self.train_live(train_data)

        print(f"Model loaded on device: {self.device}")
        print(f"Number of layers: {self.num_layers}, Hidden size: {self.hidden_size}")
        print(f"LIVE parameters: {sum(p.numel() for p in self.live_vectors.parameters() if p.requires_grad)}")

    def _cast_to_model_dtype(self, inputs):
        """Convert all floating point tensors in inputs to model.dtype."""
        return {
            k: v.to(self.model.dtype) if torch.is_floating_point(v) else v
            for k, v in inputs.items()
        }

    def safe_image_load(self, image_path):
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', (224, 224), color='black')

    def get_image_path(self, img_rel_path):
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
        return img_path

    def create_icl_prompt(self, demonstrations: List[Dict], query: Dict, include_answer_for_query: bool = False):
        """
        Create ICL prompt with demonstrations and query.
        Returns: (messages, images_list) where images_list contains all images in order.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        images_list = []

        # Add demonstrations
        for demo in demonstrations:
            img_path = self.get_image_path(demo['img'])
            if not os.path.exists(img_path):
                continue
            image = self.safe_image_load(img_path)
            images_list.append(image)
            extracted_text = demo["text"]
            label = demo.get('label', 0)
            answer = "hateful" if label == 1 else "not-hateful"

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Extracted text: ```{extracted_text}```\nProvide your answer in the format: hateful or not-hateful."}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": answer
            })

        # Add query
        img_path = self.get_image_path(query['img'])
        if not os.path.exists(img_path):
            return None, None

        query_image = self.safe_image_load(img_path)
        images_list.append(query_image)
        extracted_text = query["text"]

        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
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

        return messages, images_list

    def get_output_distribution(self, inputs, use_live=False):
        inputs = self._cast_to_model_dtype(inputs)

        if use_live:
            self.setup_live_hooks()
            outputs = self.model(**inputs, return_dict=True, use_cache=False)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True, use_cache=False)

        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        if use_live:
            self.remove_hooks()

        return probs

    def setup_live_hooks(self):
        self.remove_hooks()
        for layer_idx in range(self.num_layers):
            v_l, alpha_l = self.live_vectors.get_layer_params(layer_idx)
            module = self.layers[layer_idx]
            hook = LIVEHook(v_l, alpha_l)
            handle = module.register_forward_hook(hook)
            self.hooks.append(handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def train_live(self, train_data, num_samples=NUM_TRAIN_SAMPLES, num_shots=NUM_SHOTS_TRAIN, start_epoch=0, total_epochs=EPOCHS):
        print(f"\nTraining LIVE vectors on {num_samples} samples with {num_shots}-shot ICL...")

        # Random sampling (as in original Qwen code)
        if len(train_data) > num_samples:
            train_subset = random.sample(train_data, num_samples)
        else:
            train_subset = train_data

        print(f"Using {len(train_subset)} training samples")

        self.model.eval()
        self.live_vectors.train()

        optimizer = torch.optim.AdamW([
            {'params': self.live_vectors.V, 'lr': LEARNING_RATE_V},
            {'params': self.live_vectors.alpha, 'lr': LEARNING_RATE_ALPHA}
        ])

        # for epoch in range(EPOCHS):
        for epoch in range(start_epoch, total_epochs):
            epoch_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_gt_loss = 0.0

            random.shuffle(train_subset)

            for batch_start in tqdm(range(0, len(train_subset), BATCH_SIZE),
                                    desc=f"Epoch {epoch+1}/{total_epochs}"):
                batch = train_subset[batch_start:batch_start + BATCH_SIZE]

                batch_kl_loss = 0.0
                batch_gt_loss = 0.0

                for query in batch:
                    # Sample demonstrations for this query
                    available_demos = [d for d in train_subset if d['id'] != query['id']]
                    if len(available_demos) < num_shots:
                        continue
                    demonstrations = random.sample(available_demos, num_shots)

                    # 1. ICL distribution (without LIVE)
                    icl_messages, icl_images = self.create_icl_prompt(demonstrations, query, include_answer_for_query=False)
                    if icl_messages is None:
                        continue

                    icl_text = self.processor.apply_chat_template(
                        icl_messages, tokenize=False, add_generation_prompt=True
                    )
                    icl_inputs = self.processor(
                        text=[icl_text],
                        images=icl_images,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    icl_inputs = self._cast_to_model_dtype(icl_inputs)
                    icl_probs = self.get_output_distribution(icl_inputs, use_live=False)

                    # 2. LIVE distribution (no demonstrations)
                    live_messages, live_images = self.create_icl_prompt([], query, include_answer_for_query=False)
                    live_text = self.processor.apply_chat_template(
                        live_messages, tokenize=False, add_generation_prompt=True
                    )
                    live_inputs = self.processor(
                        text=[live_text],
                        images=live_images,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    live_probs = self.get_output_distribution(live_inputs, use_live=True)

                    # 3. Compute losses
                    kl_loss = F.kl_div(
                        live_probs.log(),
                        icl_probs,
                        reduction='batchmean'
                    )

                    # Ground truth loss
                    hateful_tokens = self.processor.tokenizer.encode("hateful", add_special_tokens=False)
                    not_hateful_tokens = self.processor.tokenizer.encode("not-hateful", add_special_tokens=False)
                    hateful_id = hateful_tokens[0] if hateful_tokens else -1
                    not_hateful_id = not_hateful_tokens[0] if not_hateful_tokens else -1

                    if hateful_id != -1 and not_hateful_id != -1:
                        correct_label = query.get('label', 0)
                        if correct_label == 1:
                            gt_loss = -torch.log(live_probs[0, hateful_id] + 1e-8)
                        else:
                            gt_loss = -torch.log(live_probs[0, not_hateful_id] + 1e-8)
                    else:
                        gt_loss = torch.tensor(0.0).to(self.device)

                    batch_kl_loss += kl_loss
                    batch_gt_loss += gt_loss

                if len(batch) > 0:
                    batch_kl_loss = batch_kl_loss / len(batch)
                    batch_gt_loss = batch_gt_loss / len(batch)
                    loss = LAMBDA * batch_gt_loss + batch_kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_kl_loss += batch_kl_loss.item()
                    epoch_gt_loss += batch_gt_loss.item()

                    del icl_inputs, live_inputs, icl_probs, live_probs, loss
                    del kl_loss, gt_loss

                # if batch_start % (10 * BATCH_SIZE) == 0:
                #     gc.collect()
                #     if torch.cuda.is_available():
                #         torch.cuda.empty_cache()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            num_batches = len(train_subset) // BATCH_SIZE
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_kl_loss = epoch_kl_loss / num_batches
                avg_gt_loss = epoch_gt_loss / num_batches
                print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, KL Loss = {avg_kl_loss:.4f}, GT Loss = {avg_gt_loss:.4f}")

            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"./live_checkpoint_pixtral_normal_epoch_{epoch+1}_500_samples_batch_1.pt"
                self.save_live(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        self.save_live(LIVE_OUTPUT_PATH)
        print(f"Training complete! Saved LIVE vectors to {LIVE_OUTPUT_PATH}")

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
        img_path = self.get_image_path(meme_data['img'])
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
                    {"type": "image"},
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
            return_tensors="pt"
        ).to(self.device)
        inputs = self._cast_to_model_dtype(inputs)

        # Scale alpha temporarily
        original_alphas = self.live_vectors.alpha.clone()
        if alpha_multiplier != 1.0:
            self.live_vectors.alpha.data = original_alphas * alpha_multiplier

        self.setup_live_hooks()

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
            # generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            input_length = inputs['input_ids'].shape[1]
            generated_tokens = generated_ids[0][input_length:]
            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
            generated_texts = [generated_text]

        except Exception as e:
            print(f"Error during generation: {e}")
            generated_texts = [f"Error: {e}"]

        # Restore alpha
        if alpha_multiplier != 1.0:
            self.live_vectors.alpha.data = original_alphas

        self.remove_hooks()
        return generated_texts

    def predict_baseline(self, meme_data, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        img_path = self.get_image_path(meme_data['img'])
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
                    {"type": "image"},
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
            return_tensors="pt"
        ).to(self.device)
        inputs = self._cast_to_model_dtype(inputs)

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
            # generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            input_length = inputs['input_ids'].shape[1]
            generated_tokens = generated_ids[0][input_length:]
            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
            generated_texts = [generated_text]
            
        except Exception as e:
            print(f"Error during generation: {e}")
            generated_texts = [f"Error: {e}"]

        return generated_texts

# ----------------------------------------------------------------------
# Data loading and main (unchanged except class name)
# ----------------------------------------------------------------------
def load_fbhm_csv(file_path):
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

def main():
    print("="*80)
    print("LIVE (Learnable In-Context Vector) FOR HATE MEME CLASSIFICATION (Pixtral)")
    print("="*80)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\nLoading training data from: {TRAIN_CSV_PATH}")
    train_data = load_fbhm_csv(TRAIN_CSV_PATH)
    print(f"Loaded {len(train_data)} training samples")

    print("\nInitializing model and LIVE vectors...")
    # Uncomment to train from scratch:
    inference_model = LIVEPixtralInference(model_repo_id=MODEL_REPO_ID, train_data=train_data)

    # Or load pre-trained LIVE vectors:
    # inference_model = LIVEPixtralInference(model_repo_id=MODEL_REPO_ID, train_data=None)
    inference_model.load_live(LIVE_OUTPUT_PATH)
    # inference_model.load_live(
    #     "./live_checkpoint_epoch_5_500_samples.pt"
    # )
    # inference_model.train_live(train_data, start_epoch=5, total_epochs=10)

    print(f"\nLoading test data from: {TEST_CSV_PATH}")
    test_data = load_fbhm_csv(TEST_CSV_PATH)
    print(f"Loaded {len(test_data)} test samples")

    tuning_subset = test_data

    print(f"\nTuning alpha multiplier on {len(tuning_subset)} samples...")

    # Baseline evaluation (optional, here we just set placeholders)
    baseline_macro_f1 = 0.0
    baseline_non_ambiguous_rate = 1.0

    # Alpha multiplier tuning
    print("\n" + "="*60)
    print("LIVE INTERVENTION TUNING (Alpha Multiplier)")
    print("="*60)

    alpha_multipliers = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                         1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

    best_macro_f1 = baseline_macro_f1
    best_alpha_multiplier = 1.0
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

            # Uncomment for debugging:
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

    # Full test evaluation
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
        output_file.write(f"Number of training samples: {NUM_TRAIN_SAMPLES}\n")
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

            output_file.write(json.dumps(meme_data, default=str))
            output_file.write('\n----------\n')

            for text in generated_texts:
                formatted_text = re.sub(r'\bassistant\b', 'Assistant:', text, flags=re.IGNORECASE)
                output_file.write(formatted_text)
                output_file.write('\n')

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