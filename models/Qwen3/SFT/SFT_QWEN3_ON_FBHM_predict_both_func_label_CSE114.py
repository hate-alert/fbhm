#!/usr/bin/env python3
"""
Qwen3-VL SFT training script - Multi-Task: Hate Classification + Functionality Prediction
Train model to generate: "label: [hateful/not-hateful]; functionality: [F1/F2/...]"
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login, create_repo, upload_folder

# ---------------------------
# USER CONFIG
# ---------------------------
FBHM_DIR = "../FBHM"
TRAIN_CSV_PATH = f"{FBHM_DIR}/train.csv"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = "./qwen3-vl-8b-sft-fbhm-15-epochs-predict-both-func-label-earlystop"

HF_UPLOAD_NAME = "pbhaskar/qwen3-vl-8b-sft-fbhm-15-epochs-predict-both-func-label-earlystop"
HF_TOKEN = ""

# training hyperparams
MAX_LENGTH = 2048
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5 
WEIGHT_DECAY = 0.01
BF16 = True
SAVE_STEPS = 500
LOGGING_STEPS = 20
SEED = 42
VALIDATION_SPLIT = 0.1

EARLY_STOPPING_PATIENCE = 3  # number of evaluation steps with no improvement before stopping

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# System prompt focusing on BOTH tasks
SYSTEM_PROMPT = """Consider the following definitions.

HATE DEFINITIONS:
1. hateful - a direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. Attack is defined as violent or dehumanizing (comparing people to non-human things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also considered hateful.
2. not-hateful - an attack which is not hateful and follows social norms.

FUNCTIONALITY DEFINITIONS:
Memes can be categorized into 25 functionality types based on their visual and textual characteristics:
- F1: Action Shown in Image (boot licking, gestures, violence depiction, etc.)
- F2: Using Emoji to Express Emotion
- F3: Pie chart based memes
- F4: Perturbations in text (spelling errors, character swaps)
- F5: Different text size, colors, fonts
- F6: Using referential/dehumanizing text
- F7: Animated memes / Cartoonish Characters
- F8: Black-white images and Large Color variations
- F9: Stacked Memes with cartoons as base
- F10: Stacked Memes without cartoons as base
- F11: Using Signs to Show Hateful Names
- F12: AI generated Image
- F13: Positive Image, Negative Sentiment in Text
- F14: Negative image, Positive Sentiment in Text
- F15: Satire memes
- F16: Using slur words in hateful memes
- F17: Using masked slur words in hateful memes
- F18: Stickers used in memes
- F19: Perturbations in memes which are human readable
- F20: Memes having slur words but are not-hateful
- F21: Memes having masked slur words but are not-hateful
- F22: Implicitly Not-hateful memes
- F23: Other Not-hateful memes
- F24: Location variant memes
- F25: Counter memes

TASK:
1. Classify the meme as hateful or not-hateful based on the above hate definitions.
2. Identify the primary functionality category of the meme from F1 to F25.

Output format: "label: [hateful/not-hateful]; functionality: [F1/F2/.../F25]"

Analyze both the image and the extracted text from the image."""

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def upload_to_hf(local_dir: str, repo_id: str, token: str) -> bool:
    try:
        login(token=token)
        create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        upload_folder(
            repo_id=repo_id,
            folder_path=local_dir,
            commit_message="Qwen3-VL-8B fine-tuned on FBHM (multi-task: hate+functionality)",
            repo_type="model"
        )
        print(f"Uploaded to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print("HF upload failed:", e)
        return False

# ---------------------------
# Dataset class - MULTI-TASK VERSION
# ---------------------------
class QwenFBHMMultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        
        # Get label and functionality
        label = s["label"]  # "hateful" or "not-hateful"
        functionality = s.get("functionality", "F1")  # Default to F1 if missing
        
        # Create multi-task target
        assistant_response = f"label: {label}; functionality: {functionality}"
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Extracted text: ```{s['text']}```\nProvide your analysis in the specified format."}
                ]
            },
            {"role": "assistant", "content": assistant_response}
        ]
        return {"conversation": conversation, "image": image}

# ---------------------------
# Helper: get target image size
# ---------------------------
def detect_target_size(proc):
    size = None
    if hasattr(proc, "image_processor") and getattr(proc.image_processor, "size", None):
        size = proc.image_processor.size
    elif getattr(proc, "feature_extractor", None) and getattr(proc.feature_extractor, "size", None):
        size = proc.feature_extractor.size
    
    if isinstance(size, dict):
        h = size.get("height") or size.get("shortest_edge") or size.get("longest_edge")
        w = size.get("width") or size.get("shortest_edge") or size.get("longest_edge")
        if h and w:
            try:
                h, w = int(h), int(w)
            except Exception:
                h, w = None, None
        else:
            h, w = None, None
    else:
        h, w = None, None
    
    if not h or not w or h > 2000 or w > 2000 or h < 32 or w < 32:
        target_w, target_h = 896, 896
    else:
        target_w, target_h = w, h
    
    print(f"Using target image size: {target_w}x{target_h}")
    return target_w, target_h

# ---------------------------
# Collator for multi-task
# ---------------------------
def data_collator(examples: List[Dict], processor: AutoProcessor, target_wh: tuple, max_length: int):
    images = [ex["image"] for ex in examples]
    convs = [ex["conversation"] for ex in examples]
    
    target_w, target_h = target_wh
    resized_images = []
    for im in images:
        if (im.width, im.height) != (target_w, target_h):
            im_resized = im.resize((target_w, target_h), resample=Image.BICUBIC)
        else:
            im_resized = im
        resized_images.append(im_resized)
    
    texts = []
    for conv in convs:
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    
    inputs = processor(
        text=texts,
        images=resized_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        do_image_splitting=False
    )
    
    # Create labels and mask tokens before assistant response
    labels = inputs["input_ids"].clone()
    tokenizer = processor.tokenizer
    
    # Find assistant start marker
    assistant_marker = "<|im_start|>assistant"
    try:
        assistant_marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)
    except Exception:
        assistant_marker_ids = []
    
    def find_sublist(hay: List[int], needle: List[int]) -> int:
        if not needle:
            return -1
        n = len(needle)
        for i in range(len(hay) - n + 1):
            if hay[i:i+n] == needle:
                return i
        return -1
    
    batch_ids = inputs["input_ids"].tolist()
    for i, input_ids in enumerate(batch_ids):
        start_idx = -1
        if assistant_marker_ids:
            pos = find_sublist(input_ids, assistant_marker_ids)
            if pos != -1:
                start_idx = pos + len(assistant_marker_ids)
        
        if start_idx == -1:
            # Fallback: search for assistant response text
            assistant_text = convs[i][-1]["content"]
            assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
            pos = find_sublist(input_ids, assistant_ids)
            if pos != -1:
                start_idx = pos
        
        if start_idx != -1:
            labels[i, :start_idx] = -100
        else:
            # Safety fallback: keep last 100 tokens unmasked (for longer responses)
            keep_last = 100
            if labels.size(1) > keep_last:
                labels[i, :-keep_last] = -100
    
    # Mask padding tokens
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    
    inputs["labels"] = labels
    
    # Debug: check loss token count
    bsz = labels.size(0)
    num_loss_tokens = (labels != -100).sum().item()
    # print(f"Batch size: {bsz}, Loss tokens per sample: {num_loss_tokens/bsz:.1f}")
    
    return inputs

# ---------------------------
# Main
# ---------------------------
def main():
    set_seed(SEED)
    
    # Load samples
    samples = []
    train_path = Path(TRAIN_CSV_PATH)
    fbhm_dir = Path(FBHM_DIR)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    
    with train_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = fbhm_dir / row["img"]
            
            if not img_path.exists():
                print(f"Warning: missing image {img_path}, skipping")
                continue
            
            # Convert label
            label_val = int(row["label"])
            label = "hateful" if label_val == 1 else "not-hateful"
            
            # Get functionality (must exist for FBHM)
            functionality = row.get("functionality", "").strip()
            if not functionality:
                print(f"Warning: missing functionality for {img_path}, skipping")
                continue
            
            samples.append({
                "image_path": str(img_path),
                "text": row.get("text", ""),
                "label": label,
                "functionality": functionality
            })
    
    print(f"Loaded {len(samples)} samples with both label and functionality")
    
    # Distribution check
    print(f"Label distribution:")
    hateful = sum(1 for s in samples if s["label"] == "hateful")
    not_hateful = len(samples) - hateful
    print(f"  hateful: {hateful} ({hateful/len(samples)*100:.1f}%)")
    print(f"  not-hateful: {not_hateful} ({not_hateful/len(samples)*100:.1f}%)")
    
    print(f"Functionality distribution (top 10):")
    func_counts = {}
    for s in samples:
        func = s.get("functionality", "unknown")
        func_counts[func] = func_counts.get(func, 0) + 1
    for func, count in sorted(func_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {func}: {count} ({count/len(samples)*100:.1f}%)")
    
    # Train/val split
    if VALIDATION_SPLIT > 0 and len(samples) >= 50:
        random.shuffle(samples)
        val_count = max(1, int(len(samples) * VALIDATION_SPLIT))
        val_samples = samples[:val_count]
        train_samples = samples[val_count:]
        print(f"\nUsing {len(train_samples)} train and {len(val_samples)} val samples")
    else:
        train_samples = samples
        val_samples = []
    
    # Initialize processor
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True, 
        do_image_splitting=False
    )
    processor.tokenizer.padding_side = "right"
    target_w, target_h = detect_target_size(processor)
    
    # Prepare model (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    print("\nLoading base model (quantized)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    
    # LoRA config
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_config = LoraConfig(
        r=LORA_R, 
        lora_alpha=LORA_ALPHA, 
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Datasets
    train_dataset = QwenFBHMMultiTaskDataset(train_samples)
    eval_dataset = QwenFBHMMultiTaskDataset(val_samples) if val_samples else None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=False,
        dataloader_num_workers=4,
        report_to="none",
        push_to_hub=False,
        # Evaluation and early stopping
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,  # evaluate at same frequency as saving
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Prepare callbacks (early stopping only if we have eval data)
    callbacks = []
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda examples: data_collator(
            examples, processor, (target_w, target_h), MAX_LENGTH
        ),
        callbacks=callbacks,
    )
    
    # Train
    print("\nStarting multi-task training (hate classification + functionality prediction)...")
    trainer.train()
    
    # Save
    print(f"\nSaving model and processor to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    # Upload to HF
    if HF_TOKEN and HF_TOKEN != "":
        print("\nUploading to Hugging Face Hub...")
        upload_to_hf(OUTPUT_DIR, HF_UPLOAD_NAME, HF_TOKEN)
    else:
        print("\nHF upload skipped. Set HF_TOKEN to upload.")
    
    print("\nDone.")

if __name__ == "__main__":
    main()