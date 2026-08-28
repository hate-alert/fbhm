#!/usr/bin/env python3
"""
Qwen3-VL SFT training script (Trainer + QLoRA)
- Uniform image resizing to avoid image_grid_thw / pos_embed mismatch
- Loss computed only on assistant response
- Optional HF upload
- Modified for FBHM dataset structure
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
# USER CONFIG — edit as needed
# ---------------------------
FBHM_DIR = "../FBHM"  # Directory containing FBHM data
TRAIN_CSV_PATH = f"{FBHM_DIR}/train.csv"
# MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME = "pbhaskar/qwen3-vl-8b-sft-fhm-15-epochs-earlystop"
OUTPUT_DIR = "./qwen3-vl-8b-sft-fhm-FBHM-15-epochs-earlystop"

HF_UPLOAD_NAME = "pbhaskar/qwen3-vl-8b-sft-fhm-FBHM-15-epochs-earlystop"  # Set to your desired HF repo name (e.g. "username/repo-name")
HF_TOKEN = ""  # <-- set your token to enable upload

login(token=HF_TOKEN)

# training hyperparams
MAX_LENGTH = 2048
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
BF16 = True  # set False if hardware doesn't support bf16
SAVE_STEPS = 50
LOGGING_STEPS = 20
SEED = 42
VALIDATION_SPLIT = 0.1  # small val split

EARLY_STOPPING_PATIENCE = 3  # number of evaluation steps with no improvement before stopping

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

SYSTEM_PROMPT = """Consider the following definitions.
1. hateful - a direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. Attack is defined as violent or dehumanizing (comparing people to non-human things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also considered hateful.
2. not-hateful - an attack which is not hateful and follows social norms.
Classify the above meme as hateful or not-hateful based on the above definitions considering the image and the extracted text from the image delimited by three backticks.
Provide your answer in the format: hateful or not-hateful."""

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
            commit_message="Qwen3-VL-8B fine-tuned on FBHM dataset (QLoRA)",
            repo_type="model"
        )
        print(f"✅ Uploaded to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print("❌ HF upload failed:", e)
        return False

# ---------------------------
# Dataset class
# ---------------------------
class QwenFhmDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Extracted text: ```{s['text']}```\nProvide your answer in the format: hateful or not-hateful."}
                ]
            },
            {"role": "assistant", "content": s["label"]}
        ]
        return {"conversation": conversation, "image": image}

# ---------------------------
# Helper: get target image size from processor, with sensible fallback
# ---------------------------
def detect_target_size(proc):
    # Try common attributes; fall back to safe default 896x896 if values look wrong
    size = None
    if hasattr(proc, "image_processor") and getattr(proc.image_processor, "size", None):
        size = proc.image_processor.size
    elif getattr(proc, "feature_extractor", None) and getattr(proc.feature_extractor, "size", None):
        size = proc.feature_extractor.size

    # Interpret size
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

    # If detected values are unrealistic or None, fallback to 896
    if not h or not w or h > 2000 or w > 2000 or h < 32 or w < 32:
        target_w, target_h = 896, 896
    else:
        target_w, target_h = w, h

    print(f"Using target image size: {target_w}x{target_h}")
    return target_w, target_h

# ---------------------------
# Collator: resize images uniformly, call processor with do_image_splitting=False,
# and mask labels so loss is only on assistant response.
# ---------------------------
def data_collator(examples: List[Dict], processor: AutoProcessor, target_wh: tuple, max_length: int):
    images = [ex["image"] for ex in examples]
    convs = [ex["conversation"] for ex in examples]

    target_w, target_h = target_wh
    # Resize all images uniformly
    resized_images = []
    for im in images:
        if (im.width, im.height) != (target_w, target_h):
            im_resized = im.resize((target_w, target_h), resample=Image.BICUBIC)
        else:
            im_resized = im
        resized_images.append(im_resized)

    # Build texts via chat template
    texts = []
    for conv in convs:
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # Ensure do_image_splitting=False here
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

    # Try locate assistant start marker first
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
            assistant_text = convs[i][-1]["content"]
            assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
            pos = find_sublist(input_ids, assistant_ids)
            if pos != -1:
                start_idx = pos

        if start_idx != -1:
            labels[i, :start_idx] = -100
        else:
            # safety fallback: keep last 50 tokens unmasked
            keep_last = 50
            if labels.size(1) > keep_last:
                labels[i, :-keep_last] = -100

    # Mask padding tokens
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    inputs["labels"] = labels

    # bsz = labels.size(0)
    # num_loss_tokens = (labels != -100).sum().item()

    # if num_loss_tokens < bsz * 3:
    #     raise RuntimeError(
    #         f"Suspicious supervision: {num_loss_tokens} loss tokens "
    #         f"for batch size {bsz}"
    #     )

    if random.random() < 0.001:   # print rarely
        print("\n===== DEBUG MASK CHECK =====")
        ids = inputs["input_ids"][0]
        labs = inputs["labels"][0]

        decoded_full = processor.tokenizer.decode(ids, skip_special_tokens=False)

        supervised_tokens = ids[labs != -100]
        decoded_supervised = processor.tokenizer.decode(
            supervised_tokens,
            skip_special_tokens=False
        )

        print("FULL INPUT:")
        print(decoded_full[-500:])

        print("\nTOKENS RECEIVING LOSS:")
        print(decoded_supervised)
        print("============================\n")

    return inputs

# ---------------------------
# Main
# ---------------------------
def main():
    set_seed(SEED)

    # Load samples from CSV
    samples = []
    train_path = Path(TRAIN_CSV_PATH)
    fbhm_dir = Path(FBHM_DIR)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    
    with train_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The 'img' column has paths like "F1/memes/0001.jpg"
            img_path = fbhm_dir / row["img"]
            
            if not img_path.exists():
                print(f"Warning: missing image {img_path}, skipping")
                continue
            
            # Convert label: 1 -> hateful, 0 -> not-hateful
            label_val = int(row["label"])
            label = "hateful" if label_val == 1 else "not-hateful"
            
            samples.append({
                "image_path": str(img_path),
                "text": row.get("text", ""),
                "label": label
            })
    
    print(f"Loaded {len(samples)} samples")

    # small validation split
    if VALIDATION_SPLIT > 0 and len(samples) >= 45:
        random.shuffle(samples)
        val_count = max(1, int(len(samples) * VALIDATION_SPLIT))
        val_samples = samples[:val_count]
        train_samples = samples[val_count:]
        print(f"Using {len(train_samples)} train and {len(val_samples)} val samples")
    else:
        train_samples = samples
        val_samples = []

    # Initialize processor (and detect target image size)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, do_image_splitting=False)
    processor.tokenizer.padding_side = "right"
    target_w, target_h = detect_target_size(processor)

    # Prepare model (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base model (quantized) — this may take a while...")
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

    # LoRA target modules (broad but safe). If you know exact names, prefer exact list.
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # "qkv", "proj", "wq", "wk", "wv", "wo"
    ]
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=target_modules,
        lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = QwenFhmDataset(train_samples)
    eval_dataset = QwenFhmDataset(val_samples) if val_samples else None

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
        data_collator=lambda examples: data_collator(examples, processor, (target_w, target_h), MAX_LENGTH),
        callbacks=callbacks,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print("Saving model and processor to", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Upload to HF if token provided
    if HF_TOKEN and HF_TOKEN != "your_hf_token_here":
        print("Uploading to Hugging Face Hub...")
        upload_to_hf(OUTPUT_DIR, HF_UPLOAD_NAME, HF_TOKEN)
    else:
        print("HF upload skipped. Set HF_TOKEN to upload.")

    print("Done.")

if __name__ == "__main__":
    main()