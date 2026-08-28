#!/usr/bin/env python3
"""
Qwen3-VL SFT training script (Trainer + QLoRA) for MAMI dataset
- Uniform image resizing to avoid image_grid_thw / pos_embed mismatch
- Loss computed only on assistant response
- Early stopping based on eval loss
- Optional HF upload
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
TRAIN_TSV_PATH = "/home/du1/21CS30035/mami-dataset/train.tsv"
IMAGE_DIR = "/home/du1/21CS30035/mami-dataset/images"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = "./qwen3-vl-8b-sft-mami-15-epochs-earlystop-2"  # local output dir for checkpoints and final model

HF_UPLOAD_NAME = "pbhaskar/qwen3-vl-8b-sft-mami-15-epochs-earlystop-2"  # <-- set your repo name for HF upload (e.g. username/repo)
HF_TOKEN = ""  # <-- set your token to enable upload

# training hyperparams
MAX_LENGTH = 2048
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
BF16 = True  # set False if hardware doesn't support bf16
SAVE_STEPS = 500
LOGGING_STEPS = 20
SEED = 42
VALIDATION_SPLIT = 0.1  # small val split from train

# Early stopping
EARLY_STOPPING_PATIENCE = 3  # number of evaluation steps with no improvement before stopping

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Updated prompt for MAMI (misogyny detection)
SYSTEM_PROMPT = """Consider the following definitions.
1. misogynistic - a meme is misogynous if it conceptually describes an offensive, sexist or hateful scene (weak or strong, implicitly or explicitly) having as target a woman or a group of women. Misogyny can be expressed in the form of shaming, stereotype, objectification and/or violence.
2. not-misogynistic - a meme that does not express any form of hate against women.
Classify the above meme as misogynistic or not-misogynistic based on the above definitions considering the image and the extracted text from the image delimited by three backticks.
Provide your answer in the format: misogynistic or not-misogynistic."""

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
            commit_message="Qwen3-VL-8B fine-tuned on MAMI dataset (QLoRA)",
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
class QwenMamiDataset(torch.utils.data.Dataset):
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
                    {"type": "text", "text": f"Extracted text: ```{s['text']}```\nProvide your answer in the format: misogynistic or not-misogynistic."}
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
    return inputs

# ---------------------------
# Main
# ---------------------------
def main():
    set_seed(SEED)

    # Load samples from TSV
    samples = []
    train_tsv = Path(TRAIN_TSV_PATH)
    image_dir = Path(IMAGE_DIR)
    if not train_tsv.exists():
        raise FileNotFoundError(f"Train file not found: {train_tsv}")
    with train_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            img_path = image_dir / row["file_name"]
            if not img_path.exists():
                print(f"Warning: missing image {img_path}, skipping")
                continue
            samples.append({
                "image_path": str(img_path),
                "text": row.get("text", ""),
                "label": "misogynistic" if int(row["label"]) == 1 else "not-misogynistic"
            })
    print(f"Loaded {len(samples)} samples")

    # small validation split
    if VALIDATION_SPLIT > 0 and len(samples) >= 50:
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
    train_dataset = QwenMamiDataset(train_samples)
    eval_dataset = QwenMamiDataset(val_samples) if val_samples else None

    # Training arguments with early stopping support
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

    # Save best model (already saved at end if load_best_model_at_end=True)
    print("Saving model and processor to", OUTPUT_DIR)
    trainer.save_model()  # saves the best model
    processor.save_pretrained(OUTPUT_DIR)

    # Upload to HF if token provided
    if HF_TOKEN and HF_TOKEN != "":
        print("Uploading to Hugging Face Hub...")
        upload_to_hf(OUTPUT_DIR, HF_UPLOAD_NAME, HF_TOKEN)
    else:
        print("HF upload skipped. Set HF_TOKEN to upload.")

    print("Done.")

if __name__ == "__main__":
    main()