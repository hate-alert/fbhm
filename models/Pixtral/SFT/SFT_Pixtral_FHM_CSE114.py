#!/usr/bin/env python3
"""
Pixtral SFT training script (Trainer + QLoRA) on FHM dataset.
- Uniform image resizing to avoid patch mismatch
- Loss computed only on assistant response
- Optional HF upload
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # adjust to your GPUs
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
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
TRAIN_JSONL_PATH = "/home/du1/21CS30035/FacebookHatefulMemes/hateful_memes/train.jsonl"
IMAGE_DIR = "/home/du1/21CS30035/FacebookHatefulMemes/hateful_memes/img"
# MODEL_NAME        = "mistralai/Pixtral-12B-2409"
MODEL_NAME = "mistral-community/pixtral-12b"
OUTPUT_DIR        = "./pixtral-12b-sft-fhm-15-epochs-earlystop-2-h100"

HF_UPLOAD_NAME    = "pbhaskar/pixtral-12b-sft-fhm-15-epochs-earlystop-2-h100"
HF_TOKEN          = ""                         # set to enable upload

# training hyperparams (adjust based on GPU memory)
MAX_LENGTH                 = 4096
PER_DEVICE_BATCH_SIZE      = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS                 = 15
LEARNING_RATE              = 2e-5
WEIGHT_DECAY               = 0.01
BF16                       = True       # set False if hardware doesn't support bf16
SAVE_STEPS                 = 500
LOGGING_STEPS              = 20
SEED                       = 42
VALIDATION_SPLIT           = 0.1

# LoRA config
LORA_R      = 16
LORA_ALPHA  = 32
LORA_DROPOUT = 0.1

# System prompt (same as in the inference script)
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
            commit_message="Pixtral-12B fine-tuned on FHM dataset (QLoRA)",
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
class PixtralFhmDataset(torch.utils.data.Dataset):
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
                    {"type": "image"},   # ← placeholder only
                    {"type": "text", "text": f"Extracted text: ```{s['text']}```\nProvide your answer in the format: hateful or not-hateful."}
                ]
            },
            {"role": "assistant", "content": s["label"]}
        ]
        return {"conversation": conversation, "image": image}

# ---------------------------
# Helper: get target image size from processor
# ---------------------------
def detect_target_size(processor):

    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        size = processor.image_processor.size
    elif hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "size"):
        size = processor.feature_extractor.size
    else:
        print("Could not detect target size, using default 512x512")
        return 896, 896

    if isinstance(size, int):
        target_h = target_w = size
    elif isinstance(size, dict):
        target_h = size.get("height", size.get("shortest_edge", 896))
        target_w = size.get("width", size.get("shortest_edge", 896))
    else:
        target_w = target_h = 896

    target_w, target_h = int(target_w), int(target_h)
    print(f"Using target image size: {target_w}x{target_h}")
    return target_w, target_h

# ---------------------------
# Collator: resize images uniformly, call processor, mask labels
# ---------------------------
def data_collator(examples: List[Dict], processor: AutoProcessor, target_wh: tuple, max_length: int):
    images = [ex["image"] for ex in examples]
    convs = [ex["conversation"] for ex in examples]

    target_w, target_h = target_wh
    # Resize all images uniformly (to avoid patch count mismatch)
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

    # Process images and texts
    inputs = processor(
        text=texts,
        images=resized_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Create labels and mask tokens before assistant response
    labels = inputs["input_ids"].clone()
    tokenizer = processor.tokenizer

    # Find assistant start marker (Mistral uses [/INST])
    marker = "[/INST]"
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    if not marker_ids:
        # fallback to assistant content search if marker not found
        marker_ids = None

    batch_ids = inputs["input_ids"].tolist()
    for i, (input_ids, conv) in enumerate(zip(batch_ids, convs)):
        start_idx = -1
        if marker_ids:
            # Find last occurrence of marker (most likely the one before assistant)
            n = len(marker_ids)
            for j in range(len(input_ids) - n, -1, -1):
                if input_ids[j:j+n] == marker_ids:
                    start_idx = j + n
                    break
        if start_idx == -1:
            # Fallback: find assistant content directly
            assistant_content = conv[-1]["content"]
            assistant_ids = tokenizer.encode(assistant_content, add_special_tokens=False)
            # helper to find sublist
            def find_sublist(hay, needle):
                if not needle:
                    return -1
                n_needle = len(needle)
                for j in range(len(hay) - n_needle + 1):
                    if hay[j:j+n_needle] == needle:
                        return j
                return -1
            start_idx = find_sublist(input_ids, assistant_ids)

        if start_idx != -1:
            # Mask everything before assistant start
            labels[i, :start_idx] = -100
        else:
            # Ultimate fallback: keep only last 50 tokens unmasked
            print(f"Warning: Could not locate assistant response in input {i}, using fallback masking.")
            keep_last = 50
            if labels.size(1) > keep_last:
                labels[i, :-keep_last] = -100

    # Mask padding tokens
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    inputs["labels"] = labels


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

    # Load samples
    samples = []
    train_path = Path(TRAIN_JSONL_PATH)
    image_dir = Path(IMAGE_DIR)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    with train_path.open("r") as f:
        for line in f:
            data = json.loads(line)
            img_path = image_dir / Path(data["img"]).name
            if not img_path.exists():
                print(f"Warning: missing image {img_path}, skipping")
                continue
            samples.append({
                "image_path": str(img_path),
                "text": data.get("text", ""),
                "label": "hateful" if int(data.get("label", 0)) == 1 else "not-hateful"
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
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"   # important for causal LM
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    target_w, target_h = detect_target_size(processor)

    # Prepare model (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base model (quantized) — this may take a while...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if BF16 else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model.enable_input_require_grads()

    # LoRA target modules (common for Mistral-based models)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # If you want to adapt the vision encoder, you could add e.g. "vision_encoder.*" here
    ]
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=target_modules,
        lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = PixtralFhmDataset(train_samples)
    eval_dataset = PixtralFhmDataset(val_samples) if val_samples else None

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="none",
        push_to_hub=False,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",

        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        eval_accumulation_steps=1,
    )

    # Trainer with custom data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda examples: data_collator(examples, processor, (target_w, target_h), MAX_LENGTH),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    print("Starting training...")
    trainer.train()
    # trainer.train(
    #     resume_from_checkpoint="./pixtral-12b-sft-fhm-15-epochs-earlystop-2-h100/checkpoint-6000"
    # )

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