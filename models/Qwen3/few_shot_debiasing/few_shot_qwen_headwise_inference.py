# ! pip install transformers torch Pillow pandas numpy tqdm accelerate bitsandbytes sentencepiece scikit-learn

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import os
import random
import re
import json
import gc

def preprocess_fbhm_data(csv_path, base_dir, target_communities):
    """
    Loads and processes the raw FBHM train.csv/test.csv to create a usable DataFrame.
    """
    print(f"--- Pre-processing real data from: {csv_path} ---")
    df = pd.read_csv(csv_path)
    # Ensure all required columns exist
    for col in ['img', 'label', 'text']:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain a '{col}' column.")

    df['Image_Path'] = df['img'].apply(lambda x: os.path.join(base_dir, x))
    df['Functionality'] = df['img'].str.extract(r'(F\d+)')[0]
    df['Target'] = df.groupby('Functionality').cumcount() % len(target_communities)
    df['Target'] = df['Target'].apply(lambda x: target_communities[x])
    label_map = {1: 'hateful', 0: 'not-hateful'}
    df['Label'] = df['label'].map(label_map)
    final_df = df[['img_id', 'img', 'Image_Path', 'Functionality', 'Target', 'text', 'Label', 'label']].copy()
    print("Pre-processing complete. Final data shape:", final_df.shape)
    return final_df

def create_few_shot_examples(train_df, num_examples_per_functionality=2):
    """
    Creates few-shot examples by sampling from each functionality.
    
    Args:
        train_df: DataFrame containing training data
        num_examples_per_functionality: Number of examples to sample from each functionality
    
    Returns:
        List of dictionaries containing few-shot examples
    """
    few_shot_examples = []
    
    # Get all unique functionalities
    functionalities = sorted(train_df['Functionality'].unique())
    print(f"Found {len(functionalities)} functionalities: {functionalities}")
    
    for functionality in functionalities:
        # Get examples from this functionality
        func_examples = train_df[train_df['Functionality'] == functionality]
        
        # Sample specified number of examples
        if len(func_examples) >= num_examples_per_functionality:
            sampled_examples = func_examples.sample(n=num_examples_per_functionality, random_state=42)
        else:
            sampled_examples = func_examples
            
        for _, example in sampled_examples.iterrows():
            few_shot_examples.append({
                'functionality': functionality,
                'image_path': example['Image_Path'],
                'text': example['text'],
                'label': example['Label']
            })
    
    print(f"Created {len(few_shot_examples)} few-shot examples")
    return few_shot_examples

class DebiasedInference:
    """
    Manages the process of running inference on Qwen3-VL while applying
    bias mitigation by subtracting a Bias Vector at specified layers.
    """
    def __init__(self, model, processor, vectors_path):
        self.model = model
        self.processor = processor
        self.device = self.model.device

        vector_data = torch.load(vectors_path, map_location='cpu')  # Load on CPU first
        
        # Extract from the updated structure
        self.bias_vectors = vector_data['bvs'] 
        self.influential_heads_map = vector_data['influential_heads_map']
        self.influential_scores_map = vector_data['influential_scores_map']
        self.global_influential_heads_with_scores = vector_data['global_influential_heads_with_scores']
        
        # Qwen3-VL specific config
        self.num_heads = self.model.config.text_config.num_attention_heads
        self.head_dim = self.model.config.text_config.hidden_size // self.num_heads

        self.max_new_tokens = 50
        self.temperature = 0.001
        
        print(f"Successfully loaded vectors. Found {len(self.bias_vectors)} bias vectors.")
        print(f"Model config: {self.num_heads} heads, {self.head_dim} head_dim")
        print(f"Model device: {self.device}")
        
        # Print some info about the loaded vectors
        print(f"Global influential heads: {len(self.global_influential_heads_with_scores)} heads")
        if self.global_influential_heads_with_scores:
            sample_head, sample_score = self.global_influential_heads_with_scores[0]
            print(f"Sample head: {sample_head}, score: {sample_score:.6f}")

    def _get_attention_layer(self, layer_idx):
        """
        Get attention layer for Qwen3-VL - same as in extraction
        """
        return self.model.model.language_model.layers[layer_idx].self_attn

    def _get_top_heads(self, influential_heads_for_target, top_x):
        """
        Returns the top-x most influential heads from the given list.
        If top_x is None, returns all heads.
        """
        if top_x is None or top_x >= len(influential_heads_for_target):
            return influential_heads_for_target
        else:
            return influential_heads_for_target[:top_x]

    def _create_head_specific_hook(self, bias_vector, alpha, layer_idx, head_idx):
        """
        Creates a hook that applies orthogonal projection to a SPECIFIC head's output.
        """
        def debiasing_hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            original_hidden_state = output[0] if is_tuple else output
            
            # Move bias vector to the same device as hidden state
            bias_vector_device = bias_vector.to(original_hidden_state.device)
            
            # Reshape to access individual heads: [batch, seq, num_heads, head_dim]
            batch_size, seq_len, hidden_dim = original_hidden_state.shape
            attn_reshaped = original_hidden_state.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Get the specific head's activation for the final token
            head_activation = attn_reshaped[:, -1, head_idx, :].to(torch.float32)
            
            # Normalize the bias direction (on the same device)
            bias_direction = bias_vector_device.to(torch.float32) 
            bias_norm = torch.linalg.norm(bias_direction) + 1e-9
            bias_direction = bias_direction / bias_norm
            
            # Compute projection onto bias direction
            projection_scalar = torch.matmul(head_activation.unsqueeze(1), bias_direction.unsqueeze(-1)).squeeze(-1)
            projection_vector = projection_scalar * bias_direction
            
            # Apply orthogonal projection (remove bias component)
            debiased_head_activation = head_activation - (alpha * projection_vector)

            # Apply direct subtraction
            # debiased_head_activation = head_activation - (alpha * bias_direction)
            
            # Put back into the tensor
            attn_reshaped[:, -1, head_idx, :] = debiased_head_activation.to(original_hidden_state.dtype)
            
            # Reshape back
            debiased_hidden_state = attn_reshaped.view(batch_size, seq_len, hidden_dim)
            
            if is_tuple:
                return (debiased_hidden_state,) + output[1:]
            else:
                return debiased_hidden_state
                
        return debiasing_hook

    def safe_image_load(self, path):
        """Safely load images with error handling"""
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return Image.new('RGB', (224, 224), color='white')

    def create_few_shot_prompt(self, test_meme_data, few_shot_examples):
        """
        Creates a few-shot prompt with examples and the test instance.
        
        Args:
            test_meme_data: Dictionary containing test meme information
            few_shot_examples: List of few-shot example dictionaries
        
        Returns:
            Formatted messages for the model
        """
        # System message
        system_message = {
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
        }
        
        messages = [system_message]
        
        # Add few-shot examples
        for example in few_shot_examples:
            # Load example image
            example_image = self.safe_image_load(example['image_path'])
            
            # User message for example
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": example_image},
                    {"type": "text", "text": f"Extracted text: ```{example['text']}```\nClassify this meme as hateful or not-hateful."}
                ]
            }
            
            # Assistant response for example
            assistant_message = {
                "role": "assistant",
                "content": example['label']
            }
            
            messages.extend([user_message, assistant_message])
        
        # Add test instance
        test_image = self.safe_image_load(test_meme_data['Image_Path'])
        test_text = test_meme_data['text']
        
        test_user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": f"Extracted text: ```{test_text}```\nClassify this meme as hateful or not-hateful."}
            ]
        }
        
        messages.append(test_user_message)
        
        return messages

    def predict(self, meme_data, few_shot_examples=None, alpha=1.0, top_x_heads=None):
        """
        Runs a prediction for a single meme with few-shot examples and returns the RAW generated text.
        
        Args:
            meme_data: Dictionary containing meme information
            few_shot_examples: List of few-shot examples for in-context learning
            alpha: Strength of debiasing
            top_x_heads: Number of top influential heads to use (None = use all)
        """
        functionality = meme_data['Functionality']
        target = meme_data['Target']
        
        bias_vector_key = (functionality, target)
        influential_heads_key = (functionality, target) 
        
        hooks = []
        applied_debiasing = False
        
        if alpha > 0 and bias_vector_key in self.bias_vectors and influential_heads_key in self.influential_heads_map:
            
            bias_vector = self.bias_vectors[bias_vector_key]
            influential_heads_for_target = self.influential_heads_map[influential_heads_key]
            
            # Get top-x heads
            top_heads_to_use = self._get_top_heads(influential_heads_for_target, top_x_heads)
            
            if top_heads_to_use:
                # Register ONE HOOK PER INFLUENTIAL HEAD
                for layer_idx, head_idx in top_heads_to_use:
                    try:
                        # Create head-specific hook
                        hook_fn = self._create_head_specific_hook(bias_vector, alpha, layer_idx, head_idx)
                        
                        # Hook the self-attention module for Qwen3-VL
                        layer_to_hook = self._get_attention_layer(layer_idx)
                        hook_handle = layer_to_hook.register_forward_hook(hook_fn)
                        hooks.append(hook_handle)
                        
                    except Exception as e_hook:
                        print(f"  ERROR registering hook for layer {layer_idx}, head {head_idx}: {e_hook}")
                
                applied_debiasing = bool(hooks)

        if not applied_debiasing:
            print(f"Running standard (biased) inference for {bias_vector_key}.")

        # Prepare prompt - with or without few-shot examples
        if few_shot_examples:
            messages = self.create_few_shot_prompt(meme_data, few_shot_examples)
        else:
            # Zero-shot prompt (original)
            image = self.safe_image_load(meme_data['Image_Path'])
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

        # Apply Qwen3-VL chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Extract all images from messages
        images = []
        for message in messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if content["type"] == "image":
                        images.append(content["image"])
        
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        generated_texts = ""
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens, 
                    temperature=self.temperature, 
                    do_sample=self.temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
        except Exception as e:
            print(f"Error during generation for index {meme_data.name}: {e}")
            generated_texts = [f"Error: {e}"]
        finally:
            for hook in hooks:
                hook.remove()

        return generated_texts

if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    BASE_DATA_DIR = './FBHM'
    TRAIN_DATA_PATH = os.path.join(BASE_DATA_DIR, 'train.csv')  # For few-shot examples
    TEST_DATA_PATH = os.path.join(BASE_DATA_DIR, 'test.csv')
    VECTORS_PATH = "batch_4_cross_target_bias_vectors_with_scores_FBHM_qwen.pth"  # Your Qwen3 vectors
    TARGET_COMMUNITIES = [
        'muslims', 'jews', 'blacks', 'whites', 'women', 'men',
        'transgenders', 'gays', 'immigrants', 'disabled'
    ]
    
    # Hyperparameters to test
    ALPHA_VALUES = [1, 30]
    TOP_X_HEADS_VALUES = [30, 50]  # Different numbers of top heads to test
    
    # --- Step 1: Pre-process the TRAIN and TEST data ---
    train_df = preprocess_fbhm_data(TRAIN_DATA_PATH, BASE_DATA_DIR, TARGET_COMMUNITIES)
    test_df = preprocess_fbhm_data(TEST_DATA_PATH, BASE_DATA_DIR, TARGET_COMMUNITIES)
    
    # --- Step 2: Create few-shot examples ---
    few_shot_examples = create_few_shot_examples(train_df, num_examples_per_functionality=2)
    print(f"Created {len(few_shot_examples)} few-shot examples for in-context learning")
    
    # --- Step 3: Load Model and Processor ---
    print("\nLoading model and processor...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:1",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model loaded successfully.")

    # --- Step 4: Run predictions for different alpha and top_x_heads values ---
    for top_x in TOP_X_HEADS_VALUES:
        for alpha in ALPHA_VALUES:
            print(f"\n{'='*60}")
            print(f"Running FEW-SHOT debiased inference with top_x = {top_x}, alpha = {alpha}")
            print(f"Using {len(few_shot_examples)} few-shot examples")
            print(f"{'='*60}")
            
            # Initialize the Debiasing Framework
            debiaser = DebiasedInference(model, processor, VECTORS_PATH)
            
            # Output file path for this combination
            OUTPUT_TXT = f"b4_few_shot_cross_target_head_projection_debiased_QWEN3_FEWSHOT_topx_{top_x}_alpha_{alpha}.txt"
            
            # Run Predictions and Log to Text File
            with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
                
                for index, meme in tqdm(test_df.iterrows(), total=len(test_df), 
                                       desc=f"FewShot TopX={top_x}, Alpha={alpha}"):
                    
                    meme_dict = meme.to_dict()
                    output_file.write(json.dumps(meme_dict))
                    output_file.write('\n----------\n')
                    
                    # Run FEW-SHOT debiased inference with current parameters
                    generated_texts = debiaser.predict(meme, 
                                                     few_shot_examples=few_shot_examples,
                                                     alpha=alpha, 
                                                     top_x_heads=top_x)

                    for text in generated_texts:
                        # Replace 'assistant' (case-insensitive) with 'Assistant:'
                        formatted_text = re.sub(r'\bassistant\b', 'Assistant:', text, flags=re.IGNORECASE)
                        output_file.write(formatted_text)
                        # print(formatted_text)
                    
                    output_file.write("\n##########\n")
                    
                    # Optional garbage collection for long runs
                    if (index + 1) % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            print(f"Output file saved to '{OUTPUT_TXT}'")
            
            # Clean up to free memory
            del debiaser
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nCompleted all FEW-SHOT experiments!")
