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

class DebiasedInferenceOProj:
    """
    Manages the process of running inference on Qwen3-VL while applying
    bias mitigation at the O_proj level using expand/repeat methods.
    """
    def __init__(self, model, processor, vectors_path, method="expand_direct"):
        """
        Args:
            method: 
                "expand_direct" - repeat bias vector and direct subtraction
                "expand_project" - repeat bias vector and orthogonal projection
        """
        self.model = model
        self.processor = processor
        self.device = self.model.device
        self.method = method

        vector_data = torch.load(vectors_path, map_location='cpu', weights_only=False)
        
        # Extract from the updated structure
        self.bias_vectors = vector_data['bvs'] 
        self.influential_heads_map = vector_data['influential_heads_map']
        self.influential_scores_map = vector_data['influential_scores_map']
        self.global_influential_heads_with_scores = vector_data['global_influential_heads_with_scores']
        
        # Qwen3-VL specific config
        self.num_heads = self.model.config.text_config.num_attention_heads
        self.head_dim = self.model.config.text_config.hidden_size // self.num_heads
        self.hidden_size = self.model.config.text_config.hidden_size

        self.max_new_tokens = 50
        self.temperature = 0.001
        
        print(f"Successfully loaded vectors. Found {len(self.bias_vectors)} bias vectors.")
        print(f"Model config: {self.num_heads} heads, {self.head_dim} head_dim, {self.hidden_size} hidden_size")
        print(f"Model device: {self.device}")
        print(f"Debiasing method: {self.method}")
        
        # Print some info about the loaded vectors
        print(f"Global influential heads: {len(self.global_influential_heads_with_scores)} heads")
        if self.global_influential_heads_with_scores:
            sample_head, sample_score = self.global_influential_heads_with_scores[0]
            print(f"Sample head: {sample_head}, score: {sample_score:.6f}")

    def _get_attention_layer(self, layer_idx):
        """
        Get attention layer for Qwen3-VL
        """
        return self.model.model.language_model.layers[layer_idx].self_attn

    def _get_top_heads(self, top_x):
        """
        Returns the top-x most influential heads from the global list.
        If top_x is None, returns all heads.
        """
        if top_x is None or top_x >= len(self.global_influential_heads_with_scores):
            return self.global_influential_heads_with_scores
        else:
            return self.global_influential_heads_with_scores[:top_x]

    def _expand_bias_vector(self, bias_vector, target_device):
        """
        Expands bias vector from head_dim to hidden_size by repeating across heads.
        """
        # Move bias vector to target device first
        bias_vector_device = bias_vector.to(target_device)
        
        # Repeat the bias vector for all heads
        if bias_vector_device.shape[0] == self.head_dim:
            expanded_bias = bias_vector_device.repeat(self.num_heads)
        else:
            expanded_bias = bias_vector_device
        
        # Final safety check
        if expanded_bias.shape[0] != self.hidden_size:
            # Truncate or pad to match hidden_size
            if expanded_bias.shape[0] < self.hidden_size:
                padding = torch.zeros(self.hidden_size - expanded_bias.shape[0], 
                                    device=expanded_bias.device, 
                                    dtype=expanded_bias.dtype)
                expanded_bias = torch.cat([expanded_bias, padding])
            else:
                expanded_bias = expanded_bias[:self.hidden_size]
        
        return expanded_bias

    def _create_o_proj_expand_direct_hook(self, bias_vector, alpha, layer_idx):
        """
        Method 1: Expand bias vector and direct subtraction from O_proj output
        """
        def debiasing_hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            original_output = output[0] if is_tuple else output
            
            # Get the device of the output tensor
            target_device = original_output.device
            
            # Expand bias vector on the correct device
            expanded_bias = self._expand_bias_vector(bias_vector, target_device)
            
            # Apply direct subtraction to the final token's output
            debiased_output = original_output.clone()
            debiased_output[:, -1, :] = original_output[:, -1, :] - (alpha * expanded_bias)
            
            if is_tuple:
                return (debiased_output,) + output[1:]
            else:
                return debiased_output
                
        return debiasing_hook

    def _create_o_proj_expand_project_hook(self, bias_vector, alpha, layer_idx):
        """
        Method 2: Expand bias vector and orthogonal projection
        """
        def debiasing_hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            original_output = output[0] if is_tuple else output
            
            # Get the device of the output tensor
            target_device = original_output.device
            
            # Expand bias vector on the correct device
            expanded_bias = self._expand_bias_vector(bias_vector, target_device)
            
            batch_size, seq_len, hidden_dim = original_output.shape
            
            # Normalize the bias direction
            bias_norm = torch.norm(expanded_bias) + 1e-9
            bias_direction = expanded_bias / bias_norm
            
            # Get the final token's representation
            final_token_repr = original_output[:, -1, :]  # [batch, hidden_dim]
            
            # Compute projection onto bias direction
            projection = torch.sum(final_token_repr * bias_direction, dim=-1, keepdim=True)  # [batch, 1]
            
            # Remove the bias component using orthogonal projection
            debiased_final_token = final_token_repr - (alpha * projection * bias_direction)
            
            # Put back
            debiased_output = original_output.clone()
            debiased_output[:, -1, :] = debiased_final_token
            
            if is_tuple:
                return (debiased_output,) + output[1:]
            else:
                return debiased_output
                
        return debiasing_hook

    def safe_image_load(self, path):
        """Safely load images with error handling"""
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return Image.new('RGB', (224, 224), color='white')

    def predict(self, meme_data, alpha=1.0, top_x_heads=None):
        """
        Runs a prediction for a single meme and returns the RAW generated text.
        
        Args:
            meme_data: Dictionary containing meme information
            alpha: Strength of debiasing
            top_x_heads: Number of top influential heads to use (None = use all)
        """
        functionality = meme_data['Functionality']
        target = meme_data['Target']
        
        bias_vector_key = (functionality, target)
        
        hooks = []
        applied_debiasing = False
        
        if alpha > 0 and bias_vector_key in self.bias_vectors:
            
            bias_vector = self.bias_vectors[bias_vector_key]
            
            # Get top-x heads from global influential heads
            top_heads_to_use = self._get_top_heads(top_x_heads)
            
            # Debug information about head selection
            # print(f"Found bias vector for {bias_vector_key}, shape: {bias_vector.shape}")
            # print(f"Using {self.method} method with alpha={alpha}, top_x_heads={top_x_heads}")
            # print(f"Using top {len(top_heads_to_use)} heads out of {len(self.global_influential_heads_with_scores)} available")
            
            # Get unique layers from the selected top heads
            layers_to_debias = set()
            for ((layer_idx, head_idx), score) in top_heads_to_use:
                if layer_idx < len(self.model.model.language_model.layers):
                    layers_to_debias.add(layer_idx)
            
            # print(f"Applying debiasing to {len(layers_to_debias)} layers")
            
            for layer_idx in layers_to_debias:
                try:
                    # Hook the O_proj module of the self-attention for Qwen3-VL
                    layer_to_hook = self._get_attention_layer(layer_idx).o_proj
                    
                    # Choose the appropriate hook based on method
                    if self.method == "expand_direct":
                        hook_fn = self._create_o_proj_expand_direct_hook(bias_vector, alpha, layer_idx)
                    elif self.method == "expand_project":
                        hook_fn = self._create_o_proj_expand_project_hook(bias_vector, alpha, layer_idx)
                    else:
                        raise ValueError(f"Unknown method: {self.method}")
                    
                    hook_handle = layer_to_hook.register_forward_hook(hook_fn)
                    hooks.append(hook_handle)
                    
                    # print(f"  Registered O_proj hook for layer {layer_idx} using {self.method} method")
                    
                except Exception as e_hook:
                    print(f"  ERROR registering hook for layer {layer_idx}: {e_hook}")
            
            applied_debiasing = bool(hooks)

        if not applied_debiasing:
            print(f"Running standard (biased) inference for {bias_vector_key}.")

        # Prepare prompt and images - Qwen3-VL format
        image = self.safe_image_load(meme_data['Image_Path'])
        extracted_text = meme_data["text"]
        
        # Qwen3-VL uses a different message format
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
        
        inputs = self.processor(
            text=[text],
            images=[image],
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
    TEST_DATA_PATH = os.path.join(BASE_DATA_DIR, 'test.csv')
    VECTORS_PATH = "batch_4_cross_target_bias_vectors_with_scores_FBHM_qwen.pth"  # Your Qwen3 vectors
    TARGET_COMMUNITIES = [
        'muslims', 'jews', 'blacks', 'whites', 'women', 'men',
        'transgenders', 'gays', 'immigrants', 'disabled'
    ]
    
    # Hyperparameters to test
    # ALPHA_VALUES = [0.5, 1, 10, 30, 50, 60]
    # TOP_X_HEADS_VALUES = [2, 5, 10, 20, 30, 40, 50]  # Different numbers of top heads to test

    ALPHA_VALUES = [0.5, 1, 10, 30, 50]
    TOP_X_HEADS_VALUES = [50]
    
    # Choose debiasing methods
    DEBIASING_METHODS = [
        "expand_direct",    # Repeat bias vector and direct subtraction
        # "expand_project",   # Repeat bias vector and orthogonal projection
    ]
    
    # --- Step 1: Pre-process the TEST data ---
    test_df = preprocess_fbhm_data(TEST_DATA_PATH, BASE_DATA_DIR, TARGET_COMMUNITIES)
    
    # --- Step 2: Load Model and Processor ---
    print("\nLoading model and processor...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model loaded successfully.")

    # --- Step 3: Run predictions for different alpha values, top_x values, and methods ---
    for method in DEBIASING_METHODS:
        for top_x in TOP_X_HEADS_VALUES:
            for alpha in ALPHA_VALUES:
                print(f"\n{'='*80}")
                print(f"Running O_proj {method} debiasing with top_x = {top_x}, alpha = {alpha}")
                print(f"{'='*80}")
                
                # Initialize the Debiasing Framework
                debiaser = DebiasedInferenceOProj(model, processor, VECTORS_PATH, method=method)
                
                # Output file path for this combination
                OUTPUT_TXT = f"b4_cross_target_o_proj_{method}_topx_{top_x}_alpha_{alpha}_debiased_QWEN3.txt"
                
                # Run Predictions and Log to Text File
                with open(OUTPUT_TXT, "w", encoding="utf-8") as output_file:
                    
                    for index, meme in tqdm(test_df.iterrows(), total=len(test_df), 
                                          desc=f"Method={method}, TopX={top_x}, Alpha={alpha}"):
                        
                        meme_dict = meme.to_dict()
                        output_file.write(json.dumps(meme_dict))
                        output_file.write('\n----------\n')
                        
                        # Run debiased inference with current parameters
                        generated_texts = debiaser.predict(meme, alpha=alpha, top_x_heads=top_x)

                        for text in generated_texts:
                            # Replace 'assistant' (case-insensitive) with 'Assistant:'
                            formatted_text = re.sub(r'\bassistant\b', 'Assistant:', text, flags=re.IGNORECASE)
                            output_file.write(formatted_text)
                            print(formatted_text)
                        
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

    print(f"\nCompleted all O_proj debiasing experiments!")