"""
Export trained model to Ollama-compatible GGUF format
"""

import os
import subprocess
from unsloth import FastLanguageModel


MODEL_PATH = "models/pythoncode-lora"
OUTPUT_DIR = "models/gguf"
QUANTIZATION = "q4_k_m"  # Good balance of size and quality


def merge_and_save_model():
    """Merge LoRA weights with base model"""
    print("Loading model with LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,  # Load in full precision for export
    )
    
    print("Merging LoRA weights with base model...")
    # Merge LoRA weights
    model = FastLanguageModel.for_inference(model)
    
    # Save merged model
    merged_dir = "models/pythoncode-merged"
    os.makedirs(merged_dir, exist_ok=True)
    
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    print("Model merged successfully!")
    return merged_dir


def convert_to_gguf(merged_model_path: str):
    """Convert merged model to GGUF format"""
    print("\n" + "="*50)
    print("Converting to GGUF format...")
    print("="*50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save as GGUF using Unsloth's built-in method
    print(f"Saving as GGUF with {QUANTIZATION} quantization...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=merged_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    
    # Save as GGUF
    output_path = f"{OUTPUT_DIR}/pythoncode-{QUANTIZATION}.gguf"
    model.save_pretrained_gguf(
        output_path,
        tokenizer,
        quantization_method=QUANTIZATION,
    )
    
    print(f"\nGGUF model saved to: {output_path}")
    return output_path


def create_modelfile(gguf_path: str):
    """Create Modelfile for Ollama"""
    modelfile_content = f"""# Python Code Generation Model
# Fine-tuned CodeLlama-7B on 25K Python code examples

FROM {gguf_path}

# System prompt
SYSTEM \"\"\"You are an expert Python programmer. Generate clean, efficient, and well-documented Python code based on the user's instructions. Always wrap your code in markdown code blocks with ```python syntax.\"\"\"

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# Stop tokens
PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"
"""
    
    modelfile_path = "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"\nModelfile created: {modelfile_path}")
    return modelfile_path


def main():
    """Main export function"""
    print("="*50)
    print("Exporting Model to Ollama Format")
    print("="*50)
    
    # Step 1: Merge LoRA with base model
    merged_path = merge_and_save_model()
    
    # Step 2: Convert to GGUF
    gguf_path = convert_to_gguf(merged_path)
    
    # Step 3: Create Modelfile
    modelfile_path = create_modelfile(gguf_path)
    
    print("\n" + "="*50)
    print("Export Complete!")
    print("="*50)
    print(f"\nGGUF Model: {gguf_path}")
    print(f"Modelfile: {modelfile_path}")
    print("\nNext steps:")
    print("1. Install Ollama: https://ollama.ai")
    print("2. Create model: ollama create pythoncode -f Modelfile")
    print("3. Run model: ollama run pythoncode 'Create a function to calculate fibonacci'")
    

if __name__ == "__main__":
    main()
