"""
Export the fine-tuned model to Ollama format (CPU version)
Merges LoRA weights and creates an Ollama-compatible model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Configuration
MODEL_DIR = "models/pythoncode-lora"  # Your trained LoRA model
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "models/pythoncode-merged"  # Merged model output


def merge_and_save():
    """Merge LoRA weights with base model and save"""
    print("="*60)
    print("Merging LoRA weights with base model...")
    print("="*60)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("2. Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None,
    )
    
    # Load LoRA adapter
    print("3. Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    
    # Merge LoRA weights into base model
    print("4. Merging weights...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"5. Saving merged model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Model merged and saved successfully!")
    return OUTPUT_DIR


def create_modelfile(model_path: str):
    """Create Ollama Modelfile"""
    modelfile_content = f"""# Python Code Generator
# Fine-tuned Qwen2.5-0.5B for Python code generation

FROM {model_path}

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt
SYSTEM \"\"\"You are an expert Python programmer. When given instructions, you write clean, efficient, and well-documented Python code. Always include docstrings and comments where appropriate.\"\"\"

# Template for code generation
TEMPLATE \"\"\"### Instruction:
{{ .Prompt }}

### Response:
```python
"""

    modelfile_path = os.path.join(model_path, "Modelfile")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"\n✅ Modelfile created at: {modelfile_path}")
    return modelfile_path


def main():
    """Main export function"""
    print("="*60)
    print("Export Model to Ollama Format")
    print("="*60)
    
    # Merge LoRA weights
    merged_path = merge_and_save()
    
    # Create Modelfile
    modelfile = create_modelfile(merged_path)
    
    # Print next steps
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("\nTo use with Ollama:")
    print(f"\n1. Create Ollama model:")
    print(f"   ollama create pythoncode -f {modelfile}")
    print(f"\n2. Test the model:")
    print(f"   ollama run pythoncode \"Write a function to calculate fibonacci numbers\"")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
