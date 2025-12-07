"""
Test the fine-tuned Python Code Generation Model (CPU version)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
MODEL_DIR = "models/pythoncode-lora"  # Your trained LoRA model
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Base model used for training


def load_model():
    """Load the fine-tuned model"""
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_code(model, tokenizer, instruction: str, max_length: int = 512):
    """Generate Python code from instruction"""
    # Format the prompt
    prompt = f"""### Instruction:
{instruction}

### Response:
```python
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    if "### Response:" in generated_text:
        response = generated_text.split("### Response:")[1].strip()
        return response
    
    return generated_text


def main():
    """Main test function"""
    print("="*60)
    print("Python Code Generation Model - Test")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model()
    
    # Test prompts
    test_prompts = [
        "Write a function to calculate the factorial of a number",
        "Create a function that checks if a string is a palindrome",
        "Write a Python function to find the largest element in a list",
        "Create a function that reverses a string",
    ]
    
    print("\nTesting model with sample prompts...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'='*60}")
        print(f"Instruction: {prompt}")
        print("-"*60)
        
        code = generate_code(model, tokenizer, prompt)
        print(code)
    
    # Interactive mode
    print(f"\n{'='*60}")
    print("Interactive Mode")
    print("="*60)
    print("Enter your instruction (or 'quit' to exit):")
    
    while True:
        instruction = input("\n> ")
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            break
        
        if not instruction.strip():
            continue
        
        print("\nGenerating code...")
        code = generate_code(model, tokenizer, instruction)
        print(code)


if __name__ == "__main__":
    main()
