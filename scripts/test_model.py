"""
Test the fine-tuned model before exporting to Ollama
"""

import torch
from unsloth import FastLanguageModel


MODEL_PATH = "models/pythoncode-lora"
MAX_SEQ_LENGTH = 2048


def load_model():
    """Load the fine-tuned model"""
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Set to inference mode
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_code(model, tokenizer, instruction: str, max_length: int = 512):
    """Generate Python code from instruction"""
    prompt = f"""### Instruction:
{instruction}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in generated_text:
        response = generated_text.split("### Response:")[1].strip()
        return response
    return generated_text


def main():
    """Test the model with sample prompts"""
    print("="*50)
    print("Testing Python Code Generation Model")
    print("="*50)
    
    model, tokenizer = load_model()
    
    # Test prompts
    test_prompts = [
        "Create a function to calculate factorial of a number",
        "Write a function to check if a string is a palindrome",
        "Implement a binary search algorithm",
        "Create a class to represent a simple bank account with deposit and withdraw methods",
        "Write a function to reverse a linked list",
    ]
    
    print("\n" + "="*50)
    print("Generating Code Examples")
    print("="*50)
    
    for i, instruction in enumerate(test_prompts, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {instruction}")
        print('='*50)
        
        generated_code = generate_code(model, tokenizer, instruction)
        print(generated_code)
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*50)
    
    while True:
        instruction = input("\nEnter your instruction: ").strip()
        if instruction.lower() in ['quit', 'exit', 'q']:
            break
        
        if instruction:
            print("\nGenerating code...")
            generated_code = generate_code(model, tokenizer, instruction)
            print("\n" + generated_code)


if __name__ == "__main__":
    main()
