"""
Unified Model Training Script
Easy-to-configure training for fine-tuning language models on custom datasets
"""

import torch
import json
import os
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
from typing import Optional
import re


# ============================================================
# CONFIGURATION - Edit these settings
# ============================================================

class Config:
    """Training configuration - modify these values as needed"""
    
    # Model Selection
    BASE_MODELS = {
        "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",       # Small, fast (CPU-friendly)
        "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",       # Medium (still CPU-friendly)
        "codellama-7b": "codellama/CodeLlama-7b-hf",     # Large (GPU recommended)
        "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",  # Code-focused
    }
    
    # Dataset Options
    DATASETS = {
        "python-18k": "iamtarun/python_code_instructions_18k_alpaca",
        "code-alpaca": "sahil2801/CodeAlpaca-20k",
        "local": "python-codes-25k.json",  # Your local file
    }
    
    # === CHANGE THESE TO CUSTOMIZE YOUR TRAINING === #
    SELECTED_MODEL = "qwen-0.5b"          # Choose from BASE_MODELS keys
    SELECTED_DATASET = "python-18k"       # Choose from DATASETS keys
    
    # Training Parameters
    MAX_SEQ_LENGTH = 512
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 2e-4
    
    # LoRA Parameters
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    # Paths
    OUTPUT_DIR = "models/finetuned-model"
    PROCESSED_DATA_DIR = "data/processed"
    
    # Training Subset (set to None to use full dataset)
    TRAIN_SUBSET_SIZE = 1000  # Use 1000 examples for quick testing
    VAL_SUBSET_SIZE = 100
    
    # Device
    USE_CPU = True  # Set to False if you have a GPU


# ============================================================
# Data Processing Functions
# ============================================================

def extract_code_from_output(output: str) -> str:
    """Extract Python code from markdown code blocks"""
    code_match = re.search(r'```python\n(.*?)\n```', output, re.DOTALL)
    if code_match:
        return code_match.group(1)
    return output


def format_instruction(instruction: str, output: str) -> dict:
    """Format data in instruction-following format"""
    code = extract_code_from_output(output)
    
    prompt = f"""### Instruction:
{instruction.strip()}

### Response:
```python
{code.strip()}
```"""
    
    return {
        "text": prompt,
        "instruction": instruction.strip(),
        "output": code.strip()
    }


def load_and_prepare_dataset(dataset_name: str, config: Config) -> DatasetDict:
    """Load and prepare dataset from Hugging Face or local file"""
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*60}")
    
    if dataset_name == "local":
        # Load from local JSON file
        with open(config.DATASETS[dataset_name], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            if 'instruction' in item and 'output' in item:
                formatted_data.append(format_instruction(item['instruction'], item['output']))
        
        df = pd.DataFrame(formatted_data)
        dataset = Dataset.from_pandas(df)
        
    else:
        # Load from Hugging Face
        dataset = load_dataset(config.DATASETS[dataset_name], split="train")
        
        # Format the data
        formatted_data = []
        for item in dataset:
            if 'instruction' in item and 'output' in item:
                formatted_data.append(format_instruction(item['instruction'], item['output']))
        
        dataset = Dataset.from_dict({k: [d[k] for d in formatted_data] for k in formatted_data[0].keys()})
    
    # Split into train/val
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })
    
    print(f"✓ Loaded {len(dataset_dict['train'])} training examples")
    print(f"✓ Loaded {len(dataset_dict['validation'])} validation examples")
    
    return dataset_dict


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples and add labels"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    # Labels are required for causal LM training
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ============================================================
# Model Loading Functions
# ============================================================

def load_model_and_tokenizer(config: Config):
    """Load base model with LoRA adapters"""
    model_name = config.BASE_MODELS[config.SELECTED_MODEL]
    
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Device: {'CPU' if config.USE_CPU else 'GPU'}")
    print(f"{'='*60}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    device_config = {
        "torch_dtype": torch.float32 if config.USE_CPU else torch.float16,
        "device_map": None if config.USE_CPU else "auto",
    }
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **device_config)
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# ============================================================
# Training Function
# ============================================================

def train_model(config: Config):
    """Main training function"""
    
    print("\n" + "="*60)
    print("FINE-TUNING CONFIGURATION")
    print("="*60)
    print(f"Model: {config.SELECTED_MODEL} ({config.BASE_MODELS[config.SELECTED_MODEL]})")
    print(f"Dataset: {config.SELECTED_DATASET}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Max sequence length: {config.MAX_SEQ_LENGTH}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    
    if config.USE_CPU:
        print("\n⚠️  WARNING: Training on CPU - this will be slow!")
        print("Consider using Google Colab for GPU training.")
    
    # Confirm before starting
    response = input("\nProceed with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Load dataset
    dataset = load_and_prepare_dataset(config.SELECTED_DATASET, config)
    
    # Use subset if specified
    if config.TRAIN_SUBSET_SIZE:
        dataset['train'] = dataset['train'].select(range(min(config.TRAIN_SUBSET_SIZE, len(dataset['train']))))
        dataset['validation'] = dataset['validation'].select(range(min(config.VAL_SUBSET_SIZE, len(dataset['validation']))))
        print(f"\n✓ Using subset: {len(dataset['train'])} train, {len(dataset['validation'])} val examples")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=dataset['train'].column_names,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=50,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        save_total_limit=2,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
    )
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print(f"Model saved to: {config.OUTPUT_DIR}")
    print("="*60)
    
    return config.OUTPUT_DIR


# ============================================================
# Testing Function
# ============================================================

def test_model(model_dir: str, config: Config):
    """Test the fine-tuned model"""
    from peft import PeftModel
    
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60)
    
    print("\nLoading model...")
    base_model_name = config.BASE_MODELS[config.SELECTED_MODEL]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map=None,
    )
    
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    
    print("Model loaded!\n")
    
    # Test prompts
    test_prompts = [
        "Write a function to calculate factorial",
        "Create a function to check if a number is prime",
        "Write a function to reverse a string",
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print("-"*60)
        
        full_prompt = f"""### Instruction:
{prompt}

### Response:
```python
"""
        
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response
        if "### Response:" in generated:
            response = generated.split("### Response:")[1].strip()
            print(response)


# ============================================================
# Main Function
# ============================================================

def select_model(config: Config) -> str:
    """Interactive model selection"""
    print("\n" + "="*60)
    print("SELECT BASE MODEL")
    print("="*60)
    
    models = list(config.BASE_MODELS.keys())
    descriptions = {
        "qwen-0.5b": "500MB, Fast, CPU-friendly",
        "qwen-1.5b": "1.5GB, Medium speed, Better quality",
        "codellama-7b": "13GB, Slow on CPU, Best quality (GPU recommended)",
        "deepseek-1.3b": "1.3GB, Code-focused, Medium speed",
    }
    
    for i, model_key in enumerate(models, 1):
        desc = descriptions.get(model_key, "")
        print(f"{i}. {model_key:15} - {desc}")
    
    while True:
        choice = input(f"\nSelect model (1-{len(models)}) or press Enter for default [{config.SELECTED_MODEL}]: ").strip()
        
        if not choice:
            return config.SELECTED_MODEL
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print("Invalid choice! Try again.")
        except ValueError:
            print("Please enter a number!")


def select_dataset(config: Config) -> str:
    """Interactive dataset selection"""
    print("\n" + "="*60)
    print("SELECT DATASET")
    print("="*60)
    
    datasets = list(config.DATASETS.keys())
    descriptions = {
        "python-18k": "18,612 Python code examples",
        "code-alpaca": "20,000 multi-language code examples",
        "local": "Your custom local file",
    }
    
    for i, dataset_key in enumerate(datasets, 1):
        desc = descriptions.get(dataset_key, "")
        print(f"{i}. {dataset_key:15} - {desc}")
    
    while True:
        choice = input(f"\nSelect dataset (1-{len(datasets)}) or press Enter for default [{config.SELECTED_DATASET}]: ").strip()
        
        if not choice:
            return config.SELECTED_DATASET
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                return datasets[idx]
            else:
                print("Invalid choice! Try again.")
        except ValueError:
            print("Please enter a number!")


def main():
    """Main entry point"""
    config = Config()
    
    print("\n" + "="*60)
    print("MODEL FINE-TUNING TOOL")
    print("="*60)
    
    # Interactive model and dataset selection
    print("\n🎯 Let's configure your training!")
    config.SELECTED_MODEL = select_model(config)
    config.SELECTED_DATASET = select_dataset(config)
    
    print("\nAvailable options:")
    print("\n1. Train model")
    print("2. Test trained model")
    print("3. Show configuration")
    print("4. Change model/dataset")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ")
    
    if choice == "1":
        model_dir = train_model(config)
        
        # Ask if user wants to test
        test_choice = input("\nTest the model now? (y/n): ")
        if test_choice.lower() == 'y':
            test_model(model_dir, config)
            
    elif choice == "2":
        model_dir = input("\nEnter model directory (press Enter for default): ").strip()
        if not model_dir:
            model_dir = config.OUTPUT_DIR
        test_model(model_dir, config)
        
    elif choice == "3":
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        print(f"\nModel: {config.SELECTED_MODEL}")
        print(f"Model path: {config.BASE_MODELS[config.SELECTED_MODEL]}")
        print(f"\nDataset: {config.SELECTED_DATASET}")
        print(f"Dataset path: {config.DATASETS[config.SELECTED_DATASET]}")
        print(f"\nTraining parameters:")
        print(f"  - Epochs: {config.NUM_EPOCHS}")
        print(f"  - Batch size: {config.BATCH_SIZE}")
        print(f"  - Learning rate: {config.LEARNING_RATE}")
        print(f"  - Max length: {config.MAX_SEQ_LENGTH}")
        print(f"\nLoRA parameters:")
        print(f"  - Rank (r): {config.LORA_R}")
        print(f"  - Alpha: {config.LORA_ALPHA}")
        print(f"  - Dropout: {config.LORA_DROPOUT}")
        print(f"\nOutput: {config.OUTPUT_DIR}")
        print(f"Device: {'CPU' if config.USE_CPU else 'GPU'}")
        
        if config.TRAIN_SUBSET_SIZE:
            print(f"\nUsing subset: {config.TRAIN_SUBSET_SIZE} training examples")
    
    elif choice == "4":
        # Restart to select again
        print("\n" + "="*60)
        main()
        
    elif choice == "5":
        print("\nExiting...")
        return
    
    else:
        print("\nInvalid choice!")


if __name__ == "__main__":
    main()
