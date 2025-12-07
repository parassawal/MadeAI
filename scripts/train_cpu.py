"""
CPU-Compatible Training Script for Python Code Generation Model
Simplified version without Unsloth for CPU-only training
"""

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import os


# Model configuration - Using smaller Qwen model for CPU training
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Much smaller and faster for CPU!
MAX_SEQ_LENGTH = 512  # Reduced for CPU training

# LoRA configuration
LORA_R = 8            # Reduced from 16 for faster CPU training
LORA_ALPHA = 16       # Reduced from 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training configuration
OUTPUT_DIR = "models/pythoncode-lora"
NUM_EPOCHS = 1        # Reduced from 3 for initial testing
BATCH_SIZE = 1        # Reduced for CPU
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50
LOGGING_STEPS = 10
SAVE_STEPS = 500


def load_model_and_tokenizer():
    """Load CodeLlama model with LoRA adapters"""
    print(f"Loading model: {MODEL_NAME}")
    print("WARNING: Training on CPU. This will be VERY slow!")
    print("Consider using Google Colab or a cloud GPU for faster training.\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (full precision for CPU)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Full precision for CPU
        device_map=None,  # CPU only
    )
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def tokenize_function(examples, tokenizer):
    """Tokenize the examples"""
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    
    # For causal LM, labels are the same as input_ids
    # This tells the model what tokens to predict
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def main():
    """Main training function"""
    print("="*50)
    print("Python Code Generation Model Training (CPU)")
    print("="*50)
    
    print("\n⚠️  CPU TRAINING WARNING ⚠️")
    print("This will be EXTREMELY SLOW on CPU!")
    print("\nRecommended alternatives:")
    print("1. Google Colab (free GPU): https://colab.research.google.com")
    print("2. Kaggle Notebooks (free GPU): https://www.kaggle.com/code")
    print("3. Cloud providers (AWS, GCP, Azure)")
    print("\nIf you want to continue on CPU anyway, it may take several days.")
    
    response = input("\nContinue with CPU training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled. Please use a GPU environment.")
        return
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = load_from_disk("data/processed")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please run 'python prepare_data.py' first.")
        return
    
    # Use a smaller subset for CPU training
    print("\nUsing smaller subset for CPU training...")
    dataset['train'] = dataset['train'].select(range(min(1000, len(dataset['train']))))
    dataset['validation'] = dataset['validation'].select(range(min(100, len(dataset['validation']))))
    
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
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
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("="*50)


if __name__ == "__main__":
    main()
