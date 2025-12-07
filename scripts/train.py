"""
Training Script for Python Code Generation Model
Fine-tunes CodeLlama-7B using LoRA on Python code dataset
"""

import torch
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import os


# Model configuration
MODEL_NAME = "codellama/CodeLlama-7b-hf"
MAX_SEQ_LENGTH = 2048  # Maximum sequence length
LOAD_IN_4BIT = True    # Use 4-bit quantization to save memory

# LoRA configuration
LORA_R = 16           # LoRA rank
LORA_ALPHA = 32       # LoRA alpha (scaling factor)
LORA_DROPOUT = 0.05   # Dropout rate
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]

# Training configuration
OUTPUT_DIR = "models/pythoncode-lora"
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
LOGGING_STEPS = 10
SAVE_STEPS = 500


def load_model_and_tokenizer():
    """Load CodeLlama model with LoRA adapters"""
    print(f"Loading model: {MODEL_NAME}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, tokenizer


def create_training_arguments():
    """Create training configuration"""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Use mixed precision training
        optim="adamw_8bit",  # 8-bit optimizer
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="none",  # Disable wandb/etc
        save_total_limit=3,  # Keep only last 3 checkpoints
    )


def main():
    """Main training function"""
    print("="*50)
    print("Python Code Generation Model Training")
    print("="*50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available! Training will be very slow on CPU.")
        print("Please ensure you have a CUDA-capable GPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk("data/processed")
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Create trainer
    print("\nSetting up trainer...")
    training_args = create_training_arguments()
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,  # Don't pack multiple examples together
    )
    
    # Start training
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    print(f"Total training steps: {len(dataset['train']) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print()
    
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
