# Google Colab Training Notebook
# Copy this entire notebook to Google Colab to train your model with free GPU

# Cell 1: Upload Dataset
from google.colab import files
print("Upload your python-codes-25k.json file:")
uploaded = files.upload()

# Cell 2: Install Dependencies
!pip install -q transformers datasets accelerate peft trl
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Cell 3: Create prepare_data.py
%%writefile prepare_data.py
"""
Data Preparation Script for Python Code Generation Model
"""

import json
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict
import re


def extract_code_from_output(output: str) -> str:
    """Extract Python code from markdown code blocks"""
    code_match = re.search(r'```python\n(.*?)\n```', output, re.DOTALL)
    if code_match:
        return code_match.group(1)
    return output


def format_instruction(instruction: str, output: str) -> Dict[str, str]:
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


def load_and_prepare_data(json_path: str, train_split: float = 0.9) -> DatasetDict:
    """Load JSON data and prepare train/validation splits"""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    formatted_data = []
    for item in data:
        if 'instruction' in item and 'output' in item:
            formatted_data.append(format_instruction(item['instruction'], item['output']))
    
    print(f"Formatted {len(formatted_data)} examples")
    
    df = pd.DataFrame(formatted_data)
    dataset = Dataset.from_pandas(df)
    split_dataset = dataset.train_test_split(test_size=1-train_split, seed=42)
    
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })
    
    print(f"Train examples: {len(dataset_dict['train'])}")
    print(f"Validation examples: {len(dataset_dict['validation'])}")
    
    return dataset_dict


# Run preparation
dataset = load_and_prepare_data('python-codes-25k.json')
dataset.save_to_disk('data/processed')
print("\nDataset prepared successfully!")

# Cell 4: Create train.py (Colab version)
%%writefile train.py
"""
Training Script for Google Colab
"""

import torch
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

MODEL_NAME = "codellama/CodeLlama-7b-hf"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "pythoncode-lora"

# Load model
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded! Training...")

# Load dataset
dataset = load_from_disk("data/processed")

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
fp16=True,
    optim="adamw_8bit",
    report_to="none",
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("\nTraining complete!")

# Cell 5: Run Data Preparation
!python prepare_data.py

# Cell 6: Start Training (This will take 2-4 hours)
!python train.py

# Cell 7: Download Trained Model
from google.colab import files
import shutil

# Zip the model
shutil.make_archive('pythoncode-lora', 'zip', 'pythoncode-lora')

# Download
files.download('pythoncode-lora.zip')
print("Download complete! Extract this on your Mac.")
