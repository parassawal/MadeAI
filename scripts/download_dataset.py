"""
Simple script to download a Python code dataset from Hugging Face
"""
from datasets import load_dataset
import json

# Download a Python code instruction dataset
print("Downloading Python code dataset...")
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

# Convert to the format expected by prepare_data.py
data = []
for item in dataset:
    data.append({
        "instruction": item["instruction"],
        "output": item["output"]
    })

# Save as JSON
with open("python-codes-25k.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Downloaded {len(data)} examples and saved to python-codes-25k.json")
print("You can now run: python prepare_data.py")
