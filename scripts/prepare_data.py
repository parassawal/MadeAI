"""
Data Preparation Script for Python Code Generation Model
Loads python-codes-25k.json and prepares it for fine-tuning
"""

import json
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict, List
import re


def extract_code_from_output(output: str) -> str:
    """Extract Python code from markdown code blocks"""
    # Match ```python ... ``` blocks
    code_match = re.search(r'```python\n(.*?)\n```', output, re.DOTALL)
    if code_match:
        return code_match.group(1)
    # If no code block, return as-is
    return output


def format_instruction(instruction: str, output: str) -> Dict[str, str]:
    """Format data in instruction-following format"""
    # Extract clean code
    code = extract_code_from_output(output)
    
    # Create instruction-following format
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
    """
    Load JSON data and prepare train/validation splits
    
    Args:
        json_path: Path to python-codes-25k.json
        train_split: Fraction of data to use for training (default 0.9)
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # Format all examples
    formatted_data = []
    for item in data:
        if 'instruction' in item and 'output' in item:
            formatted_data.append(format_instruction(item['instruction'], item['output']))
    
    print(f"Formatted {len(formatted_data)} examples")
    
    # Convert to DataFrame then Dataset
    df = pd.DataFrame(formatted_data)
    dataset = Dataset.from_pandas(df)
    
    # Split into train/validation
    split_dataset = dataset.train_test_split(test_size=1-train_split, seed=42)
    
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })
    
    print(f"Train examples: {len(dataset_dict['train'])}")
    print(f"Validation examples: {len(dataset_dict['validation'])}")
    
    return dataset_dict


def main():
    """Main execution function"""
    # Load and prepare data
    dataset = load_and_prepare_data('python-codes-25k.json')
    
    # Save to disk
    output_dir = 'data/processed'
    dataset.save_to_disk(output_dir)
    print(f"\nDataset saved to {output_dir}")
    
    # Show sample
    print("\n--- Sample Example ---")
    sample = dataset['train'][0]
    print(sample['text'])
    print("\n" + "="*50)
    
    # Show statistics
    print(f"\nDataset Statistics:")
    print(f"Total examples: {len(dataset['train']) + len(dataset['validation'])}")
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    
    avg_length = sum(len(ex['text']) for ex in dataset['train']) / len(dataset['train'])
    print(f"Average prompt length: {avg_length:.0f} characters")


if __name__ == "__main__":
    main()
