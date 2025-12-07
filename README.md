# Model Fine-Tuning Made Easy

Train your own AI code generation model in just a few steps!

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt  # or requirements-cpu.txt for CPU-only

# 2. Run the unified training script
python scripts/unified_train.py
```

## ⚙️ Configuration

Edit the `Config` class in `scripts/unified_train.py` to customize your training:

### Choose Your Model
```python
SELECTED_MODEL = "qwen-0.5b"  # Options: qwen-0.5b, qwen-1.5b, codellama-7b, deepseek-1.3b
```

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `qwen-0.5b` | 500MB | Fast | CPU training, quick tests |
| `qwen-1.5b` | 1.5GB | Medium | Better quality, still CPU-friendly |
| `codellama-7b` | 13GB | Slow | GPU training, best quality |
| `deepseek-1.3b` | 1.3GB | Medium | Code-specific tasks |

### Choose Your Dataset
```python
SELECTED_DATASET = "python-18k"  # Options: python-18k, code-alpaca, local
```

| Dataset | Examples | Description |
|---------|----------|-------------|
| `python-18k` | 18,612 | Python code with instructions |
| `code-alpaca` | 20,000 | Multi-language code examples |
| `local` | Custom | Your own `python-codes-25k.json` file |

### Training Parameters
```python
NUM_EPOCHS = 1              # Number of training passes
BATCH_SIZE = 1              # Samples per batch
LEARNING_RATE = 2e-4        # How fast the model learns
TRAIN_SUBSET_SIZE = 1000    # Use subset for quick testing (or None for full dataset)
USE_CPU = True              # Set to False for GPU training
```

## 📖 Features

The unified script provides:

1. **Train Model** - Fine-tune a model on your dataset
2. **Test Model** - Generate code samples and see results  
3. **Show Config** - View current settings
4. **Exit** - Close the program

## 💡 Usage Examples

### Train on Different Dataset
```python
# In scripts/unified_train.py, change:
SELECTED_DATASET = "code-alpaca"
```

### Use Larger Model (GPU recommended)
```python
# In scripts/unified_train.py, change:
SELECTED_MODEL = "qwen-1.5b"
USE_CPU = False
```

### Full Dataset Training
```python
# In scripts/unified_train.py, change:
TRAIN_SUBSET_SIZE = None  # Use all examples
NUM_EPOCHS = 3            # Train for longer
```

## 📁 Project Structure

```
madeAI/
├── scripts/                   # All Python scripts
│   ├── unified_train.py       # ⭐ Main training script (use this!)
│   ├── train.py               # GPU training script
│   ├── train_cpu.py           # CPU training script
│   ├── test_model.py          # Model testing (GPU)
│   ├── test_model_cpu.py      # Model testing (CPU)
│   ├── export_to_ollama.py    # Export for Ollama (GPU)
│   ├── export_to_ollama_cpu.py # Export for Ollama (CPU)
│   ├── prepare_data.py        # Data preparation
│   ├── download_dataset.py    # Dataset downloader
│   └── colab_training.py      # Google Colab training
├── docs/                      # Documentation
│   └── GPU_OPTIONS.md         # GPU training options
├── models/                    # Trained models (gitignored)
│   ├── pythoncode-lora/       # LoRA adapters
│   └── pythoncode-merged/     # Merged models
├── data/                      # Data processing
│   └── processed/             # Processed dataset cache (gitignored)
├── python-codes-25k.json      # Training dataset
├── Train_on_Colab.ipynb       # Jupyter notebook for Colab
├── requirements.txt           # Dependencies (GPU)
├── requirements-cpu.txt       # Dependencies (CPU-only)
├── quickstart.sh              # Quick setup script
└── README.md                  # This file
```

## 🎯 Training Results

Your previous training results:

- **Model**: Qwen-0.5B
- **Dataset**: Python-18k (1,000 examples)
- **Time**: 40 minutes
- **Loss**: 13.94 → 0.20 (98.6% improvement)
- **Output**: `models/pythoncode-lora/`

## 🔧 What We Built

**Completed:**
- ✅ Downloaded proper Python code dataset (18,612 examples)
- ✅ Fixed tokenization for causal language model training
- ✅ Trained Qwen2.5-0.5B model on CPU (40 min)
- ✅ Created unified training script with easy configuration
- ✅ Model generates correct Python functions

**Test Results:**
```python
# Your trained model successfully generates:
- Factorial functions
- Palindrome checkers
- String reversal
- And more!
```

## 🚀 Next Steps

1. **Try different models**: Change `SELECTED_MODEL` to test larger models
2. **Use more data**: Set `TRAIN_SUBSET_SIZE = None` for full dataset
3. **Train longer**: Increase `NUM_EPOCHS` for better results
4. **Add your data**: Replace `python-codes-25k.json` with your own examples

## 💻 Advanced: Add Your Own Dataset

Create a JSON file with this format:
```json
[
  {
    "instruction": "Write a function to sort a list",
    "output": "```python\ndef sort_list(lst):\n    return sorted(lst)\n```"
  }
]
```

Then update the config in `scripts/unified_train.py`:
```python
DATASETS = {
    "my-data": "my_dataset.json",  # Add your file
}
SELECTED_DATASET = "my-data"
```

## ⚡ Performance Tips

**CPU Training:**
- Use `qwen-0.5b` or `qwen-1.5b` models
- Keep `TRAIN_SUBSET_SIZE` ≤ 1000 for faster iteration
- Expected time: 30-60 minutes per epoch

**GPU Training (Google Colab):**
- Can use `codellama-7b` for best quality
- Set `USE_CPU = False`
- Set `TRAIN_SUBSET_SIZE = None` for full dataset
- Expected time: 30-45 minutes total

## 📚 Resources

- **Model Hub**: https://huggingface.co/models
- **Dataset Hub**: https://huggingface.co/datasets
- **LoRA Paper**: https://arxiv.org/abs/2106.09685

---

**You're all set!** Run `python scripts/unified_train.py` and select option 1 to start training. 🎉
# MadeAI
