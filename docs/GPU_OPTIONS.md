# GPU Training Options for macOS Users

Since you're on macOS without CUDA support, you have several options for training the model:

## Option 1: Use Google Colab (Recommended - FREE)

Google Colab provides free GPU access. Here's how to use it:

### Steps:

1. **Go to Google Colab**: https://colab.research.google.com

2. **Upload your dataset**:
   ```python
   # In a Colab cell, upload the dataset
   from google.colab import files
   uploaded = files.upload()  # Select python-codes-25k.json
   ```

3. **Install dependencies**:
   ```python
   !pip install -q transformers datasets accelerate peft trl
   !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

4. **Copy your training scripts**: Upload `prepare_data.py` and `train.py` to Colab

5. **Enable GPU**: Runtime → Change runtime type → GPU → Save

6. **Run training**:
   ```python
   !python prepare_data.py
   !python train.py
   ```

7. **Download trained model**:
   ```python
   from google.colab import files
   !zip -r model.zip models/pythoncode-lora
   files.download('model.zip')
   ```

## Option 2: Use Kaggle Notebooks (FREE)

Kaggle also provides free GPU access:

1. Go to https://www.kaggle.com/code
2. Create a new notebook
3. Enable GPU in settings
4. Upload dataset and scripts
5. Run training

## Option 3: CPU Training (VERY SLOW - NOT RECOMMENDED)

If you still want to train on your Mac (will take days):

```bash
# Install CPU-compatible dependencies
pip install -r requirements-cpu.txt

# Use CPU training script
python train_cpu.py
```

**Warning**: This will be 50-100x slower than GPU training!

## Option 4: Cloud GPU Services (PAID)

For faster training with more control:

- **RunPod**: ~$0.20/hour for RTX 3090
- **Vast.ai**: ~$0.15/hour for GPU instances
- **AWS/GCP/Azure**: More expensive but reliable

## Recommendation

**Use Google Colab** - It's free, fast, and perfect for this project. Training will take 2-4 hours instead of days on CPU.

## Local Testing

You can still test the project locally on your Mac:
- Run data preprocessing: `python prepare_data.py` ✅
- View dataset statistics and samples ✅
- Test a pre-trained model ✅

But for training, use a GPU environment!
