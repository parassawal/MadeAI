#!/bin/bash
# Quick Start Script for AI Model Training

echo "========================================"
echo "Python Code Generator - Quick Start"
echo "========================================"

# Check Python version
python_ver=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_ver"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No GPU detected. Training will be slow!"
fi

echo ""
echo "========================================"
echo "Step 1: Create Virtual Environment"
echo "========================================"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then install dependencies:"
echo "  pip install -r requirements.txt"
echo ""
echo "========================================"
echo "Quick Start Steps"
echo "========================================"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Prepare data:"
echo "   python scripts/prepare_data.py"
echo ""
echo "4. Train model (use unified script):"
echo "   python scripts/unified_train.py"
echo ""
echo "5. Test model:"
echo "   python scripts/test_model.py"
echo ""
echo "6. Export to Ollama:"
echo "   python scripts/export_to_ollama.py"
echo ""
echo "7. Create Ollama model:"
echo "   ollama create pythoncode -f models/pythoncode-merged/Modelfile"
echo ""
echo "8. Run your model:"
echo "   ollama run pythoncode"
echo ""
echo "========================================"
echo "For more details, see README.md"
echo "========================================"
