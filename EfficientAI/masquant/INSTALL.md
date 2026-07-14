# Installation Guide

This guide provides step-by-step instructions for installing MAS-Quant and its dependencies.

## System Requirements

- **Operating System:** Linux (tested on Ubuntu 20.04/22.04)
- **Python:** 3.8 or higher
- **CUDA:** 11.8 or higher (for GPU acceleration)
- **GPU Memory:** 
  - Minimum 12GB for 3B models
  - Recommended 24GB+ for 7B models

## Step 1: Create Conda Environment

Create a new conda environment with Python 3.10:

```bash
conda create -n masquant python=3.10 -y
conda activate masquant
```

## Step 2: Install MAS-Quant

Navigate to the MAS-Quant directory and install the package:

```bash
cd MAS-Quant-Open
pip install --upgrade pip
pip install -e .
```

This will install all the required dependencies specified in `pyproject.toml`.

## Step 3: Install Flash Attention (Required)

Flash Attention is required for efficient attention computation in multimodal models:

```bash
pip uninstall -y flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

**Note:** Flash Attention installation may take several minutes and requires a compatible CUDA version.

## Step 4: Install lmms-eval (For Multimodal Evaluation)

For multimodal benchmark evaluation, install lmms-eval:

```bash
pip install lmms-eval
```

Or install from source for the latest version:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .
```

## Step 5: Verify Installation

Test if the installation was successful:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import flash_attn; print('Flash Attention installed successfully')"
```

You should see output similar to:
```
PyTorch version: 2.x.x
Transformers version: 4.x.x
Flash Attention installed successfully
```

## Additional Dependencies

### For Qwen2.5-VL Models

Qwen2.5-VL models require `transformers>=4.44.0`:

```bash
pip install transformers>=4.44.0
```

### For Qwen2.5-Omni Models

Qwen2.5-Omni models require audio processing libraries:

```bash
pip install librosa soundfile moviepy
```

### For MiniCPM-V Models

MiniCPM-V models require:

```bash
pip install transformers==4.44.2
```

## Troubleshooting

### CUDA Out of Memory During Installation

If you encounter CUDA OOM errors during Flash Attention installation:

```bash
# Use CPU-only installation first
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

### Transformers Version Conflicts

Different models may require different transformers versions:

- **For general use:** `transformers>=4.31.0`
- **For Qwen2.5-Omni:** `transformers>=4.52.0`
- **For MiniCPM-V:** `transformers==4.44.2`

Create separate conda environments if you need to work with different models:

```bash
# For Qwen2.5-Omni
conda create --name masquant-omni --clone masquant
conda activate masquant-omni
pip install transformers==4.52.0

# For MiniCPM-V
conda create --name masquant-minicpm --clone masquant
conda activate masquant-minicpm
pip install transformers==4.44.2
```

### Flash Attention Build Errors

If Flash Attention fails to build:

1. **Check CUDA version:**
   ```bash
   nvcc --version
   ```
   Ensure CUDA 11.8+ is installed.

2. **Install build essentials:**
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential
   ```

3. **Try a different version:**
   ```bash
   pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir
   ```

### Import Errors

If you encounter import errors after installation:

```bash
# Reinstall in development mode
pip install -e . --force-reinstall --no-deps
```

## Quick Test

Run a quick test to verify everything is working:

```bash
# Test if main modules can be imported
python -c "from quantize.masquant import masquant; print('MAS-Quant imported successfully')"

# Test model loading (requires a model path)
# python main.py --model /path/to/model --epochs 0 --wbits 16 --abits 16
```

## Next Steps

After installation, proceed to the [examples](./examples/) directory to run quantization on your models:

- [Qwen2.5-VL Example](./examples/qwen2_5_vl/)
- [Qwen2.5-Omni Example](./examples/qwen2_5_omni/)

## Support

If you encounter any issues during installation, please:

1. Check the [troubleshooting section](#troubleshooting) above
2. Ensure all system requirements are met
3. Verify CUDA and GPU compatibility
4. Check that you're using compatible versions of all dependencies

For persistent issues, please open an issue on the GitHub repository with:
- Your system configuration (OS, CUDA version, GPU model)
- Python and package versions (`pip list`)
- Complete error messages
