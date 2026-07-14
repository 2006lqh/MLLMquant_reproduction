# Qwen2.5-VL Quantization Example

This directory contains examples for quantizing Qwen2.5-VL models using MAS-Quant.

## Quick Start

### Step 1: Generate Activation Scales

```bash
bash 1_generate_act_scales.sh
```

This script:
- Calculates activation distribution statistics from calibration data
- Supports `text-vision` modality combination
- Generates scales saved to `./act_scales/` directory

**Configuration:** Edit the script to set:
- `MODEL_PATH`: Path to your Qwen2.5-VL model
- `GPU_ID`: GPU device to use
- `NSAMPLES`: Number of calibration samples (default: 128)

### Step 2: Train and Evaluate

Choose one of the following configurations:

**W4A6 Quantization:**
```bash
bash 2_train_and_eval_w4a6.sh
```

**W4A8 Quantization:**
```bash
bash 2_train_and_eval_w4a8.sh
```

**W8A8 Quantization:**
```bash
bash 2_train_and_eval_w8a8.sh
```

These scripts:
- Train modality-aware quantization scales
- Use `loss_multi_modal_mae_alpha` loss function
- Evaluate on multimodal benchmarks (TextVQA, ScienceQA, etc.)
- Save quantized parameters to `./outputs_vl/`

## Supported Models

- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct

## Supported Quantization Configurations

| Configuration | Weight Bits | Activation Bits | Group Size | Symmetric |
|---------------|-------------|-----------------|------------|-----------|
| W4A16         | 4           | 16              | 128        | No        |
| W4A8          | 4           | 8               | 0          | Yes       |
| W4A6          | 4           | 6               | 0          | Yes       |
| W8A8          | 8           | 8               | 0          | Yes       |

## Evaluation Tasks

The scripts support evaluation on:
- TextVQA
- ScienceQA
- VizWiz VQA
- OCRBench
- SEEDBench
- MMMU

You can modify the `--tasks_multimodal` parameter in the training scripts to select tasks.

## Tips

1. **Memory Requirements:** 
   - W4A6/W4A8: ~12GB GPU memory for 3B model
   - W8A8: ~16GB GPU memory for 3B model

2. **Training Time:**
   - 2 epochs: ~30-60 minutes depending on hardware
   - Increase `--epochs` for better quantization quality

3. **Inference Mode:**
   - Use `export inference_mode="split_scales"` for modality-aware quantization (recommended)
   - Use `export inference_mode="merged_scales"` for unified quantization

## Common Issues

**Q: CUDA out of memory?**
A: Reduce `--nsamples` or use a smaller model.

**Q: Evaluation takes too long?**
A: Set `--limit_multimodal 0.1` to use only 10% of evaluation data for quick testing.

**Q: Where are the quantized parameters?**
A: Saved in `${OUTPUT_DIR}/mas_parameters.pth`
