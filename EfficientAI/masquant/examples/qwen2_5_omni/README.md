# Qwen2.5-Omni Quantization Example

This directory contains examples for quantizing Qwen2.5-Omni models using MAS-Quant.

## Quick Start

### Step 1: Generate Activation Scales

```bash
bash 1_generate_act_scales.sh
```

This script:
- Calculates activation distribution statistics for **all three modalities**: text, audio, and vision
- Generates scales saved to `./act_scales/` directory

**Configuration:** Edit the script to set:
- `MODEL_PATH`: Path to your Qwen2.5-Omni model
- `GPU_ID`: GPU device to use
- `DATASET_TYPE`: Use `text-audio-vision` for all modalities
- `NSAMPLES`: Number of calibration samples (default: 128)

### Step 2: Train and Evaluate

Choose one of the following configurations:

**W4A8 Quantization:**
```bash
bash 2_train_and_eval_w4a8.sh
```

**W8A8 Quantization:**
```bash
bash 2_train_and_eval_w8a8.sh
```

These scripts:
- Train **modality-aware** quantization scales for text, audio, and vision
- Use `loss_multi_modal_mae_alpha` loss function with gradient-based weighting
- Evaluate perplexity (PPL) on WikiText2
- Save quantized parameters to `./outputs_omni/`

## Supported Models

- Qwen2.5-Omni-3B
- Qwen2.5-Omni-7B

## Supported Quantization Configurations

| Configuration | Weight Bits | Activation Bits | Group Size | Symmetric |
|---------------|-------------|-----------------|------------|-----------|
| W4A16         | 4           | 16              | 128        | No        |
| W4A8          | 4           | 8               | 0          | Yes       |
| W8A8          | 8           | 8               | 0          | Yes       |

## Three-Modality Support

Qwen2.5-Omni is a **tri-modal** model supporting:
- **Text**: Language understanding and generation
- **Audio**: Speech recognition and understanding
- **Vision**: Image understanding

MAS-Quant learns **separate quantization scales** for each modality, optimizing quantization for each modality's unique characteristics.

## Evaluation

The scripts include:
- **PPL Evaluation:** Perplexity on WikiText2 dataset
- **Custom Tasks:** You can add `--eval_omni_task` for custom multimodal evaluation

## Modality-Aware Loss Function

The training uses `--loss_multi_modal_mae_alpha`, which:
- Computes separate losses for text, audio, and vision
- Applies gradient-based weighting to balance different modalities
- Recommended gradient weights:
  - Text: 1.0
  - Audio: 0.55 (Qwen2.5-Omni-3B)
  - Vision: 0.15 (Qwen2.5-Omni-3B)

## Tips

1. **Memory Requirements:** 
   - W4A8: ~14GB GPU memory for 3B model
   - W8A8: ~20GB GPU memory for 3B model

2. **Training Time:**
   - 2 epochs: ~45-90 minutes depending on hardware
   - More epochs improve quantization quality

3. **Inference Mode:**
   - Always use `export inference_mode="split_scales"` for tri-modal models
   - This enables independent scale optimization for each modality

4. **Calibration Data:**
   - Use `--dataset-type text-audio-vision` to include all modalities
   - Or use `--dataset-type mas_mix_dataset` for balanced mixed data

## Common Issues

**Q: CUDA out of memory?**
A: Omni models require more memory. Use W4A16 configuration or reduce `--nsamples`.

**Q: Audio/Vision modality performance degraded?**
A: Ensure you used the correct `--dataset-type text-audio-vision` during activation scale generation.

**Q: How to evaluate on audio benchmarks?**
A: Use `--eval_omni_task` with appropriate audio datasets (LibriSpeech, WenetSpeech).

**Q: Where are the quantized parameters?**
A: Saved in `${OUTPUT_DIR}/mas_parameters.pth`
