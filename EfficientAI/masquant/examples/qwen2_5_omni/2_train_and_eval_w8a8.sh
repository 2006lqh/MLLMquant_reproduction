#!/bin/bash

# Step 2: Train modality-aware scales and evaluate (W8A8 configuration)
# This script trains the modality-aware quantization scales for Qwen2.5-Omni (3 modalities)
# and evaluates on relevant benchmarks.

# Configuration
MODEL_PATH="/path/to/Qwen2.5-Omni-3B"  # Change this to your model path
WBITS=8  # Weight quantization bits
ABITS=8  # Activation quantization bits
EPOCHS=2  # Training epochs
GPU_ID=0  # GPU device ID
OUTPUT_DIR="./outputs_omni"

# Set inference mode to use split scales (modality-aware)
export inference_mode="split_scales"

echo "========================================="
echo "Training MAS-Quant W${WBITS}A${ABITS} (3 Modalities)"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Quantization: W${WBITS}A${ABITS}"
echo "Epochs: ${EPOCHS}"
echo "Inference Mode: ${inference_mode}"
echo "========================================="

# Create necessary directories
mkdir -p ./cache ${OUTPUT_DIR}

# Training and evaluation
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --model ${MODEL_PATH} \
    --mode train \
    --epochs ${EPOCHS} \
    --wbits ${WBITS} \
    --abits ${ABITS} \
    --let \
    --loss_multi_modal_mae_alpha \
    --dataset-type text-audio-vision \
    --nsamples 128 \
    --output_dir ${OUTPUT_DIR} \
    --symmetric \
    --group_size 0 \
    --eval_ppl

echo "========================================="
echo "Training and evaluation completed!"
echo "Quantized parameters saved to: ${OUTPUT_DIR}"
echo "========================================="
