#!/bin/bash

# Step 1: Generate activation scales for Qwen2.5-Omni model
# This script calculates the activation distribution statistics from calibration data
# for all three modalities: text, audio, and vision.

# Configuration
MODEL_PATH="/path/to/Qwen2.5-Omni-3B"  # Change this to your model path
DATASET_TYPE="text-audio-vision"  # For Omni model, use all three modalities
NSAMPLES=128  # Number of calibration samples
GPU_ID=0  # GPU device ID

echo "========================================="
echo "Generating Activation Scales (3 Modalities)"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset Type: ${DATASET_TYPE}"
echo "Number of Samples: ${NSAMPLES}"
echo "========================================="

# Create necessary directories
mkdir -p ./cache ./act_scales

# Generate activation scales
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_act_scale_shift.py \
    --model ${MODEL_PATH} \
    --dataset-type ${DATASET_TYPE} \
    --nsamples ${NSAMPLES}

echo "========================================="
echo "Activation scales generated successfully!"
echo "Output saved to: ./act_scales/"
echo "========================================="
