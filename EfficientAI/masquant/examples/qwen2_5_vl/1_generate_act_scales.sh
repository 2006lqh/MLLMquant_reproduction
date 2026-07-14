#!/bin/bash

# Step 1: Generate activation scales for Qwen2.5-VL model
# This script calculates the activation distribution statistics from calibration data
# which will be used to initialize the modality-aware scales in the quantization process.

# Configuration
MODEL_PATH="/nas/yuehu/models/vl/Qwen2.5-VL-3B-Instruct"  # Change this to your model path
DATASET_TYPE="text-vision"  # For VL model, use text-vision
NSAMPLES=128  # Number of calibration samples
GPU_ID=0  # GPU device ID

echo "========================================="
echo "Generating Activation Scales"
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
