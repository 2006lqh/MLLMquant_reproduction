#!/bin/bash

# 设置模型路径
model_path=/path/to/model

export inference_mode="split_scales"

# 定义权重和激活配置
configs=("4 8")
modalities=3
scales_path="path/to/scales"
quant_cmc=0
ranks=('0.2')
#ranks=('0.05' '0.2')
#ranks=('0.05' '0.1' '0.2')
cali_data_types=("vision-audio-only")

# 循环处理不同配置
for cali_data_type in "${cali_data_types[@]}"; do
    for rank in "${ranks[@]}"; do
        for config in "${configs[@]}"; do
            # 解析权重和激活位数
            wbits=$(echo $config | cut -d' ' -f1)
            abits=$(echo $config | cut -d' ' -f2)
            base_model_name=$(basename "$model_path")

            postfix="w${wbits}a${abits}_grad_max_value"
            output_dir="./paper_vl/train_log_outputs_${postfix}/"
            mkdir -p ${output_dir}

            # 设置通用参数，并根据配置调整参数
            common_args="--model ${model_path} --wbits ${wbits} --abits ${abits} --output_dir ${output_dir} --n_cali_samples 128 --net qwen2.5-vl-3b "
            common_args+="--mode infer --LR --quantize --n_cali_samples 128 --rank ${rank} "
            common_args+="--scales_path ${scales_path} --quant_cmc ${quant_cmc} --cali_data_type ${cali_data_type} "
        #   common_args+=" --tasks_multimodal textvqa --limit_multimodal 0.1 "
#            common_args+="--eval_ppl "
#            common_args+=" --tasks_multimodal mmmu --limit_multimodal 0.1 "
#            common_args+=" --tasks_multimodal mmmutextvqa,scienceqa,vizwiz_vqa,ocrbench --limit_multimodal 0.1 "
            common_args+=" --tasks_multimodal textvqa,scienceqa,vizwiz_vqa,ocrbench"

            echo ${common_args}
            CUDA_VISIBLE_DEVICES=2 python infer_mas_vl.py ${common_args}
            wait
        done
    done
done