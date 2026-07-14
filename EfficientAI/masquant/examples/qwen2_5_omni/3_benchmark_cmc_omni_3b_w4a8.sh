#!/bin/bash

# 设置模型路径
model_path="path/to/model"

# 定义要使用的输入文件
input_files=(
    "./data/jsonls/omnibench_128.jsonl"
#    "./data/jsonls/libri_test_other_128.jsonl"
#    "./data/jsonls/wenetspeech_test_net_128.jsonl"
)

benchmark_log_dir=./logs_1104/
mkdir -p $benchmark_log_dir

benchmark_output_dir=./poc/benchmark_ali_1104/
mkdir -p $benchmark_output_dir


export inference_mode="split_scales"
# export inference_mode="merged_scales"

# 定义权重和激活配置
# configs=("4 16" "4 8" "4 6" "4 5" "3 16")
configs=("4 8")
#configs=("4 16" "4 8" "4 6")
modalities=3
scales_path="/path/to/scales"
quant_cmc=0
#ranks=('512')
ranks=('1')
#ranks=('0' '0.05' '0.1' '0.125' '0.15' '0.2' '0.25' '0.5' '0.75' '1' '32' '64' '128' '256')
cali_data_types=("text-audio-vision")
#cali_data_types=("vision-audio-only" "no-white" "text-audio-vision")

# 循环处理不同配置
for cali_data_type in "${cali_data_types[@]}"; do
    for rank in "${ranks[@]}"; do
        for config in "${configs[@]}"; do
            # 解析权重和激活位数
            wbits=$(echo $config | cut -d' ' -f1)
            abits=$(echo $config | cut -d' ' -f2)
            base_model_name=$(basename "$model_path")

            # 设置通用参数，并根据配置调整参数
            common_args="--model ${model_path} --wbits ${wbits} --abits ${abits} --net qwen2.5-omni-3b "
#            common_args+="--eval_omni_task "
#            common_args+="--eval_sqnr "
#            common_args+="--eval_ppl "
            common_args+="--mode infer --LR --quantize --n_cali_samples 128 --rank ${rank} "
            common_args+="--scales_path ${scales_path} --modalities ${modalities} --quant_cmc ${quant_cmc} --cali_data_type ${cali_data_type} "

            # 循环处理不同输入文件
            for input_file in "${input_files[@]}"; do
                # 获取输入文件的基础名称
                input_filename=$(basename "$input_file" .jsonl)
                echo ${input_filename}
                if [[ "${input_filename}" == *"omnibench"* ]]; then
                    common_args+="--eval_sqnr "
                fi

                # 根据输入文件名设置输出文件前缀
                output_file="${benchmark_output_dir}/benchmark_${input_filename}_${base_model_name}_${inference_mode}_w${wbits}a${abits}_modalities${modalities}_quantcmc${quant_cmc}_cali_data_type${cali_data_type}_rank${rank}.txt"
                sqnr_result_file="${benchmark_output_dir}/sqnr_${base_model_name}_${inference_mode}_w${wbits}a${abits}_modalities${modalities}_quantcmc${quant_cmc}_cali_data_type${cali_data_type}_rank${rank}.csv"
                echo ${common_args}
                # 运行评估任务
                CUDA_VISIBLE_DEVICES=3 python ./infer_mas.py ${common_args} \
                    --output_file ${output_file} \
                    --input_file ${input_file} \
                    --sqnr_result ${sqnr_result_file} \
                    --batch_size 1
        #            --batch_size 1 >${benchmark_log_dir}/eval_${input_filename}_${base_model_name}_${inference_mode}_w${wbits}a${abits}.log 2>&1

            done
        done
    done
done

# 
