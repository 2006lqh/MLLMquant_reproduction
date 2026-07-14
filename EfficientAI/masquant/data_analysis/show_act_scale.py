# coding=utf-8 
# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import sys

def analyze_tensor_stats_and_save(file_path):
    """
    加载 PyTorch .pt 文件，计算其中每个 Tensor 值的 min/max/mean 统计信息，
    并将结果保存到与原始文件同目录下的 .txt 文件中。

    Args:
        file_path (str): .pt 文件的完整路径。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件未找到在路径 '{file_path}'")
        return

    # 构造输出文件的路径：在原文件路径基础上添加 .txt 后缀
    output_file_path = file_path + ".txt"
    
    # 获取原始文件的目录
    output_dir = os.path.dirname(output_file_path)
    # 如果目录不存在，创建它（尽管通常目标文件已经存在）
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"✨ 正在加载文件: {file_path} ...")
    
    try:
        # 使用 torch.load 加载文件
        data = torch.load(file_path, map_location='cpu')
        print("✅ 文件加载成功。")

    except Exception as e:
        print(f"❌ 错误：加载文件时发生异常: {e}")
        return

    # 使用 with 语句打开文件进行写入，确保文件最后被关闭
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # 同时打印到控制台和写入文件
        def log_output(message):
            print(message)
            f.write(message + '\n')
            
        log_output(f"\n--- 统计信息报告：{os.path.basename(file_path)} ---")
        log_output(f"--- 原始文件路径：{file_path} ---")
        log_output(f"--- 生成时间：{torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'}")

        # 检查加载的数据是否为字典类型
        if not isinstance(data, dict):
            log_output(f"⚠️ 警告：加载的数据类型不是字典，而是 {type(data)}。")
            if isinstance(data, torch.Tensor):
                log_output("\n--- 统计信息 (Single Tensor) ---")
                log_output(f"  - Key: 'root'")
                log_output(f"  - Shape: {data.shape}")
                log_output(f"  - Min:   {data.min().item():.6f}")
                log_output(f"  - Max:   {data.max().item():.6f}")
                log_output(f"  - Mean:  {data.mean().item():.6f}")
            else:
                log_output("❌ 错误：无法处理非字典或非 Tensor 类型的数据。")
            return
            
        
        # 遍历字典中的 key-value 对
        for key, value in data.items():
            # 确保 value 是一个 PyTorch Tensor
            if isinstance(value, torch.Tensor):
                try:
                    # 检查 Tensor 是否为空
                    if value.numel() == 0:
                        log_output(f"  - Key: '{key}' | ⚠️ Tensor 为空 (numel=0)。跳过统计。")
                        log_output("-" * 30)
                        continue
                    
                    # 计算统计信息
                    # 为了提高性能和避免多线程问题，计算通常在 CPU 上进行（如果 map_location='cpu'）
                    min_val = value.min().item()
                    max_val = value.max().item()
                    mean_val = value.mean().item()
                    
                    # 记录结果
                    log_output(f"  - Key:   '{key}'")
                    log_output(f"  - Shape: {value.shape}")
                    log_output(f"  - Min:   {min_val:.6f}")
                    log_output(f"  - Max:   {max_val:.6f}")
                    log_output(f"  - Mean:  {mean_val:.6f}")
                    log_output("-" * 30)
                    
                except Exception as e:
                    log_output(f"  - Key: '{key}' | ❌ 错误：计算统计信息时发生异常: {e}")
                    log_output("-" * 30)
                    
            else:
                # 记录非 Tensor 类型的键
                log_output(f"  - Key: '{key}' | ℹ️ 值类型为 {type(value)}，跳过统计。")
                log_output("-" * 30)

    print(f"\n🎉 统计报告已成功保存到文件: {output_file_path}")

# --- 主执行部分 ---
# 假设的文件路径，请根据你的实际情况修改
# file_to_load = "act_scales/Qwen2.5-Omni-3B-text-audio-vision-128.pt"
file_to_load = "act_scales/Qwen2.5-Omni-3B-mas_mix_dataset-128.pt"

analyze_tensor_stats_and_save(file_to_load)