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
from collections import OrderedDict

# file_path = "grad_info/qwen2.5-omni-3b-grad.pth"
file_path = "grad_info/qwen2.5-vl-3b-grad-128-sample.pth"
# file_path = "grad_info/qwen2.5-omni-7b-grad-128-sample.pth"


try:
    state_dict = torch.load(file_path)
    vis_cap_ratio_total = []
    aud_cap_ratio_total = []
    
    # 遍历 state_dict 中的每个键值对
    for key, value in state_dict.items():
        # 提取所需的梯度值
        vis_grad = value.get('vis_avg_grad')
        cap_grad = value.get('cap_avg_grad')
        aud_grad = value.get('aud_avg_grad', None)

        # 检查 cap_grad 是否接近于零，以避免除以零的错误
        if cap_grad is not None and cap_grad != 0:
            # 计算比例
            vis_cap_ratio = vis_grad / cap_grad if vis_grad is not None else float('nan')
            aud_cap_ratio = aud_grad / cap_grad if aud_grad is not None else float('nan')
            vis_cap_ratio_total.append(vis_cap_ratio)
            aud_cap_ratio_total.append(aud_cap_ratio)
            

            # 打印结果
            print(f"Key: {key}")
            print(f"  vis_avg_grad / cap_avg_grad: {vis_cap_ratio:.6f}")
            print(f"  aud_avg_grad / cap_avg_grad: {aud_cap_ratio:.6f}")
        else:
            print(f"Key: {key} - 错误：cap_avg_grad 为 None 或零，跳过比例计算。")
    print(f'vis_cap_ratio_total mean: {sum(vis_cap_ratio_total)/len(vis_cap_ratio_total)}, max: {max(vis_cap_ratio_total)}, min: {min(vis_cap_ratio_total)}')
    print(f'aud_cap_ratio_total mean: {sum(aud_cap_ratio_total)/len(aud_cap_ratio_total)}, max: {max(aud_cap_ratio_total)}, min: {min(aud_cap_ratio_total)}')
except Exception as e:
    print(f"加载或处理文件时发生错误：{e}")