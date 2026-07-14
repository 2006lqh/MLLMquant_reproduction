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

from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
from quantize.int_matmul import QuantMatMul
from models.transformation import *
import os

def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)

def get_mas_parameters(model, use_shift=True):
    params = []
    param_ids = set()  # 用于去重，避免重复添加同一个参数对象
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            # 使用id()来判断是否是同一个参数对象，避免重复添加
            param_id = id(m)
            if param_id not in param_ids:
                m._name = n
                params.append(m)
                param_ids.add(param_id)
    return iter(params)

import re
from typing import List

def replace_projections_general(
    name: str, 
    targets: List[str], 
    replacement: str
) -> str:
    """
    在一个字符串中，将 targets 列表中的任何一个子串替换为 replacement。

    Args:
        name: 待处理的原始字符串（例如参数名）。
        targets: 一个字符串列表，包含所有需要被替换的子串（例如 ['k_proj', 'v_proj']）。
        replacement: 替换后的目标字符串（例如 'q_proj'）。

    Returns:
        替换后的字符串。如果未找到任何目标子串，则返回原始字符串。
    """
    
    # 1. 构建正则表达式模式
    #    使用 '|' 符号将所有目标子串连接起来，创建“或”匹配模式。
    #    例如：['k_proj', 'v_proj'] 转换为 'k_proj|v_proj'
    #    re.escape() 用于确保 targets 中的特殊字符被正确处理（尽管在此例中不是必需的，但更安全）
    pattern = '|'.join(re.escape(t) for t in targets)
    
    # 2. 执行替换
    #    re.sub(pattern, repl, string)
    return re.sub(pattern, replacement, name)
def mas_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            # 注意，这里需要和 reuse_scale里面保持一致，默认是使用 q_proj 来共享 kv，否则原始情况下这里会保存成 k_proj 的
            if os.getenv('inference_mode', 'merged_scales') == 'merged_scales':
                name = replace_projections_general(name, ['k_proj', 'v_proj'], 'q_proj')
                name = replace_projections_general(name, ['gate_proj'], 'up_proj')
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     


def truncate_scale(model, args):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)

def reuse_scale(model, args):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                # kv 复用 q 的 scale，gate 复用 up 的 scale ，适合于开源的 smoothquant
                if os.getenv('inference_mode', 'merged_scales') == 'merged_scales':
                    # 这种是训练过程中，kv == q, up == gate
                    if name == 'self_attn.k_proj' or name == 'self_attn.v_proj':
                        print(f'reuse scale for {name} with q_proj. ')
                        module.all_in_one_smooth_scale = model.self_attn.q_proj.all_in_one_smooth_scale
                        module.text_smooth_scale = model.self_attn.q_proj.text_smooth_scale
                        module.audio_smooth_scale = model.self_attn.q_proj.audio_smooth_scale
                        module.vision_smooth_scale = model.self_attn.q_proj.vision_smooth_scale
                    if name == 'mlp.gate_proj':
                        print(f'reuse scale for {name} with up_proj. ')
                        module.all_in_one_smooth_scale = model.mlp.up_proj.all_in_one_smooth_scale
                        module.text_smooth_scale = model.mlp.up_proj.text_smooth_scale
                        module.audio_smooth_scale = model.mlp.up_proj.audio_smooth_scale
                        module.vision_smooth_scale = model.mlp.up_proj.vision_smooth_scale
                else:
                    #这种是训练过程中，k != v != q, up != gate， 独立更新.
                    if name == 'self_attn.k_proj' or name == 'self_attn.v_proj':
                        print(f'reuse scale for {name} with q_proj , but update independently. ')
                        module.all_in_one_smooth_scale.data.copy_(model.self_attn.q_proj.all_in_one_smooth_scale.data)
                        module.text_smooth_scale.data.copy_(model.self_attn.q_proj.text_smooth_scale.data) 
                        module.audio_smooth_scale.data.copy_(model.self_attn.q_proj.audio_smooth_scale.data) 
                        module.vision_smooth_scale.data.copy_(model.self_attn.q_proj.vision_smooth_scale.data)
                    if name == 'mlp.gate_proj':
                        print(f'reuse scale for {name} with up_proj, but update independently.  ')
                        module.all_in_one_smooth_scale.data.copy_(model.mlp.up_proj.all_in_one_smooth_scale.data)
                        module.text_smooth_scale.data.copy_(model.mlp.up_proj.text_smooth_scale.data)
                        module.audio_smooth_scale.data.copy_(model.mlp.up_proj.audio_smooth_scale.data)
                        module.vision_smooth_scale.data.copy_(model.mlp.up_proj.vision_smooth_scale.data)

def smooth_and_quant_temporary(model, args, isllama):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            # smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
            #                     model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args, isllama):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            print(f'smooth for {name}')
        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        # smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
        #                     model.qkt_smooth_scale)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
