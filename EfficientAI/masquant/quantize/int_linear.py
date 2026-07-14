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
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import os
import math


class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(
            self,
            org_module: nn.Linear,
            weight_quant_params: dict = {},
            act_quant_params: dict = {},
            disable_input_quant=False,
            support_training=False,
            name="none",
            layer_index=0,
            mode="train"
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.name = name
        self.layer_index = layer_index
        self.register_buffer('weight', org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        if support_training == True:
            self.all_in_one_smooth_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.bfloat16))
            self.text_smooth_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.bfloat16))
            self.vision_smooth_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.bfloat16))
            self.audio_smooth_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.bfloat16))
        else:
            self.register_buffer('all_in_one_smooth_scale', torch.ones(self.in_features, dtype=torch.bfloat16))
            self.register_buffer('text_smooth_scale', torch.ones(self.in_features, dtype=torch.bfloat16))
            self.register_buffer('vision_smooth_scale', torch.ones(self.in_features, dtype=torch.bfloat16))
            self.register_buffer('audio_smooth_scale', torch.ones(self.in_features, dtype=torch.bfloat16))

        self.register_buffer("Lv", None)
        self.register_buffer("Rv", None)
        self.register_buffer("La", None)
        self.register_buffer("Ra", None)

        # merged_scales  : 统一scale;   split_scales : 分模态的scale
        self.inference_mode = os.getenv('inference_mode', 'merged_scales')
        # if self.inference_mode == 'split_scales':
        #     print(f'分模态的 scale')
        # else:
        #     print(f'统一 scale')
        #区分是infer模式还是train模式
        self.mode = mode

    def get_masked_value(self, mask_value, input_tensor):
        # 检查形状
        if mask_value is None or mask_value.shape != input_tensor.shape:
            # 使用 torch.zeros_like 创建一个与 input_tensor 相同形状的零张量
            print(f'do not support !!!!')
            import pdb;
            pdb.set_trace()

        # 使用 torch.where 进行掩码选择
        return torch.where(mask_value, input_tensor, torch.zeros_like(input_tensor))

    def find_different_indices(self, tensor1, tensor2):
        # 检查形状是否相同
        if tensor1.shape != tensor2.shape:
            print(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
            return None

        # 找到不相等的位置
        not_equal = ~torch.eq(tensor1, tensor2)  # 或者 tensor1 != tensor2
        indices = torch.where(not_equal)

        # 返回索引和对应的值
        return indices, tensor1[indices], tensor2[indices]

    def forward_mas_train(self, input: torch.Tensor, multi_modal_mask=None):
        cur_dtype = input.dtype
        target_dtype = torch.bfloat16
        inference_mode = self.inference_mode
        scale_dtype = self.vision_smooth_scale.dtype
        weight_dtype = self.weight.dtype

        if cur_dtype != target_dtype or cur_dtype != scale_dtype or cur_dtype != weight_dtype:
            print(f'data type is invalid')
        # if self.layer_index == 0:
        #     inference_mode = 'split_scales'

        # 只有是在 prefill 阶段，才有 mask 信息，所以在 decode 阶段，就按照是文本输出的方式来进行计算。
        if multi_modal_mask is not None and input.shape[1] != 1:
            (audio_mask, image_mask, text_mask) = multi_modal_mask
        else:
            audio_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)
            image_mask = audio_mask
            text_mask = torch.full(input.shape, True, dtype=torch.bool, device=input.device)

        if audio_mask is None:
            audio_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)

        if image_mask is None:
            image_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)

        if text_mask is None:
            text_mask = torch.full(input.shape, True, dtype=torch.bool, device=input.device)

        # 为了避免出现scale中有含0的情况，需要做一些处理
        # 1e-5 在 bfloat16 中可以很好地表示
        epsilon_value = 1e-5

        # 将 epsilon 转换为与张量相同的类型和设备
        epsilon = torch.tensor(
            epsilon_value,
            dtype=self.vision_smooth_scale.dtype,
            device=self.vision_smooth_scale.device
        )
        # with torch.no_grad():
        #     if inference_mode == 'merged_scales':
        #         self.all_in_one_smooth_scale.data.add_(epsilon)
        #     else:
        #         self.vision_smooth_scale.data.add_(epsilon)
        #         self.audio_smooth_scale.data.add_(epsilon)
        #         self.text_smooth_scale.data.add_(epsilon)

        # 统一scale
        if inference_mode == 'merged_scales':
            if self.use_weight_quant:
                weight_scaled = self.weight_quantizer(
                    (self.weight.to(target_dtype) * self.all_in_one_smooth_scale).to(cur_dtype))
            else:
                weight_scaled = self.weight
            bias = self.bias

            if self.use_act_quant and not self.disable_input_quant:
                input_scaled = self.act_quantizer((input.to(target_dtype) / self.all_in_one_smooth_scale).to(cur_dtype))
            else:
                input_scaled = input

            out = self.fwd_func(input_scaled, weight_scaled,
                                bias=self.bias.to(cur_dtype) if self.bias is not None else None)
            out = out.to(cur_dtype)
            # if not math.isfinite(self.all_in_one_smooth_scale[0].item()):
            #     import pdb; pdb.set_trace()

        # 分模态scale
        else:
            # self.audio_smooth_scale = self.text_smooth_scale
            # self.vision_smooth_scale = self.text_smooth_scale
            if self.use_weight_quant:
                weight_audio = self.weight_quantizer(
                    (self.weight.to(target_dtype) * self.audio_smooth_scale).to(cur_dtype))
                weight_vision = self.weight_quantizer(
                    (self.weight.to(target_dtype) * self.vision_smooth_scale).to(cur_dtype))
                weight_text = self.weight_quantizer(
                    (self.weight.to(target_dtype) * self.text_smooth_scale).to(cur_dtype))
            else:
                weight_audio = (self.weight.to(target_dtype) * self.audio_smooth_scale).to(cur_dtype)
                weight_vision = (self.weight.to(target_dtype) * self.vision_smooth_scale).to(cur_dtype)
                weight_text = (self.weight.to(target_dtype) * self.text_smooth_scale).to(cur_dtype)

            if self.use_act_quant and not self.disable_input_quant:
                input_audio = self.act_quantizer(
                    self.get_masked_value(audio_mask, input.to(target_dtype)) / (self.audio_smooth_scale + epsilon)).to(
                    cur_dtype)
                input_vision = self.act_quantizer(self.get_masked_value(image_mask, input.to(target_dtype)) / (
                            self.vision_smooth_scale + epsilon)).to(cur_dtype)
                input_text = self.act_quantizer(
                    self.get_masked_value(text_mask, input.to(target_dtype)) / (self.text_smooth_scale + epsilon)).to(
                    cur_dtype)
            else:
                input_audio = (self.get_masked_value(audio_mask, input.to(target_dtype)) / (
                            self.audio_smooth_scale + epsilon)).to(cur_dtype)
                input_vision = (self.get_masked_value(image_mask, input.to(target_dtype)) / (
                            self.vision_smooth_scale + epsilon)).to(cur_dtype)
                input_text = (self.get_masked_value(text_mask, input.to(target_dtype)) / (
                            self.text_smooth_scale + epsilon)).to(cur_dtype)

            out_audio = self.fwd_func(input_audio, weight_audio, bias=None)
            out_vision = self.fwd_func(input_vision, weight_vision, bias=None)
            out_text = self.fwd_func(input_text, weight_text,
                                     bias=self.bias.to(cur_dtype) if self.bias is not None else None)
            # if input_vision.mean().item() <= 0.0000001:
            # import pdb;pdb.set_trace()
            # print(f'layer_{self.layer_index}, name: {self.name}, text_smooth_scale: {self.text_smooth_scale.mean()}, audio_smooth_scale: {self.audio_smooth_scale.mean()}, vision_smooth_scale: {self.vision_smooth_scale.mean()}')
            # print(f'layer_{self.layer_index}, name: {self.name}, out_text: {out_text.mean()}, out_audio: {out_audio.mean()}, out_vision: {out_vision.mean()}')
            out = out_text + out_vision + out_audio
            if torch.isnan(out_vision).any() or torch.isnan(out_text).any() or torch.isnan(out_audio).any():
                import pdb;
                pdb.set_trace()

            out = out.to(cur_dtype)

        return out

        # return out, input_text, input_vision, input_audio, weight_text, weight_vision, weight_audio

    def forward_mas_infer(self, input: torch.Tensor, multi_modal_mask=None):
        # 三模态补偿模式
        cur_dtype = input.dtype
        target_dtype = torch.bfloat16

        # 只有是在 prefill 阶段，才有 mask 信息，所以在 decode 阶段，就按照是文本输出的方式来进行计算。
        if multi_modal_mask is not None and input.shape[1] != 1:
            (audio_mask, image_mask, text_mask) = multi_modal_mask
        else:
            audio_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)
            image_mask = audio_mask
            text_mask = torch.full(input.shape, True, dtype=torch.bool, device=input.device)

        if audio_mask is None:
            audio_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)

        if image_mask is None:
            image_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)

        if text_mask is None:
            text_mask = torch.full(input.shape, True, dtype=torch.bool, device=input.device)

        # 分模态scale, 权重已经提前量化好
        q_input_audio = self.act_quantizer(
            self.get_masked_value(audio_mask, input.to(target_dtype)) / self.audio_smooth_scale).to(cur_dtype)
        q_input_vision = self.act_quantizer(
            self.get_masked_value(image_mask, input.to(target_dtype)) / self.vision_smooth_scale).to(cur_dtype)
        q_input_text = self.act_quantizer(
            self.get_masked_value(text_mask, input.to(target_dtype)) / self.text_smooth_scale).to(cur_dtype)

        input_audio = (self.get_masked_value(audio_mask, input.to(target_dtype)) / self.audio_smooth_scale).to(
            cur_dtype)
        input_vision = (self.get_masked_value(image_mask, input.to(target_dtype)) / self.vision_smooth_scale).to(
            cur_dtype)

        out_audio = self.fwd_func(q_input_audio, self.q_weight, bias=None)
        out_vision = self.fwd_func(q_input_vision, self.q_weight, bias=None)
        out_text = self.fwd_func(q_input_text, self.q_weight,
                                 bias=self.bias.to(cur_dtype) if self.bias is not None else None)  # 这里其实可以简化，但考虑到数值一致性。
        out = out_text + out_vision + out_audio

        if self.Lv is not None:
            # vision低秩分解的另一条旁路
            xvL = torch.matmul(input_vision, self.Lv)
            xvLR = torch.matmul(xvL, self.Rv).to(cur_dtype)
        else:
            xvLR = 0

        if self.La is not None:
            # audio低秩分解的另一条旁路
            xaL = torch.matmul(input_audio, self.La)
            xaLR = torch.matmul(xaL, self.Ra).to(cur_dtype)
        else:
            xaLR = 0

        out = out + xvLR + xaLR
        out = out.to(cur_dtype)

        # weight_vision = self.q_weight.double() + torch.matmul(self.Lv, self.Rv).T
        # weight_audio = self.q_weight.double() + torch.matmul(self.La, self.Ra).T
        # return out, input_text, input_vision, input_audio, self.q_weight, weight_vision.to(cur_dtype), weight_audio.to(cur_dtype)
        return out

    def forward(self, input: torch.Tensor, multi_modal_mask=None):
        if self.mode == "infer":
            return self.forward_mas_infer(input, multi_modal_mask)
        else:
            return self.forward_mas_train(input, multi_modal_mask)
        # # a1, b1, c1, d1, e1, f1, g1 = self.forward_mas_train(input, multi_modal_mask)
        # # a2, b2, c2, d2, e2, f2, g2 = self.forward_mas_infer(input, multi_modal_mask)
        #
        # # return self.forward_mas_train(input, multi_modal_mask)
        # return self.forward_mas_infer(input, multi_modal_mask)
        # # return a2

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
