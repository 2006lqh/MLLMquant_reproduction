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
from quantize.int_linear import QuantLinear

def mas_quantize_model(
        model, low_rank_adapters, text_scales, vision_scales, audio_scales, args
):
    dev = 'cuda'
    if "omni" in args.model.lower():
        layers = model.model.layers
        from models.int_qwen_omni_layer import QuantQwenDecoderLayerV2
    else:
        layers = model.model.language_model.layers
        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        print(f"=== Start quantize layer {i} ===")
        if "omni" in args.model.lower():
            qlayer = QuantQwenDecoderLayerV2(model.config.text_config, layer, args, layer_idx=i)
        else:
            qlayer = QuantQwenDecoderLayerV2(model.config, layer, args, layer_idx=i) #vl模型
        qlayer = qlayer.to(dev)
        qlayer.set_quant_state(weight_quant=True, act_quant=True)
        layers[i] = qlayer
    filter_modules = ['visual', 'lm_head', 'audio']
    for name, m in model.named_modules():
        #复制低秩分支
        if isinstance(m, QuantLinear) and not any(f in name for f in filter_modules):
            # ------------------------------------------------------------------
            # Upstream implementation retained for comparison
            # Upstream: if args.rank > 0 and name in low_rank_adapters["vision"].keys():
            # Local: attach whichever vision/audio adapter dictionaries exist.
            # This supports text-audio CMC without requiring a vision adapter.
            # ------------------------------------------------------------------
            if args.rank > 0:
                if "vision" in low_rank_adapters and name in low_rank_adapters["vision"]:
                    m.Lv = low_rank_adapters["vision"][name]["L"].to(m.weight.dtype)
                    m.Rv = low_rank_adapters["vision"][name]["R"].to(m.weight.dtype)
                else:
                    m.Lv = None
                    m.Rv = None
                if "audio" in low_rank_adapters and name in low_rank_adapters["audio"]:
                    m.La = low_rank_adapters["audio"][name]["L"].to(m.weight.dtype)
                    m.Ra = low_rank_adapters["audio"][name]["R"].to(m.weight.dtype)
                else:
                    m.La = None
                    m.Ra = None
            m.text_smooth_scale = text_scales[name]
            m.vision_smooth_scale = vision_scales[name]
            m.audio_smooth_scale = audio_scales[name]
            target_dtype = torch.bfloat16
            cur_dtype = m.weight.dtype
            m.q_weight = m.weight_quantizer(
                (m.weight.to(target_dtype) * m.text_smooth_scale).to(cur_dtype)) #real quant
    return model
