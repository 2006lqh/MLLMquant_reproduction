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
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.omni_norm import OmniLlamaRMSNorm
from collections import OrderedDict
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, LlamaRMSNorm, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
import copy
from models.transformation import *

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, Qwen2RMSNorm
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available

from transformers.utils import (
    auto_docstring,
    check_torch_load_is_safe,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_flex_attn_available,
    logging,
)

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb
else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


class QuantQwenMLP(nn.Module):
    def __init__(
            self,
            org_module: nn.Module,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            args=None,
            layer_idx=0
    ):
        super().__init__()

        support_training = (args.mode == 'train')
        self.gate_proj = QuantLinear(org_module.gate_proj,
                                     args.weight_quant_params,
                                     args.act_quant_params,
                                     support_training=support_training, name=f'gate_proj_{layer_idx}',
                                     layer_index=layer_idx, mode=args.mode)
        self.down_proj = QuantLinear(org_module.down_proj,
                                     args.weight_quant_params,
                                     args.act_quant_params, name=f'down_proj_{layer_idx}', layer_index=layer_idx,
                                     mode=args.mode)
        self.up_proj = QuantLinear(org_module.up_proj,
                                   args.weight_quant_params,
                                   args.act_quant_params,
                                   support_training=support_training, name=f'up_proj_{layer_idx}',
                                   layer_index=layer_idx, mode=args.mode)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x, multi_modal_mask=None):
        Gate = self.gate_proj(x, multi_modal_mask)
        Up = self.up_proj(x, multi_modal_mask)

        return self.down_proj(self.act_fn(Gate) * Up)


class QuantQwenAttentionV2(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 org_module: nn.Module,
                 config: LlamaConfig,
                 args=None, layer_idx=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_idx
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        support_training = (args.mode == 'train')
        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.weight_quant_params,
            args.act_quant_params, support_training=support_training,
            name=f'k_proj_{layer_idx}', layer_index=layer_idx,
            mode=args.mode
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.weight_quant_params,
            args.act_quant_params, support_training=support_training,
            name=f'v_proj_{layer_idx}', layer_index=layer_idx,
            mode=args.mode
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.weight_quant_params,
            args.act_quant_params, support_training=support_training,
            name=f'q_proj_{layer_idx}', layer_index=layer_idx,
            mode=args.mode
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, args.weight_quant_params, args.act_quant_params, support_training=support_training,
            name=f'o_proj_{layer_idx}', layer_index=layer_idx,
            mode=args.mode
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
        )

        self.use_weight_quant = False
        self.use_act_quant = False
        # self.forward = org_module.forward
        self.rope_scaling = config.rope_scaling
        self.attention_dropout = config.attention_dropout
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.is_causal = True

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward_flash_attn(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            multi_modal_mask=None,
    ):

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states, multi_modal_mask)
        key_states = self.k_proj(hidden_states, multi_modal_mask)
        value_states = self.v_proj(hidden_states, multi_modal_mask)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            print(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
                self.config.use_sliding_window
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output, multi_modal_mask)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def forward_sdpa(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            multi_modal_mask=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2_5OmniModel is using Qwen2_5OmniSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, multi_modal_mask)
        key_states = self.k_proj(hidden_states, multi_modal_mask)
        value_states = self.v_proj(hidden_states, multi_modal_mask)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        # attn_output = self.o_proj(attn_output)
        attn_output = self.o_proj(attn_output, multi_modal_mask)

        return attn_output, None, past_key_value

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            multi_modal_mask=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # forward_flash_attn
        # forward_sdpa
        if True or self.training:
            # if False:
            # if self.training:
            return self.forward_sdpa(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                multi_modal_mask=multi_modal_mask,
            )
        else:
            return self.forward_flash_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                multi_modal_mask=multi_modal_mask,
            )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


# 可以使用新版本的: pip install transformers==4.52.0
class QuantQwenDecoderLayerV2(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 ori_layer,
                 args, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantQwenAttentionV2(
            org_module=ori_layer.self_attn,
            config=config,
            args=args, layer_idx=layer_idx
        )
        self.mlp = QuantQwenMLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
            layer_idx=layer_idx
        )
        self.input_layernorm = OmniLlamaRMSNorm(ori_layer.input_layernorm,
                                                eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = OmniLlamaRMSNorm(ori_layer.post_attention_layernorm,
                                                         eps=ori_layer.post_attention_layernorm.variance_epsilon)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            multi_modal_mask=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if torch.isnan(hidden_states).any():
            import pdb;
            pdb.set_trace()
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            multi_modal_mask=multi_modal_mask,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states, multi_modal_mask)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.input_layernorm,
                                    [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale, self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.post_attention_layernorm, [self.mlp.up_proj, self.mlp.gate_proj],
                                    self.fc1_smooth_scale, self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj, self.self_attn.o_proj,
                                   self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                 self.qkt_smooth_scale)
            self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter = True

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.input_layernorm,
                                  [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                  self.qkv_smooth_scale, self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.post_attention_layernorm, [self.mlp.up_proj, self.mlp.gate_proj],
                                  self.fc1_smooth_scale, self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj, self.self_attn.o_proj,
                                 self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                               self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter = False

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)

    def omni_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()