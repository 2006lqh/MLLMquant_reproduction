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
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutputWithPast
from dataclasses import dataclass


@dataclass
class MiniCPMVModelOutputWithPast(BaseModelOutputWithPast):
    """
    MiniCPM-V model output with multi-modal information.
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MiniCPMVModelWrapper(nn.Module):
    """
    Wrapper for MiniCPM-V model that adds multi-modal mask support.
    This wrapper intercepts the forward pass and generates image_mask and text_mask
    based on image token positions, similar to Qwen2.5-VL implementation.
    """
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        
        # MiniCPM uses <image> and </image> tokens to mark image positions
        # We need to identify these tokens
        self.im_start_token = getattr(config, 'im_start_token', '<image>')
        self.im_end_token = getattr(config, 'im_end_token', '</image>')
        
        # Try to get tokenizer to find image token IDs
        # Note: This will be set externally if needed
        self.image_token_id = None
        self.im_start_id = None
        self.im_end_id = None
    
    def set_image_token_ids(self, tokenizer):
        """Set image token IDs from tokenizer."""
        if hasattr(tokenizer, 'encode'):
            # Try to encode image tokens
            try:
                self.im_start_id = tokenizer.encode(self.im_start_token, add_special_tokens=False)[0]
                self.im_end_id = tokenizer.encode(self.im_end_token, add_special_tokens=False)[0]
            except:
                pass
        
        # Try to get from config
        if hasattr(self.config, 'im_start_id'):
            self.im_start_id = self.config.im_start_id
        if hasattr(self.config, 'im_end_id'):
            self.im_end_id = self.config.im_end_id
    
    def _create_multi_modal_mask(self, input_ids, inputs_embeds):
        """
        Create multi-modal mask based on input_ids.
        For MiniCPM-V, image tokens are typically marked by special tokens.
        
        Returns:
            tuple: (audio_mask, image_mask, text_mask)
                   audio_mask is always None for MiniCPM-V (vision-only model)
        """
        if input_ids is None or inputs_embeds is None:
            return None, None, None
        
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        device = inputs_embeds.device
        
        # Initialize masks
        image_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool, device=device)
        
        # MiniCPM-V: Find image tokens
        # The image embedding typically replaces the entire sequence between <image> and </image>
        # For simplicity, we detect any special image token or use heuristics
        
        # Method 1: If we have image token IDs, use them
        if self.im_start_id is not None and self.im_end_id is not None:
            for b in range(batch_size):
                in_image_region = False
                for i in range(seq_len):
                    token_id = input_ids[b, i].item()
                    if token_id == self.im_start_id:
                        in_image_region = True
                    elif token_id == self.im_end_id:
                        in_image_region = False
                    elif in_image_region:
                        image_mask[b, i, :] = True
        
        # Method 2: Heuristic - detect large embedding changes (image embeddings often have different magnitudes)
        # This is a fallback if we don't have explicit image token IDs
        else:
            # Use a simple heuristic: if embedding norm is significantly different from text embeddings
            # This is not perfect but can work in practice
            pass  # Skip heuristic for now, will rely on explicit token IDs
        
        # Create text mask (all non-image tokens)
        all_true = torch.full(image_mask.shape, True, dtype=torch.bool, device=device)
        text_mask = all_true & ~image_mask
        
        # Audio mask is None for vision-only models
        audio_mask = None
        
        return audio_mask, image_mask, text_mask
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, MiniCPMVModelOutputWithPast]:
        """
        Forward pass with multi-modal mask generation and传递.
        """
        # Get inputs_embeds if not provided
        if inputs_embeds is None and input_ids is not None:
            # This will be handled by the underlying model
            pass
        
        # For prefill stage (not using cache), generate multi-modal mask
        multi_modal_mask = None
        if input_ids is not None and (past_key_values is None or len(past_key_values) == 0):
            # We need to get inputs_embeds to create the mask
            # Call the model's embedding layer
            if hasattr(self.model, 'get_input_embeddings'):
                temp_inputs_embeds = self.model.get_input_embeddings()(input_ids)
                audio_mask, image_mask, text_mask = self._create_multi_modal_mask(input_ids, temp_inputs_embeds)
                if image_mask is not None:
                    multi_modal_mask = (audio_mask, image_mask, text_mask)
        
        # Call the underlying model with multi_modal_mask
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            multi_modal_mask=multi_modal_mask,
            **kwargs
        )
        
        return MiniCPMVModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0],
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        ) if return_dict else outputs


def wrap_minicpm_model_with_mask(model, tokenizer=None):
    """
    Wrap a MiniCPM-V model to add multi-modal mask support.
    
    Args:
        model: The MiniCPM-V model to wrap
        tokenizer: Optional tokenizer to extract image token IDs
    
    Returns:
        Wrapped model with multi-modal mask support
    """
    config = model.config
    wrapper = MiniCPMVModelWrapper(model, config)
    
    if tokenizer is not None:
        wrapper.set_image_token_ids(tokenizer)
    
    return wrapper
