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

# Monkey patch for MiniCPM Resampler to fix _initialize_weights issue
def patch_minicpm_resampler():
    """Patch MiniCPM's Resampler class to add missing _initialize_weights method."""
    original_import = __builtins__.__import__
    
    def custom_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if 'resampler' in name.lower():
            if hasattr(module, 'Resampler'):
                Resampler = module.Resampler
                if hasattr(Resampler, '_init_weights') and not hasattr(Resampler, '_initialize_weights'):
                    original_init = Resampler._init_weights
                    def _initialize_weights(self, module=None):
                        """Initialize weights for transformers compatibility."""
                        if module is not None:
                            original_init(self, module)
                        else:
                            self.apply(original_init)
                    Resampler._initialize_weights = _initialize_weights
        return module
    
    __builtins__.__import__ = custom_import

patch_minicpm_resampler()

import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
import argparse
import torch.nn as nn
import transformers

from datasets import load_dataset
import functools
from tqdm import tqdm
from datautils import get_loaders
try:
    from llava.model import *   # required for llava
except ImportError:
    print("If want to quantize llave models, you should manually install llava from https://github.com/haotian-liu/LLaVA")

# 仅考虑vision/audio/text 的情况，其中 vision/audio 是不包含各自的 start/end token; text 是其余 system/start/end/video 相关的.
# ----------------------------------------------------------------------
# Upstream implementation retained for comparison
# Upstream commit: 3d32ae427eec57166ea67f3018cd4568be84496f
# Purpose of local change: record calibration modality type so inactive vision
# scales can use a text-neutral fallback for text-audio calibration.
# Upstream: def get_act_scales(model, dataloader, num_samples=128, model_type="qwen"):
# Local: accepts dataset_type and preserves the three-scale checkpoint shape.
# This changes calibration data handling only.
# ----------------------------------------------------------------------
def get_act_scales(model, dataloader, num_samples=128, model_type="qwen", dataset_type="text-audio-vision"):
    if isinstance(model, transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration):
        audio_token_index = -1
        image_token_index = model.config.image_token_id
    elif model_type == "minicpm":
        # MiniCPM uses Qwen2 architecture, has different token indices
        # For MiniCPM, we need to check the actual token IDs used
        # Based on the model config, image tokens are special tokens
        # We'll use a placeholder value and detect from actual data
        audio_token_index = -1  # MiniCPM doesn't have audio
        # MiniCPM image token ID - need to get from config or use common value
        # MiniCPM typically uses <image> token, which gets tokenized
        # We'll detect image tokens dynamically or use -2 as placeholder
        image_token_index = -2  # Will be detected from data
        # model is already the llm part for MiniCPM
    else:
        audio_token_index = model.config.thinker_config.audio_token_index
        image_token_index = model.config.thinker_config.image_token_index
        video_token_id = model.config.thinker_config.video_token_id
        model = model.thinker
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}
    text_mask = None; image_mask = None; audio_mask = None

    def stat_tensor(name, tensor, text_mask=None, image_mask=None, audio_mask=None, act_scales=None):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        
        # Upstream: text_mask = text_mask.squeeze().unsqueeze(-1).expand_as(tensor)
        # Local: move masks to the hooked tensor device before expansion.
        text_mask = text_mask.to(tensor.device).squeeze().unsqueeze(-1).expand_as(tensor)
        image_mask = image_mask.to(tensor.device).squeeze().unsqueeze(-1).expand_as(tensor)
        audio_mask = audio_mask.to(tensor.device).squeeze().unsqueeze(-1).expand_as(tensor)
        
        tensor_text = tensor[text_mask].view(-1, hidden_dim).abs().detach()
        comming_max_text = torch.max(tensor_text, dim=0)[0].float().cpu()
        
        tensor_audio = tensor[audio_mask].view(-1, hidden_dim).abs().detach()
        if tensor_audio.numel() == 0:
            comming_max_audio = torch.zeros(tensor_audio.shape[1], device="cpu")
        else:
            comming_max_audio = torch.max(tensor_audio, dim=0)[0].float().cpu()
            # print(f'{name} audio mask is not none, current max: {comming_max_audio.max()} ')
        
        tensor_vision = tensor[image_mask].view(-1, hidden_dim).abs().detach()
        if tensor_vision.numel() == 0:
            comming_max_vision = torch.zeros(tensor_vision.shape[1], device="cpu")
        else:
            comming_max_vision = torch.max(tensor_vision, dim=0)[0].float().cpu()        
        
        if any(k.startswith(name + ".") for k in act_scales.keys()):
            act_scales[name+".all_in_one_scale"] = torch.max(act_scales[name+".all_in_one_scale"], comming_max)
            act_scales[name+".text_scale"] = torch.max(act_scales[name+".text_scale"], comming_max_text)
            act_scales[name+".vision_scale"] = torch.max(act_scales[name+".vision_scale"], comming_max_vision)
            act_scales[name+".audio_scale"] = torch.max(act_scales[name+".audio_scale"], comming_max_audio)
            # print(f'update {name}')
        else:
            act_scales[name+".all_in_one_scale"] = comming_max
            act_scales[name+".text_scale"] = comming_max_text
            act_scales[name+".vision_scale"] = comming_max_vision
            act_scales[name+".audio_scale"] = comming_max_audio
            # print(f'first insert {name}')

    def stat_input_hook(m, x, y, name, act_scales):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, text_mask=text_mask, image_mask=image_mask, audio_mask=audio_mask, act_scales=act_scales)

    hooks = []
    filter_modules = ['visual', 'lm_head', 'audio']
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(f in name for f in filter_modules):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name, act_scales=act_scales)))

    for i in tqdm(range(num_samples), desc="generate act scale"):
        
        if transformers.__version__ == "4.31.0":
            model(dataloader[i][0].to(device))
        else:
            image_mask = (dataloader[i]['input_ids'] == image_token_index) 
            audio_mask = (dataloader[i]['input_ids'] == audio_token_index)
            all_true = torch.full(image_mask.shape, True, dtype=torch.bool)
            text_mask = all_true & ~audio_mask & ~image_mask
            inputs = {k: v.to(device) for k, v in dataloader[i].items()}
            
            model(**inputs)             

    for h in hooks:
        h.remove()

    if dataset_type == "text-audio":
        for key in list(act_scales):
            if key.endswith(".vision_scale"):
                text_key = key[: -len(".vision_scale")] + ".text_scale"
                act_scales[key] = act_scales[text_key].clone()
        print("active calibration modalities: text,audio; inactive vision scales use text-neutral fallback")
    else:
        print(f"active calibration dataset type: {dataset_type}")
    print(f'act_scales tensor count: {len(act_scales)}')
    return act_scales

from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

# ----------------------------------------------------------------------
# Upstream implementation retained for comparison
# Upstream: def build_model_and_tokenizer(model_name):
# Local: supports local-only assets, explicit attention backend, and a memory cap.
# This is model-loading infrastructure and does not change model parameters.
# ----------------------------------------------------------------------
def build_model_and_tokenizer(model_name, attn_implementation="eager", local_files_only=False, max_memory_gib=None):
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    max_memory = None
    if max_memory_gib is not None and torch.cuda.is_available():
        max_memory = {i: f"{max_memory_gib}GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "64GiB"
    
    # 注意，我们只使用 omni 的 thinker 模块
    if 'Qwen2.5-Omni' in model_name:
        from models.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "enable_audio_output": False,
            "attn_implementation": attn_implementation,
            "local_files_only": local_files_only,
        }
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **kwargs)
    elif 'MiniCPM' in model_name:
        # For MiniCPM in activation scale generation, we only use the llm part
        # We don't need the vision encoder since we're only quantizing the llm
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": 'auto', "trust_remote_code": True}
        full_model = AutoModel.from_pretrained(model_name, **kwargs)
        model = full_model.llm
        # For MiniCPM, use only tokenizer since we'll use text-only data
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Don't store full_model to avoid recursion issues
        print(f'we will use minicpm llm model only.')
    elif 'Qwen2.5-VL' in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration    
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": 'auto', "trust_remote_code": True, 'attn_implementation': 'flash_attention_2'}
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        model.to('cuda')        
    else:
        kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, processor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/nas/yuehu/models/vl/Qwen2.5-VL-3B-Instruct', help='model name')
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',
                        help='where to save the act scales')
    parser.add_argument("--calib_dataset",type=str,default="omnibench",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--cache_dir", type=str, default='./cache')
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="text-audio-vision",
        choices=[
            "text-only",
            "vision-only",
            "audio-only",
            "text-vision",
            "text-audio",
            "vision-audio",
            "text-audio-vision",
            "mas_mix_dataset"
        ],
        help="Data type to calculate activation. Options: text-only, vision-only, audio-only, text-vision, text-audio, vision-audio, mas_mix_dataset"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_memory_gib", type=int, default=None)
    parser.add_argument("--librispeech_root", default="/data/liuqinheng/benchmark/LibriSpeech")
    parser.add_argument("--split", default="test-other")
        
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(
        args.model,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        max_memory_gib=args.max_memory_gib,
    )
    args.net = args.model.rstrip('/').split('/')[-1]
    args.model_family = args.net.split('-')[0]
    cache_dataloader = f'{args.cache_dir}/dataloader_{args.net}_{args.dataset_type}_{args.nsamples}.cache'
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader, weights_only=False)
        print(f"load calibration from {cache_dataloader}")
    else:     
        if 'Qwen' in args.model or 'MiniCPM' in args.model:
            from custom_dataset import prepare_dataset, prepare_dataset_before_quant
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                AutoProcessor,
            )
            calibration_dataset = prepare_dataset(
                n_sample=args.nsamples,
                data_type=args.dataset_type,
                librispeech_root=args.librispeech_root,
                split=args.split,
            )
            # Use the tokenizer/processor already created in build_model_and_tokenizer
            processor = tokenizer
            is_minicpm = 'MiniCPM' in args.model
            # For MiniCPM, we simplified to use text-only data
            dataloader = prepare_dataset_before_quant(processor, calibration_dataset, batch_size=args.batch_size, is_minicpm=is_minicpm)
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=args.seq_len,
            )
        torch.save(dataloader, cache_dataloader)    
    
    # Determine model type
    model_type = "qwen"  # default
    if 'MiniCPM' in args.model:
        model_type = "minicpm"
    elif 'Qwen2.5-VL' in args.model or 'Qwen2.5-Omni' in args.model:
        model_type = "qwen"
    
    act_scales = get_act_scales(
        model,
        dataloader,
        args.nsamples,
        model_type=model_type,
        dataset_type=args.dataset_type,
    )
    save_path = os.path.join(args.scales_output_path,f'{args.net}-{args.dataset_type}-{args.nsamples}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

if __name__ == '__main__':
    main()
