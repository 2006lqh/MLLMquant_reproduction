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

import sys
from pathlib import Path
import torch
import random
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
# from models.int_qwen_omni_layer import omni_quantize_model
from quantize.svd_utils import get_id_matrix, get_white_matrix, modality_err_low_rank_decomposition, trans_scales, compute_sqnr
from models.LMClass import LMClass
import os
# ----------------------------------------------------------------------
# Upstream implementation retained for comparison
# Upstream: from main import evaluate
# Local: import the shared ASR prompt so inference and JSONL generation use one protocol.
# This changes evaluation entry-point configuration only.
# ----------------------------------------------------------------------
from main import ASR_PROMPT, evaluate
import utils
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)
from quantize.infer_quant import mas_quantize_model


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str) #sqnr推理数据集
parser.add_argument("--output_file", type=str) #sqnr推理结果保存
parser.add_argument("--mode", type=str, default="train", choices=['train', 'infer'])
parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
parser.add_argument("--output_dir_postfix", type=str, default="", help="post fix for output dir")
parser.add_argument("--batch_size", default=1, type=int)

parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--scales_path",
    type=str,
    default="act_scales/llama-2-7b.pt",
)
parser.add_argument(
    "--attn_implementation",
    type=str, required=False, default="eager",
    choices=["eager", "sdpa", "flash_attention_2"],
    help="attention implementation that the model works with",
)
net_choices = [
    "qwen2.5-omni-3b",
    "qwen2.5-omni-7b",
    "qwen2.5-vl-3b",
    "qwen2.5-vl-7b",
]

parser.add_argument("--net", type=str, default="qwen2.5-omni-3b", choices=net_choices)
parser.add_argument("--limit", type=int, default=-1)
parser.add_argument("--n_cali_samples", type=int, default=8)  # 白化校准集样本数
parser.add_argument("--cali_data_type", type=str, default="text-audio-vision")  # 白化校准集类型
parser.add_argument("--n_samples", type=int, default=None)  # 评测ppl测试集样本数
parser.add_argument("--rank", type=float, default=None)  # 低秩分解的rank
parser.add_argument("--quant_cmc", type=int, default=0)  # 白化矩阵存储
parser.add_argument("--save_white_matrix_path", type=str, default="")  # 白化矩阵存储
parser.add_argument("--save_low_rank_adapters", type=str, default="")  # 低秩适配器存储
parser.add_argument("--abits", type=int, default=8)
parser.add_argument("--wbits", type=int, default=4)
parser.add_argument("--LR", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--eval_ppl", action="store_true")
parser.add_argument("--eval_sqnr", action="store_true")
parser.add_argument("--eval_omni_task", action="store_true")
# ----------------------------------------------------------------------
# Upstream implementation retained for comparison
# Upstream: no LibriSpeech/ASR-specific inference arguments were defined here.
# Local: expose local data, generation, and validation controls required for ASR evaluation.
# This changes benchmark plumbing; MAS/CMC numerical kernels are unchanged.
# ----------------------------------------------------------------------
parser.add_argument("--compute_wer", action="store_true")
parser.add_argument("--librispeech_root", default="/data/liuqinheng/benchmark/LibriSpeech")
parser.add_argument("--split", default="test-other")
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--asr_prompt", default=ASR_PROMPT)
parser.add_argument("--local_files_only", action="store_true")
parser.add_argument("--max_memory_gib", type=int, default=None)
parser.add_argument("--modalities", type=int, default=3)
parser.add_argument(
    "--dataset-type",
    dest="dataset_type",
    default="text-audio-vision",
    choices=["text-audio", "text-audio-vision"],
)
parser.add_argument("--limit_multimodal", type=float, default=0.1)
parser.add_argument("--num_fewshot", type=int, default=0)
parser.add_argument("--tasks", default="")
parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
parser.add_argument("--tasks_multimodal", default="")
#eval参数
parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
parser.add_argument("--sqnr_result", default="outputs/sqnr_result.csv")
parser.add_argument("--ppl_result", default="outputs/ppl_result.csv")

args = parser.parse_args()

scales_path = args.scales_path
scales_path_prefix = os.path.basename(os.path.dirname(scales_path))
n_samples = args.n_samples
n_cali_samples = args.n_cali_samples
# cali_data_type = args.cali_data_type
rank = args.rank
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

Path("./white_matrix/").mkdir(parents=True, exist_ok=True)
if args.save_white_matrix_path == "":
    save_white_matrix_path = f"./white_matrix/white_matrix_{scales_path_prefix}_{args.cali_data_type}.pt"
else:
    save_white_matrix_path = args.save_white_matrix_path

Path("./low_rank_adapters/").mkdir(parents=True, exist_ok=True)
if args.save_low_rank_adapters == "":
    save_low_rank_adapters = f"./low_rank_adapters/low_rank_adapters_{scales_path_prefix}_{args.cali_data_type}_quantcmc{int(args.quant_cmc)}_rank{args.rank}.pt"
else:
    save_low_rank_adapters = args.save_low_rank_adapters
print(save_low_rank_adapters)

args.weight_quant_params = {'n_bits': args.wbits, 'per_channel_axes': [0], 'symmetric': args.abits < 16,
                            'dynamic_method': 'per_channel', 'group_size': 128 if args.abits == 16 else 0,
                            'lwc': False, 'disable_zero_point': False}

args.act_quant_params = {'n_bits': args.abits, 'per_channel_axes': [], 'symmetric': True,
                         'dynamic_method': 'per_token'}

args.q_quant_params = {
    "n_bits": args.abits,
    "per_channel_axes": [],
    "symmetric": False,
    "dynamic_method": 'per_token',
}
args.k_quant_params = {
    "n_bits": args.abits,
    "per_channel_axes": [],
    "symmetric": False,
    "dynamic_method": 'per_token',
}
args.v_quant_params = {
    "n_bits": args.abits,
    "per_channel_axes": [],
    "symmetric": False,
    "dynamic_method": 'per_token',
}
args.p_quant_params = {
    "n_bits": 16,
    "metric": "fix0to1",
}
args.model_family = args.net.split('-')[0]

#加载模型
llm = LMClass(args)
llm.seqlen = 2048
llm.model.eval()
for param in llm.model.parameters():
    param.requires_grad = False
# ----------------------------------------------------------------------
# Upstream implementation retained for comparison
# Upstream: llm.model.to("cuda")
# Upstream: processor = AutoProcessor.from_pretrained(args.model)
# Upstream: inference_mode = "split_scales"
# Local: keep device placement with the wrapper and allow local-only processor loading.
# ----------------------------------------------------------------------
processor = AutoProcessor.from_pretrained(
    args.model,
    local_files_only=args.local_files_only,
    trust_remote_code=True,
)
inference_mode = os.getenv("inference_mode", "split_scales")

# 加载日志
# init logger
if args.output_dir:
    from datetime import datetime

    current_time = datetime.now()
    formatted_with_ms = current_time.strftime("%m%d-%H%M%S.%f")

    # args.output_dir = f'{args.output_dir}/{args.net}-{args.dataset_type}-{args.epochs}epochs-w{args.wbits}a{args.abits}-{args.output_dir_postfix}-{formatted_with_ms}'
    args.output_dir = f'{args.output_dir}/{args.net}-w{args.wbits}a{args.abits}-{args.output_dir_postfix}-{formatted_with_ms}-{inference_mode}'
    print(f'output_dir is: {args.output_dir}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
logger = utils.create_logger(output_dir)
logger.info(args)

print(f'act_scales_path:  {scales_path}, a_bit: {args.abits}, w_bit: {args.wbits}')
if "omni" in args.model.lower():
    model_type = "omni"
elif "vl" in args.model.lower():
    model_type = "vl"
else:
    raise Exception("Unknown model type")
scales = torch.load(scales_path)
print(f"MAS parameters loaded: True ({scales_path})")
if model_type == "omni":
    down_shape = llm.model.model.layers[0].mlp.down_proj.weight.shape[1]
else:
    down_shape = llm.model.model.language_model.layers[0].mlp.down_proj.weight.shape[1]
#转换scale格式
text_scales, vision_scales, audio_scales = trans_scales(scales, down_shape, model_type)

#计算白化矩阵和低秩分支
white_matrix = None
low_rank_adapters = None
if args.LR:
    if os.path.isfile(save_low_rank_adapters): #如果已经做好了低秩分解，直接读取
        low_rank_adapters = torch.load(save_low_rank_adapters)
    elif rank > 0:
        # 先计算白化矩阵
        if os.path.isfile(save_white_matrix_path):
            white_matrix = torch.load(save_white_matrix_path)
        else:
            if args.cali_data_type == "text-audio-vision": #使用omnibench128校准:
                vision_white_matrix = get_white_matrix(llm.model, processor, vision_scales, args,  2048,
                                                       "text-audio-vision", "vision")
                if model_type == "omni":
                    audio_white_matrix = get_white_matrix(llm.model, processor, audio_scales, args, 2048,
                                                       "text-audio-vision", "audio")
            # ------------------------------------------------------------------
            # Upstream implementation retained for comparison
            # Upstream supports text-audio-vision, no-white, and vision-audio-only.
            # Local adds an audio-only CMC whitening branch for text-audio calibration.
            # The low-rank algorithm is unchanged; only the active modality set differs.
            # ------------------------------------------------------------------
            elif args.cali_data_type == "text-audio":
                if model_type != "omni":
                    raise ValueError("text-audio CMC calibration requires an Omni model")
                audio_white_matrix = get_white_matrix(
                    llm.model,
                    processor,
                    audio_scales,
                    args,
                    2048,
                    "text-audio",
                    "audio",
                )
            elif args.cali_data_type == "no-white": #不做白化
                vision_white_matrix = get_id_matrix(llm.model)
                if model_type == "omni":
                    audio_white_matrix = get_id_matrix(llm.model)
            elif args.cali_data_type == "vision-audio-only": #使用coco + libri-test 数据集校准
                vision_white_matrix = get_white_matrix(llm.model, processor, vision_scales, args, 2048,
                                                       "vision-only", "vision")
                if model_type == "omni":
                    audio_white_matrix = get_white_matrix(llm.model, processor, audio_scales, args, 2048,
                                                      "audio-only", "audio")
            else:
                raise ValueError('Unknown cali_data_type')
            white_matrix = dict()
            if args.cali_data_type != "text-audio":
                white_matrix["vision"] = vision_white_matrix
            if model_type == "omni":
                white_matrix["audio"] = audio_white_matrix
            if save_white_matrix_path:
                torch.save(white_matrix, save_white_matrix_path)
        # 再计算低秩分解后的L1L2
        vision_low_rank_adapters = None
        if "vision" in white_matrix:
            vision_low_rank_adapters = modality_err_low_rank_decomposition(
                llm.model,
                text_scales,
                vision_scales,
                white_matrix["vision"],
                rank,
                args.weight_quant_params,
                args.quant_cmc,
            )
        if model_type == "omni":
            audio_low_rank_adapters = modality_err_low_rank_decomposition(llm.model, text_scales, audio_scales, white_matrix["audio"],
                                                                rank, args.weight_quant_params, args.quant_cmc)
        low_rank_adapters = dict()
        if vision_low_rank_adapters is not None:
            low_rank_adapters["vision"] = vision_low_rank_adapters
        if model_type == "omni":
            low_rank_adapters["audio"] = audio_low_rank_adapters
        if save_low_rank_adapters:
            torch.save(low_rank_adapters, save_low_rank_adapters)

    # smooth_lm(model, text_scales)  # 按照文本模态的scale先平滑权重
    else:
        low_rank_adapters = {}
else:
    low_rank_adapters = {}

if args.quantize:
    model = mas_quantize_model(
        llm.model,
        low_rank_adapters=low_rank_adapters,
        text_scales=text_scales,
        vision_scales=vision_scales,
        audio_scales=audio_scales,
        # 注意，如果要统计不同模态对最终激活量化的影响，这里就不能设置为 True，否则会影响统计结果.
        args=args
    )
    # ------------------------------------------------------------------
    # Local verification metadata
    # Purpose: expose the attached quantized thinker and CMC adapter counts in logs.
    # Upstream had no equivalent runtime assertions. This does not alter inference.
    # ------------------------------------------------------------------
    llm.model = model
    from quantize.int_linear import QuantLinear
    quantized_layer_count = sum(
        layer.__class__.__name__ == "QuantQwenDecoderLayerV2"
        for layer in llm.model.model.layers
    )
    quant_linear_count = sum(isinstance(module, QuantLinear) for module in llm.model.modules())
    cmc_module_count = sum(
        isinstance(module, QuantLinear)
        and any(getattr(module, name, None) is not None for name in ("Lv", "Rv", "La", "Ra"))
        for module in llm.model.modules()
    )
    cmc_enabled = bool(args.LR and rank is not None and rank > 0 and cmc_module_count > 0)
    print(
        f"wbits={args.wbits}, abits={args.abits}, dataset_type={args.dataset_type}, "
        f"modalities={args.modalities}, inference_mode={inference_mode}"
    )
    print(f"quantized thinker layers: {quantized_layer_count}")
    print(f"QuantLinear modules: {quant_linear_count}")
    print(f"CMC modules with low-rank adapters: {cmc_module_count}")
    print(f"CMC enabled: {cmc_enabled}, rank={rank}, quant_cmc={args.quant_cmc}")
    print("Skipped module filters: visual, audio, lm_head")
    if args.compute_wer and quantized_layer_count == 0:
        raise RuntimeError("Refusing WER evaluation: no quantized thinker layers were found")
    if args.compute_wer and args.LR and rank is not None and rank > 0 and not cmc_enabled:
        raise RuntimeError("Refusing MAS+CMC WER evaluation: low-rank CMC adapters are not attached")
    del white_matrix
    white_matrix = None
    del low_rank_adapters
    low_rank_adapters = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()
evaluate(llm, args, logger)
