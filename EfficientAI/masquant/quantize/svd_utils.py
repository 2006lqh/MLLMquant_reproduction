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

import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers

import functools
from custom_dataset import prepare_dataset, prepare_dataset_before_quant
from quantize.quantizer import UniformAffineQuantizer

def trans_scales(scales, down_shape, model_type):
    # 将omniquant的三模态scale文件格式，转换为smoothquant的单独scale文件格式
    text_scales = {}
    vision_scales = {}
    audio_scales = {}
    if model_type == 'vl':
        prefix = "model.language_model."
    elif model_type == 'omni':
        prefix = "model."
    else:
        raise Exception("Unknown model type")
    for i in range(len(scales)):  # 每层单独赋值
        text_scales[f"{prefix}layers.{i}.self_attn.q_proj"] = scales[i]["self_attn.q_proj.text_smooth_scale"].to("cuda")
        text_scales[f"{prefix}layers.{i}.self_attn.k_proj"] = scales[i]["self_attn.k_proj.text_smooth_scale"].to("cuda")
        text_scales[f"{prefix}layers.{i}.self_attn.v_proj"] = scales[i]["self_attn.v_proj.text_smooth_scale"].to("cuda")
        text_scales[f"{prefix}layers.{i}.self_attn.o_proj"] = scales[i]["self_attn.o_proj.text_smooth_scale"].to("cuda")
        text_scales[f"{prefix}layers.{i}.mlp.gate_proj"] = scales[i]["mlp.gate_proj.text_smooth_scale"].to("cuda")
        text_scales[f"{prefix}layers.{i}.mlp.up_proj"] = scales[i]["mlp.up_proj.text_smooth_scale"].to("cuda")

        vision_scales[f"{prefix}layers.{i}.self_attn.q_proj"] = scales[i]["self_attn.q_proj.vision_smooth_scale"].to("cuda")
        vision_scales[f"{prefix}layers.{i}.self_attn.k_proj"] = scales[i]["self_attn.k_proj.vision_smooth_scale"].to("cuda")
        vision_scales[f"{prefix}layers.{i}.self_attn.v_proj"] = scales[i]["self_attn.v_proj.vision_smooth_scale"].to("cuda")
        vision_scales[f"{prefix}layers.{i}.self_attn.o_proj"] = scales[i]["self_attn.o_proj.vision_smooth_scale"].to("cuda")
        vision_scales[f"{prefix}layers.{i}.mlp.gate_proj"] = scales[i]["mlp.gate_proj.vision_smooth_scale"].to("cuda")
        vision_scales[f"{prefix}layers.{i}.mlp.up_proj"] = scales[i]["mlp.up_proj.vision_smooth_scale"].to("cuda")

        audio_scales[f"{prefix}layers.{i}.self_attn.q_proj"] = scales[i]["self_attn.q_proj.audio_smooth_scale"].to("cuda")
        audio_scales[f"{prefix}layers.{i}.self_attn.k_proj"] = scales[i]["self_attn.k_proj.audio_smooth_scale"].to("cuda")
        audio_scales[f"{prefix}layers.{i}.self_attn.v_proj"] = scales[i]["self_attn.v_proj.audio_smooth_scale"].to("cuda")
        audio_scales[f"{prefix}layers.{i}.self_attn.o_proj"] = scales[i]["self_attn.o_proj.audio_smooth_scale"].to("cuda")
        audio_scales[f"{prefix}layers.{i}.mlp.gate_proj"] = scales[i]["mlp.gate_proj.audio_smooth_scale"].to("cuda")
        audio_scales[f"{prefix}layers.{i}.mlp.up_proj"] = scales[i]["mlp.up_proj.audio_smooth_scale"].to("cuda")

        text_scales[f"{prefix}layers.{i}.mlp.down_proj"] = torch.ones(down_shape, dtype=torch.bfloat16).to("cuda") #down没有参与训练
        vision_scales[f"{prefix}layers.{i}.mlp.down_proj"] = torch.ones(down_shape, dtype=torch.bfloat16).to("cuda") #down没有参与训练
        audio_scales[f"{prefix}layers.{i}.mlp.down_proj"] = torch.ones(down_shape, dtype=torch.bfloat16).to("cuda") #down没有参与训练
        # text_scales[f"model.layers.{i}.mlp.down_proj"] = scales[i]["mlp.down_proj.text_smooth_scale"].to("cuda")
        # vision_scales[f"model.layers.{i}.mlp.down_proj"] = scales[i]["mlp.down_proj.vision_smooth_scale"].to("cuda")
        # audio_scales[f"model.layers.{i}.mlp.down_proj"] = scales[i]["mlp.down_proj.audio_smooth_scale"].to("cuda")

    return text_scales, vision_scales, audio_scales #三模态各自返回

def trans_scales_mbq(scales, o_shape, down_shape):
    # 将omniquant的scale文件格式，转换为smoothquant的scale文件格式
    s_scales = {}
    layer_num = int(len(scales["scale"]) / 3)
    for i in range(layer_num):  # 每层单独赋值
        s_scales[f"model.layers.{i}.self_attn.q_proj"] = scales['scale'][3*i][2].to("cuda").to(torch.bfloat16)
        s_scales[f"model.layers.{i}.self_attn.k_proj"] = scales['scale'][3*i][2].to("cuda").to(torch.bfloat16)
        s_scales[f"model.layers.{i}.self_attn.v_proj"] = scales['scale'][3*i][2].to("cuda").to(torch.bfloat16)
        # s_scales[f"model.layers.{i}.self_attn.o_proj"] = scales[i]["out_smooth_scale"].to("cuda")
        s_scales[f"model.layers.{i}.self_attn.o_proj"] = torch.ones(o_shape, dtype=torch.float16).to("cuda").to(torch.bfloat16)
        s_scales[f"model.layers.{i}.mlp.gate_proj"] = scales['scale'][3*i+1][2].to("cuda").to(torch.bfloat16)
        s_scales[f"model.layers.{i}.mlp.up_proj"] = scales['scale'][3*i+1][2].to("cuda").to(torch.bfloat16)
        s_scales[f"model.layers.{i}.mlp.down_proj"] = torch.ones(down_shape, dtype=torch.bfloat16).to("cuda") #后续需要修改，在omniquant框架中学出一个down来
        # s_scales[f"model.layers.{i}.mlp.down_proj"] = scales['scale'][3*i+2][2].to("cuda").half()
    return s_scales, s_scales, s_scales

# def trans_scales(scales):
#     return scales

def get_id_matrix(model):
    # 不加白化的选项下，相当于白化矩阵是单位阵
    white_matrix = {}
    filter_modules = ['visual', 'lm_head', 'audio', 'down']
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(f in name for f in filter_modules):
            white_matrix[name] = torch.eye(m.weight.shape[1], dtype=torch.float64)
    return white_matrix

def get_white_matrix(model, processor, multimodal_scales, args, seq_len=512, data_type='text-only', output_type="vision"):
    model.eval()
    device = next(model.parameters()).device
    covs = {}
    if "omni" not in args.model.lower(): #vl模型
        if output_type == "vision":
            multimodal_token_index = model.config.image_token_id
        else:
            multimodal_token_index = -1
    else: #omni模型
        if output_type == "vision":
            multimodal_token_index = model.config.image_token_index
        else:
            multimodal_token_index = model.config.audio_token_index

    def stat_tensor(name, tensor, multimodal_mask=None):
        # ------------------------------------------------------------------
        # Upstream implementation retained for comparison
        # Upstream: tmp_tensor = tensor.div(multimodal_scales[name])
        # Upstream: multimodal_mask = multimodal_mask.squeeze().unsqueeze(-1).expand_as(tensor).cuda()
        # Local: align scales and masks with the hooked tensor device and dtype.
        # This is device-safe whitening preparation; the SVD formulation is unchanged.
        # ------------------------------------------------------------------
        modality_scale = multimodal_scales[name].to(device=tensor.device, dtype=tensor.dtype)
        tmp_tensor = tensor.div(modality_scale) #先除以vision_token_scale,这是vision_token的真实激活
        multimodal_mask = multimodal_mask.squeeze().unsqueeze(-1).expand_as(tensor).to(tensor.device)
        multimodal_tensor = tmp_tensor * multimodal_mask
        adds = torch.matmul(multimodal_tensor.transpose(1, 2), multimodal_tensor)
        adds_sum = torch.sum(adds, dim=0)
        if name in covs:
            covs[name] = covs[name] + adds_sum
        else:
            covs[name] = adds_sum

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, multimodal_mask=multimodal_mask)

    hooks = []
    filter_modules = ['visual', 'lm_head', 'audio', 'down']
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(f in name for f in filter_modules):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))

    cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{data_type}_{args.n_cali_samples}.cache'
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader, weights_only=False)
        print(f"load calibration from {cache_dataloader}")
    else:
        if 'qwen' in args.net.lower():
            # Upstream: calibration_dataset = prepare_dataset(n_sample=args.n_cali_samples, data_type=data_type)
            # Local: pass the configured local LibriSpeech root and split.
            calibration_dataset = prepare_dataset(
                n_sample=args.n_cali_samples,
                data_type=data_type,
                librispeech_root=getattr(args, "librispeech_root", None),
                split=getattr(args, "split", "test-other"),
            )
            dataloader = prepare_dataset_before_quant(processor, calibration_dataset, batch_size=args.batch_size)
        else:
            raise ValueError
        torch.save(dataloader, cache_dataloader)

    for i in tqdm(range(args.n_cali_samples), desc="Collecting activation stats"):
        multimodal_mask = (dataloader[i]['input_ids'] == multimodal_token_index)
        inputs = {k: v.to(device) for k, v in dataloader[i].items()}
        with torch.no_grad():
            model(**inputs)
    for h in hooks:
        h.remove()

    #求白化矩阵
    white_matrix = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(f in name for f in filter_modules):
            raw_scaling_diag_matrix = covs[name].double().to("cuda")
            # try:
            #     scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            # except Exception as e:
            #     print("Warning: eigen scaling_diag_matrix is not positive!")
            #     eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
            #     raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-5) * torch.eye(raw_scaling_diag_matrix.shape[0]).to("cuda")
            #     scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            #     eigenvalues = None
            #     del eigenvalues
            u, s, vh = torch.linalg.svd(raw_scaling_diag_matrix, full_matrices=False)
            s_sqrt = s.sqrt()
            white_matrix[name] = (u * s_sqrt).T.cpu() #右乘，需要转置一下
            print("gen whtie matrix: ", name, white_matrix[name].shape)
            raw_scaling_diag_matrix = None
            del raw_scaling_diag_matrix
            torch.cuda.empty_cache()
    return white_matrix


def _svd(white_matrix, wo_eq):
    wo_eq = wo_eq.to(
        device=white_matrix.device if white_matrix is not None else wo_eq.device,
        dtype=torch.float64,
    )
    if white_matrix is None:
        return torch.linalg.svd(wo_eq, full_matrices=False)
    else:
        mat = torch.matmul(white_matrix, wo_eq) #左乘白化矩阵
        if torch.isnan(mat).any():
            print("Warning: svd input matrix contains NaN!")
        elif torch.isinf(mat).any():
            print("Warning: svd input matrix contains Inf!")
        return torch.linalg.svd(mat, full_matrices=False)

def _low_rank(u, sigma, v, T, rank, dtype):
    if rank <= 1: #此时输入的是一个rank比例
        rank = int(sigma.size(0) * rank)
    else: #此时输入的是rank的绝对值
        rank = int(rank)
    if rank is None:
        diff = sigma[0] - sigma[4]
        for i in range(1, 17):
            diff_new = sigma[i * 4] - sigma[i * 4 + 4]
            rank = i * 4
            if diff / diff_new < 1.05:
                break
            diff = diff_new
    if rank < 32:
        rank = 32

    ur = u[:, :rank]
    sigma_mat = torch.diag(sigma[:rank])
    vr = v[:rank, :]

    def clamp_tensor(tensor):
        return torch.clamp(tensor, min=-65500, max=65500)

    if T is None:
        la = clamp_tensor(ur.float())
        lb = clamp_tensor(torch.matmul(sigma_mat, vr).float())
        return la, lb, rank
    else:
        Tinv = torch.linalg.inv(T)
        la = torch.matmul(Tinv, ur)
        lb = torch.matmul(sigma_mat, vr)
        la = clamp_tensor(la).to(dtype)
        lb = clamp_tensor(lb).to(dtype)
        return la, lb, rank

def get_effectivate_rank(sigma, modality_err): #计算残差矩阵的有效秩
    eps = 1e-15
    S = sigma[sigma > eps]
    if S.numel() == 0:
        effective_rank = torch.tensor(0.0, device=sigma.device, dtype=sigma.dtype)
    else:
        # Normalize to probability distribution
        p = S / S.sum()
        # Compute Shannon entropy (add small eps for numerical stability)
        entropy = -(p * torch.log(p + 1e-12)).sum()
        # Exponentiate to get "effective rank"
        effective_rank = torch.exp(entropy)
    # 最大可能秩
    min_dim = min(modality_err.shape)
    erank_ratio = effective_rank / min_dim  # 有效秩比例（越小越低秩）
    return effective_rank, min_dim, erank_ratio

def modality_err_low_rank_decomposition(model, text_scales, multimodal_scales, white_matrix, rank, weight_quant_params, quant_cmc):
    low_rank_adapters = {}
    effective_ranks = {}
    for name, m in model.named_modules():
        filter_modules = ['visual', 'lm_head', 'audio', 'down']
        if isinstance(m, nn.Linear) and not any(f in name for f in filter_modules):
            print("low rank decomposition: ", name)
            T = white_matrix[name].to("cuda") if white_matrix is not None else None
            device, dtype = m.weight.device, m.weight.dtype
            text_scale = text_scales[name].to(device=device, dtype=dtype)
            multimodal_scale = multimodal_scales[name].to(device=device, dtype=dtype)
            stW = m.weight.data.mul(text_scale.view(1, -1))
            weight_quantizer = UniformAffineQuantizer(**weight_quant_params, shape=stW.shape)
            qstW = weight_quantizer(stW)
            smW = m.weight.data.mul(multimodal_scale.view(1, -1))
            if quant_cmc: #补偿到量化权重
                qsmW = weight_quantizer(smW)
            else: #补偿到浮点权重
                qsmW = smW
            # scale_err = (multimodal_scale - text_scale).view(1, -1)
            modality_err =  qsmW - qstW #pytorch的linear写法是 X*(weight^T)
            u, sigma, v = _svd(T, modality_err.T)
            effective_rank, min_dim, erank_ratio,  = get_effectivate_rank(sigma, modality_err)
            effective_ranks[name] = (
                effective_rank.detach().cpu().item(), min_dim, erank_ratio.detach().cpu().item())
            print(f"[{name}] Effective Rank: {erank_ratio}")

            l, r, rank_svd = _low_rank(u, sigma, v, T, rank, torch.float64) #高精度存储
            reconstruct = torch.matmul(l,r)
            e = reconstruct - modality_err.T.to(device=reconstruct.device, dtype=reconstruct.dtype)
            print("矩阵重构误差: ", e.mean().item())
            adapters = {"L": l.cpu(), "R": r.cpu(), "rank": rank_svd}
            low_rank_adapters[name] = adapters
    return low_rank_adapters

def compute_sqnr(original, quantized):
    error = original - quantized
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(error ** 2)
    sqnr = 10 * torch.log10(signal_power / noise_power)

    return sqnr.item()

def get_ranks(model, text_scales, multimodal_scales, white_matrix, rank, weight_quant_params, quant_cmc):
    low_rank_adapters = {}
    effective_ranks = {}
    for name, m in model.named_modules():
        filter_modules = ['visual', 'lm_head', 'audio', 'down']
        if isinstance(m, nn.Linear) and not any(f in name for f in filter_modules):
            print("low rank decomposition: ", name)
            T = white_matrix[name].to("cuda") if white_matrix is not None else None
            device, dtype = m.weight.device, m.weight.dtype
            text_scale = text_scales[name].to(device=device, dtype=dtype)
            multimodal_scale = multimodal_scales[name].to(device=device, dtype=dtype)
            stW = m.weight.data.mul(text_scale.view(1, -1))
            weight_quantizer = UniformAffineQuantizer(**weight_quant_params, shape=stW.shape)
            qstW = weight_quantizer(stW)
            smW = m.weight.data.mul(multimodal_scale.view(1, -1))
            if quant_cmc: #补偿到量化权重
                qsmW = weight_quantizer(smW)
            else: #补偿到浮点权重
                qsmW = smW
            # scale_err = (multimodal_scale - text_scale).view(1, -1)
            modality_err =  qsmW - qstW #pytorch的linear写法是 X*(weight^T)
            u, sigma, v = _svd(T, modality_err.T)
            effective_rank, min_dim, erank_ratio,  = get_effectivate_rank(sigma, modality_err)
            effective_ranks[name] = (
                effective_rank.detach().cpu().item(), min_dim, erank_ratio.detach().cpu().item())
            print(f"[{name}] Effective Rank: {erank_ratio}")
    return effective_ranks
