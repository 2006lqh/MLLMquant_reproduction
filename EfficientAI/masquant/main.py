# import torch
# torch.backends.cudnn.deterministic = True
# torch.backends.cuda.matmul.allow_tf32 = False # 禁用 TF32
# torch.use_deterministic_algorithms(True)
# import torch
# torch.autograd.set_detect_anomaly(True)

# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

# patch_minicpm_resampler()

import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lmms_eval import evaluator as eval_multimodal
# from lm_eval import evaluator
# from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.masquant import masquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_llama_layer_v2 import QuantLlamaDecoderLayerV2
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
import json
import csv
import pdb
import re
import sacrebleu.tokenizers as sacrebleu_tokenizers
if not hasattr(sacrebleu_tokenizers, "TOKENIZERS"):
    from sacrebleu.tokenizers.tokenizer_none import NoneTokenizer

    sacrebleu_tokenizers.TOKENIZERS = {"none": NoneTokenizer}
from eval_utils.qwen_asr.evaluate_tokenizer import EvaluationTokenizer
from eval_utils.qwen_asr.whisper_normalizer.basic import BasicTextNormalizer
from eval_utils.qwen_asr.whisper_normalizer.english import EnglishTextNormalizer

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]

# ----------------------------------------------------------------------
# Upstream implementation retained for comparison
# Upstream commit: 3d32ae427eec57166ea67f3018cd4568be84496f
# Upstream did not include LibriSpeech JSONL construction, ASR generation, or
# global WER scoring utilities in this entry point.
# Local implementation adds those evaluation utilities for reproducible
# Qwen2.5-Omni audio-text assessment. MAS/CMC quantization math is unchanged.
# ----------------------------------------------------------------------
ASR_PROMPT = "Please transcribe the speech in the audio. Do not add any explanation."
ASR_PROMPT_P0 = "Transcribe the speech into English. Output only the transcription text."
ASR_SYSTEM_PROMPT_P1 = "You are a speech recognition model."
ASR_PROMPT_P1 = "Transcribe the English audio into text without any punctuation marks."
ASR_PREFIXES = (
    "the transcription is:",
    "the speech says:",
    "transcription:",
    "it says:",
)

_QWEN_ENGLISH_NORMALIZER = EnglishTextNormalizer()
_QWEN_BASIC_NORMALIZER = BasicTextNormalizer()
_QWEN_EVALUATION_TOKENIZER = EvaluationTokenizer(
    tokenizer_type="none",
    lowercase=True,
    punctuation_removal=True,
    character_tokenization=False,
)


def normalize_asr_text(text):
    """Legacy ``librispeech_basic`` normalization retained for historical scores."""
    text = str(text or "").strip().lower()
    for prefix in ASR_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def normalize_qwen_asr_en_text(text):
    """Apply the pinned Qwen2-Audio English ASR scoring pipeline symmetrically."""
    text = str(text or "")
    text = re.sub(r"<\|.*?\|>", " ", text)
    text = _QWEN_ENGLISH_NORMALIZER(text)
    text = _QWEN_BASIC_NORMALIZER(text)
    text = _QWEN_EVALUATION_TOKENIZER.tokenize(text)
    return " ".join(text.split())


def _fallback_word_error_counts(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # Each cell is (edit_distance, substitutions, insertions, deletions).
    previous = [(j, 0, j, 0) for j in range(len(hyp_words) + 1)]
    for i, ref_word in enumerate(ref_words, start=1):
        current = [(i, 0, 0, i)]
        for j, hyp_word in enumerate(hyp_words, start=1):
            if ref_word == hyp_word:
                current.append(previous[j - 1])
                continue
            diagonal = previous[j - 1]
            above = previous[j]
            left = current[j - 1]
            candidates = [
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (above[0] + 1, above[1], above[2], above[3] + 1),
                (left[0] + 1, left[1], left[2] + 1, left[3]),
            ]
            current.append(min(enumerate(candidates), key=lambda item: (item[1][0], item[0]))[1])
        previous = current
    _, substitutions, insertions, deletions = previous[-1]
    return substitutions, insertions, deletions


def word_error_counts(reference, hypothesis, require_jiwer=False):
    try:
        import jiwer
        from importlib.metadata import version

        result = jiwer.process_words(reference, hypothesis)
        return (
            result.substitutions,
            result.insertions,
            result.deletions,
            "jiwer.process_words",
            version("jiwer"),
        )
    except ImportError as exc:
        if require_jiwer:
            raise RuntimeError(
                "qwen_asr_en requires jiwer; install it before running a strict ASR score."
            ) from exc
        substitutions, insertions, deletions = _fallback_word_error_counts(reference, hypothesis)
        return substitutions, insertions, deletions, "levenshtein_fallback", None


def normalize_librispeech_text(text, normalization="librispeech_basic"):
    if normalization == "librispeech_basic":
        return normalize_asr_text(text)
    if normalization == "qwen_asr_en":
        return normalize_qwen_asr_en_text(text)
    else:
        raise ValueError(f"Unsupported ASR normalization: {normalization}")


def load_librispeech_samples(root, split="test-other", max_samples=None, sample_order="sorted", sample_seed=42):
    split_path = Path(root).expanduser().resolve() / split
    if not split_path.is_dir():
        raise FileNotFoundError(f"LibriSpeech split does not exist: {split_path}")

    transcripts = {}
    for transcript_path in sorted(split_path.rglob("*.trans.txt")):
        with transcript_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Malformed transcript line: {transcript_path}:{line_number}")
                utterance_id, reference = parts
                if utterance_id in transcripts and transcripts[utterance_id] != reference:
                    raise ValueError(f"Conflicting transcript for {utterance_id}")
                transcripts[utterance_id] = reference

    audio_by_id = {path.stem: path.resolve() for path in sorted(split_path.rglob("*.flac"))}
    missing_transcripts = sorted(set(audio_by_id) - set(transcripts))
    missing_audio = sorted(set(transcripts) - set(audio_by_id))
    if missing_transcripts or missing_audio:
        raise ValueError(
            f"LibriSpeech id mismatch: missing_transcripts={len(missing_transcripts)}, "
            f"missing_audio={len(missing_audio)}"
        )

    samples = [
        {
            "id": utterance_id,
            "speaker": utterance_id.split("-")[0],
            "chapter": utterance_id.split("-")[1],
            "audio": str(audio_by_id[utterance_id]),
            "audio_path": str(audio_by_id[utterance_id]),
            "reference": transcripts[utterance_id],
        }
        for utterance_id in sorted(audio_by_id)
    ]
    if sample_order == "random":
        samples = list(samples)
        random.Random(sample_seed).shuffle(samples)
    elif sample_order == "speaker_stratified":
        by_speaker = {}
        for sample in samples:
            by_speaker.setdefault(sample["speaker"], []).append(sample)
        ordered = []
        speakers = sorted(by_speaker)
        while any(by_speaker.values()):
            for speaker in speakers:
                if by_speaker[speaker]:
                    ordered.append(by_speaker[speaker].pop(0))
        samples = ordered
    elif sample_order != "sorted":
        raise ValueError(f"Unsupported LibriSpeech sample_order: {sample_order}")

    if max_samples is not None and max_samples >= 0:
        samples = samples[:max_samples]
    return samples


def build_librispeech_prompt(audio_path, asr_prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": asr_prompt},
            ],
        }
    ]


def make_librispeech_jsonl(args):
    samples = load_librispeech_samples(
        args.librispeech_root,
        args.split,
        args.max_samples,
        args.sample_order,
        args.sample_seed,
    )
    if args.split == "test-other" and args.max_samples is None and len(samples) != 2939:
        raise ValueError(f"Expected 2939 test-other samples, found {len(samples)}")

    if not args.output_file:
        raise ValueError("--output_file is required with --make_librispeech_jsonl")
    output_path = Path(args.output_file)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing JSONL: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_order_name = args.sample_order
    if args.sample_order == "sorted" and args.max_samples is None:
        sample_order_name = "sorted_full"
    subset_name = args.subset_name or (
        f"{args.split}_full" if args.max_samples is None else f"{args.split}_{args.sample_order}_{args.max_samples}_seed{args.sample_seed}"
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                **sample,
                "prompt": build_librispeech_prompt(sample["audio_path"], args.asr_prompt),
                "split": args.split,
                "sample_order": sample_order_name,
                "sample_seed": args.sample_seed,
                "subset_name": subset_name,
                "asr_prompt": args.asr_prompt,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        "LIBRISPEECH_JSONL_WRITTEN "
        + json.dumps(
            {
                "output_file": str(output_path),
                "samples": len(samples),
                "split": args.split,
                "sample_order": sample_order_name,
                "subset_name": subset_name,
                "asr_prompt": args.asr_prompt,
            },
            ensure_ascii=False,
        )
    )


def get_record_hypothesis(record):
    for key in ("hypothesis", "response", "prediction", "text"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def _first_floating_parameter_dtype(module):
    for _, parameter in module.named_parameters():
        if torch.is_floating_point(parameter):
            return str(parameter.dtype)
    return None


def _first_q_proj_dtype(module):
    for name, parameter in module.named_parameters():
        if name.endswith("q_proj.weight"):
            return str(parameter.dtype)
    return None


def _attention_implementation(config):
    if config is None:
        return None
    value = getattr(config, "_attn_implementation", None)
    if value is None:
        value = getattr(config, "attn_implementation", None)
    return value


def _parameter_devices(module):
    return sorted({str(parameter.device) for parameter in module.parameters()})


def collect_qwen_omni_runtime_metadata(llm, generation_model, args):
    """Capture observed runtime properties without changing model state."""
    thinker = getattr(generation_model, "thinker", llm.model)
    config = getattr(generation_model, "config", None)
    thinker_config = getattr(thinker, "config", None)
    text_config = getattr(thinker_config, "text_config", None)
    actual_attention = _attention_implementation(config)
    thinker_attention = _attention_implementation(thinker_config)
    text_attention = _attention_implementation(text_config)
    device_map = getattr(generation_model, "hf_device_map", None)
    if isinstance(device_map, dict):
        device_map = {str(key): str(value) for key, value in device_map.items()}
    elif device_map is not None:
        device_map = str(device_map)

    metadata = {
        "wrapper_class": generation_model.__class__.__name__,
        "thinker_class": thinker.__class__.__name__,
        "wrapper_first_float_dtype": _first_floating_parameter_dtype(generation_model),
        "thinker_first_float_dtype": _first_floating_parameter_dtype(thinker),
        "thinker_q_proj_dtype": _first_q_proj_dtype(thinker),
        "attention_requested": args.attn_implementation,
        "attention_actual": actual_attention,
        "thinker_attention_actual": thinker_attention,
        "text_decoder_attention_actual": text_attention,
        "talker_enabled": bool(getattr(generation_model, "has_talker", False)),
        "hf_device_map": device_map,
        "parameter_devices": _parameter_devices(generation_model),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "visible_gpu_count": torch.cuda.device_count(),
    }
    if args.method_name == "dense-bf16":
        expected = str(torch.bfloat16)
        if metadata["thinker_first_float_dtype"] != expected or metadata["thinker_q_proj_dtype"] != expected:
            raise RuntimeError(
                "dense-bf16 requires BF16 Thinker parameters and first q_proj weight; "
                f"observed thinker={metadata['thinker_first_float_dtype']}, "
                f"q_proj={metadata['thinker_q_proj_dtype']}."
            )
        if metadata["talker_enabled"]:
            raise RuntimeError("dense-bf16 ASR requires the Qwen talker to be disabled.")
        if metadata["visible_gpu_count"] != 1:
            raise RuntimeError(
                "dense-bf16 ASR requires exactly one CUDA-visible GPU; "
                f"observed {metadata['visible_gpu_count']}."
            )
        if metadata["text_decoder_attention_actual"] != args.attn_implementation:
            raise RuntimeError(
                "dense-bf16 ASR attention mismatch for the text decoder: "
                f"requested={args.attn_implementation}, "
                f"actual={metadata['text_decoder_attention_actual']}."
            )
    return metadata


def asr_response_metadata(args, runtime_metadata, generation_config):
    dtype_evidence = "runtime_verified" if args.method_name == "dense-bf16" else "runtime_observed"
    return {
        "model": args.model,
        "method": args.method_name,
        "model_dtype": args.model_dtype_label,
        "model_dtype_torch": runtime_metadata["thinker_first_float_dtype"],
        "dtype_evidence": dtype_evidence,
        "paper_baseline_label": "Dense FP16",
        "dataset": "LibriSpeech",
        "split": args.split,
        "prompt_label": args.prompt_label,
        "system_prompt": args.asr_system_prompt,
        "user_prompt": args.asr_prompt,
        "prompt_source": args.prompt_source,
        "attention_requested": runtime_metadata["attention_requested"],
        "attention_actual": runtime_metadata["attention_actual"],
        "thinker_attention_actual": runtime_metadata["thinker_attention_actual"],
        "text_decoder_attention_actual": runtime_metadata["text_decoder_attention_actual"],
        "talker_enabled": runtime_metadata["talker_enabled"],
        "return_audio": False,
        "generation_config": generation_config,
        "normalization": args.normalization,
        "subset_name": args.subset_name,
        "quantization": "None" if args.method_name == "dense-bf16" else "unspecified",
        "logical_weight_bits": getattr(args, "wbits", None),
        "logical_activation_bits": getattr(args, "abits", None),
        "mas_enabled": bool(getattr(args, "quantize", False)),
        "cmc_enabled": bool(getattr(args, "LR", False) and getattr(args, "rank", 0) > 0),
        "cmc_scope": getattr(args, "cmc_scope", ""),
        "quant_cmc": getattr(args, "quant_cmc", None),
        "rank_argument": getattr(args, "rank", None),
        "full_rank": bool(getattr(args, "full_rank", False)),
        "fullrank_adapter_sha256": getattr(args, "fullrank_adapter_sha256", ""),
        "scales_sha256": getattr(args, "scales_sha256", ""),
        "white_matrix_sha256": getattr(args, "white_matrix_sha256", ""),
        "normalizer_source_repository": "QwenLM/Qwen2-Audio" if args.normalization == "qwen_asr_en" else "",
        "normalizer_source_commit": "595360e82b5839c1507492ec83cae5bda6d5c7d4" if args.normalization == "qwen_asr_en" else "",
        "normalizer_pipeline": [
            "str",
            "remove_qwen_special_tokens",
            "EnglishTextNormalizer",
            "BasicTextNormalizer",
            "EvaluationTokenizer(tokenizer_type=none, lowercase=True, punctuation_removal=True, character_tokenization=False)",
            "collapse_whitespace_strip",
        ] if args.normalization == "qwen_asr_en" else [],
        "prefix_stripping": False if args.normalization == "qwen_asr_en" else None,
    }


def score_librispeech_jsonl(args):
    input_path = Path(args.input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Response JSONL does not exist: {input_path}")

    scored_path = Path(args.scored_output) if args.scored_output else None
    summary_path = Path(args.score_output) if args.score_output else None
    for path in (scored_path, summary_path):
        if path and path.exists():
            raise FileExistsError(f"Refusing to overwrite existing score output: {path}")
    if scored_path:
        scored_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {"substitutions": 0, "insertions": 0, "deletions": 0, "reference_words": 0}
    sample_count = 0
    failed = 0
    empty_hyp = 0
    hit_max = 0
    examples = []
    wer_impl = "levenshtein_fallback"
    jiwer_version = None
    response_metadata = None

    scored_handle = scored_path.open("w", encoding="utf-8") if scored_path else None
    try:
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_count += 1
                if response_metadata is None:
                    response_metadata = {
                        key: record.get(key)
                        for key in (
                            "model",
                            "method",
                            "model_dtype",
                            "model_dtype_torch",
                            "dtype_evidence",
                            "paper_baseline_label",
                            "quantization",
                            "dataset",
                            "split",
                            "subset_name",
                            "prompt_label",
                            "system_prompt",
                            "user_prompt",
                            "prompt_source",
                            "attention_requested",
                            "attention_actual",
                            "thinker_attention_actual",
                            "text_decoder_attention_actual",
                            "talker_enabled",
                            "return_audio",
                            "generation_config",
                            "logical_weight_bits",
                            "logical_activation_bits",
                            "mas_enabled",
                            "cmc_enabled",
                            "cmc_scope",
                            "quant_cmc",
                            "rank_argument",
                            "full_rank",
                            "fullrank_adapter_sha256",
                            "scales_sha256",
                            "white_matrix_sha256",
                            "normalizer_source_repository",
                            "normalizer_source_commit",
                            "normalizer_pipeline",
                            "prefix_stripping",
                        )
                    }
                reference_raw = record.get("reference", "")
                hypothesis_raw = get_record_hypothesis(record)
                reference = normalize_librispeech_text(reference_raw, args.normalization)
                hypothesis = normalize_librispeech_text(hypothesis_raw, args.normalization)
                if record.get("error"):
                    failed += 1
                if not hypothesis:
                    empty_hyp += 1
                if record.get("hit_max_new_tokens"):
                    hit_max += 1
                substitutions, insertions, deletions, wer_impl, jiwer_version = word_error_counts(
                    reference,
                    hypothesis,
                    require_jiwer=args.normalization == "qwen_asr_en",
                )
                reference_words = len(reference.split())
                totals["substitutions"] += substitutions
                totals["insertions"] += insertions
                totals["deletions"] += deletions
                totals["reference_words"] += reference_words
                sample_wer = (substitutions + insertions + deletions) / reference_words if reference_words else 0.0
                scored_record = {
                    **record,
                    "hypothesis": hypothesis_raw,
                    "reference_normalized": reference,
                    "hypothesis_normalized": hypothesis,
                    "substitutions": substitutions,
                    "insertions": insertions,
                    "deletions": deletions,
                    "reference_words": reference_words,
                    "sample_wer": sample_wer,
                    "normalization": args.normalization,
                    "line_number": line_number,
                }
                if len(examples) < 3:
                    examples.append(
                        {
                            "id": record.get("id"),
                            "reference": reference,
                            "hypothesis": hypothesis,
                            "sample_wer": sample_wer,
                        }
                    )
                if scored_handle:
                    scored_handle.write(json.dumps(scored_record, ensure_ascii=False) + "\n")
    finally:
        if scored_handle:
            scored_handle.close()

    edit_distance = totals["substitutions"] + totals["insertions"] + totals["deletions"]
    wer_fraction = edit_distance / totals["reference_words"] if totals["reference_words"] else 0.0
    summary = {
        **(response_metadata or {}),
        "input_file": str(input_path),
        "scored_output": str(scored_path) if scored_path else "",
        "samples": sample_count,
        "failed": failed,
        "empty_hyp": empty_hyp,
        "hit_max_new_tokens": hit_max,
        "reference_words": totals["reference_words"],
        "substitutions": totals["substitutions"],
        "insertions": totals["insertions"],
        "deletions": totals["deletions"],
        "edit_distance": edit_distance,
        "wer_fraction": wer_fraction,
        "wer_percent": wer_fraction * 100,
        "wer_implementation": wer_impl,
        "jiwer_version": jiwer_version,
        "normalization": args.normalization,
        "examples": examples,
    }
    if summary_path:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("LIBRISPEECH_SCORE_SUMMARY " + json.dumps(summary, ensure_ascii=False))


@torch.no_grad()
def evaluate_librispeech_wer(llm, args, logger):
    from qwen_omni_utils import process_mm_info
    from transformers import AutoProcessor

    samples = load_librispeech_samples(args.librispeech_root, args.split, args.max_samples)
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    generation_model = getattr(llm, "model_origin", llm.model)
    if hasattr(generation_model, "thinker"):
        generation_model.thinker = llm.model
    generation_model.eval()
    runtime_metadata = collect_qwen_omni_runtime_metadata(llm, generation_model, args)

    output_handle = None
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    totals = {"substitutions": 0, "insertions": 0, "deletions": 0, "reference_words": 0}
    failed = 0
    empty_hyp = 0
    possible_truncation = 0
    examples = []
    wer_impl = "levenshtein_fallback"
    jiwer_version = None

    print(f"ASR model class: {generation_model.__class__.__name__}")
    print(f"ASR thinker class: {llm.model.__class__.__name__}")
    print(f"ASR talker enabled: {bool(getattr(generation_model, 'has_talker', False))}")
    print("ASR runtime metadata: " + json.dumps(runtime_metadata, ensure_ascii=False))
    print(f"ASR samples: {len(samples)}, split: {args.split}, max_new_tokens: {args.max_new_tokens}")

    try:
        for index, sample in enumerate(samples, start=1):
            reference = normalize_librispeech_text(sample["reference"], args.normalization)
            if not reference:
                print(f"[{index}/{len(samples)}] {sample['id']} skipped: empty reference")
                failed += 1
                continue

            hypothesis_raw = ""
            generated_token_count = 0
            error = None
            generation_kwargs = {
                "return_audio": False,
                "use_audio_in_video": False,
                "do_sample": False,
                "repetition_penalty": 1.0,
                "num_beams": 1,
            }
            if hasattr(generation_model, "thinker"):
                generation_kwargs["thinker_max_new_tokens"] = args.max_new_tokens
            else:
                generation_kwargs["max_new_tokens"] = args.max_new_tokens
            try:
                messages = []
                if args.asr_system_prompt:
                    messages.append(
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": args.asr_system_prompt}],
                        }
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": sample["audio"]},
                            {"type": "text", "text": args.asr_prompt},
                        ],
                    }
                )
                conversations = [messages]
                text = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
                inputs = processor(
                    text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=False,
                )
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.to(llm.device)
                        if torch.is_floating_point(value):
                            value = value.to(llm.model.dtype)
                        inputs[key] = value

                output_ids = generation_model.generate(**inputs, **generation_kwargs)
                prompt_length = inputs["input_ids"].shape[1]
                generated_ids = output_ids[0, prompt_length:]
                generated_token_count = int(generated_ids.numel())
                hypothesis_raw = processor.batch_decode(
                    [generated_ids],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            except Exception as exc:
                failed += 1
                error = f"{type(exc).__name__}: {exc}"

            hypothesis = normalize_librispeech_text(hypothesis_raw, args.normalization)
            if not hypothesis:
                empty_hyp += 1
            hit_max_new_tokens = generated_token_count >= args.max_new_tokens - 1
            if hit_max_new_tokens:
                possible_truncation += 1

            substitutions, insertions, deletions, wer_impl, jiwer_version = word_error_counts(
                reference,
                hypothesis,
                require_jiwer=args.normalization == "qwen_asr_en",
            )
            reference_words = len(reference.split())
            totals["substitutions"] += substitutions
            totals["insertions"] += insertions
            totals["deletions"] += deletions
            totals["reference_words"] += reference_words
            edit_distance = totals["substitutions"] + totals["insertions"] + totals["deletions"]
            cumulative_wer = edit_distance / totals["reference_words"] if totals["reference_words"] else 0.0

            record = {
                **sample,
                **asr_response_metadata(args, runtime_metadata, generation_kwargs),
                "reference_normalized": reference,
                "hypothesis": hypothesis_raw,
                "hypothesis_normalized": hypothesis,
                "generated_token_count": generated_token_count,
                "hit_max_new_tokens": hit_max_new_tokens,
                "error": error,
                "cumulative_wer_percent": cumulative_wer * 100,
            }
            if len(examples) < 3:
                examples.append({"id": sample["id"], "reference": reference, "hypothesis": hypothesis})
            if output_handle:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
            print(
                f"[{index}/{len(samples)}] id={sample['id']} tokens={generated_token_count} "
                f"hit_max={hit_max_new_tokens} cumulative_wer={cumulative_wer * 100:.4f}%"
            )
            print(f"  ref: {reference}")
            print(f"  hyp: {hypothesis}")
            if error:
                print(f"  error: {error}")
    finally:
        if output_handle:
            output_handle.close()

    edit_distance = totals["substitutions"] + totals["insertions"] + totals["deletions"]
    final_wer = edit_distance / totals["reference_words"] if totals["reference_words"] else 0.0
    summary = {
        "samples": len(samples),
        "failed": failed,
        "empty_hyp": empty_hyp,
        "possible_truncation": possible_truncation,
        "reference_words": totals["reference_words"],
        "substitutions": totals["substitutions"],
        "insertions": totals["insertions"],
        "deletions": totals["deletions"],
        "edit_distance": edit_distance,
        "wer_fraction": final_wer,
        "wer_percent": final_wer * 100,
        "wer_implementation": wer_impl,
        "jiwer_version": jiwer_version,
        "runtime_metadata": runtime_metadata,
        "method": args.method_name,
        "model_dtype": args.model_dtype_label,
        "model_dtype_torch": runtime_metadata["thinker_first_float_dtype"],
        "dtype_evidence": "runtime_verified" if args.method_name == "dense-bf16" else "runtime_observed",
        "paper_baseline_label": "Dense FP16",
        "prompt_label": args.prompt_label,
        "system_prompt": args.asr_system_prompt,
        "user_prompt": args.asr_prompt,
        "normalization": args.normalization,
        "examples": examples,
    }
    print("LIBRISPEECH_WER_SUMMARY " + json.dumps(summary, ensure_ascii=False))
    logger.info("LibriSpeech WER summary: %s", summary)
    return summary

def compute_sqnr(original, quantized):
    error = original - quantized
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(error ** 2)
    sqnr = 10 * torch.log10(signal_power / noise_power)
    
    return sqnr.item()

def compute_sqnr_per_modality(original, quantized, audio_mask, vision_mask, vision_alpha=0.5, audio_alpha=1.0):
    """
    计算每个 token 的 SQNR，然后返回平均值
    
    Args:
        original: [batch, seq_len, hidden_dim] 如 [1, 4464, 2048]
        quantized: [batch, seq_len, hidden_dim] 如 [1, 4464, 2048]
    
    Returns:
        mean_sqnr: 所有 token 的平均 SQNR
    """
    error = original.float() - quantized.float()
    # 在最后一维（hidden_dim）上计算信号和噪声功率
    # 结果 shape: [1, 4464]
    signal_power = torch.mean(original ** 2, dim=-1)  # [1, 4464]
    noise_power = torch.mean(error ** 2, dim=-1)      # [1, 4464]
    # 计算每个 token 的 SQNR
    # 添加一个小的 epsilon 避免除零
    sqnr = 10 * torch.log10(signal_power / (noise_power))  # [1, 4464]
    all_true = torch.full(vision_mask.shape, True, dtype=torch.bool).to(vision_mask.device)
    text_mask = all_true & ~audio_mask & ~vision_mask
    sqnr_vision = torch.mean(sqnr[vision_mask])
    sqnr_audio = torch.mean(sqnr[audio_mask])
    sqnr_text = torch.mean(sqnr[text_mask])
  
    # 对所有 token 求平均，注意 vision 的乘以了一个 0.5的系数!!!!
    mean_sqnr = torch.mean(torch.stack([vision_alpha*sqnr_vision, audio_alpha*sqnr_audio, sqnr_text]))
    return mean_sqnr.item()

def compute_sqnr_per_token(original, quantized):
    """
    计算每个 token 的 SQNR，然后返回所有 token 的平均 SQNR。
    
    Args:
        original (torch.Tensor): 原始浮点隐藏状态 (batch_size, sequence_length, hidden_size)
        quantized (torch.Tensor): 量化隐藏状态 (batch_size, sequence_length, hidden_size)
        
    Returns:
        float: 所有 token 的平均 SQNR (dB)
    """
    # original/quantized shape: (batch_size, sequence_length, hidden_size)

    # 1. 计算误差 (噪声)
    error = original - quantized

    # 2. 计算信号功率：对每个 token 向量在 hidden_size 维度上求均值
    # 结果 shape: (batch_size, sequence_length)
    signal_power = torch.mean(original ** 2, dim=-1)

    # 3. 计算噪声功率：对每个 token 向量在 hidden_size 维度上求均值
    # 结果 shape: (batch_size, sequence_length)
    noise_power = torch.mean(error ** 2, dim=-1)

    # 4. 计算每个 token 的 SQNR (dB)
    # 使用 torch.clamp 来避免除以零或对零取对数
    sqnr_matrix = 10 * torch.log10(signal_power / torch.clamp(noise_power, min=1e-5))
    
    # 5. 计算所有 token 的平均 SQNR，并返回 Python float
    mean_sqnr = torch.mean(sqnr_matrix)
    
    return mean_sqnr.item()

@torch.no_grad()
def evaluate(llm, args, logger):
    results = {}
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(llm.model.model.decoder.layers)
            input_device = llm.model.model.decoder.layers[0].device
            output_device = llm.model.model.decoder.layers[-1].device
            llm._device = input_device
            assert input_device == output_device
            llm.model.model.decoder.embed_positions.to(input_device)
            llm.model.model.decoder.embed_tokens.to(input_device)
            llm.model.model.decoder.final_layer_norm.to(output_device)
            llm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower() or "mixtral" in args.net.lower() or "qwen" in args.net.lower():
            map_layers_to_multi_gpus(llm.model.model.layers)
            input_device = llm.model.model.layers[0].device
            output_device = llm.model.model.layers[-1].device
            assert input_device == output_device
            llm._device = input_device
            llm.model.model.embed_tokens.to(input_device)
            llm.model.model.norm.to(output_device)
            llm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(llm.model.transformer.h)
            input_device = llm.model.transformer.h[0].device
            output_device = llm.model.transformer.h[-1].device
            assert input_device == output_device
            llm._device = input_device
            llm.model.transformer.word_embeddings.to(input_device)
            llm.model.transformer.ln_f.to(output_device)
            llm.model.lm_head.to(output_device)
    else:
        if "opt" in args.net.lower():
            llm.model.model.decoder = llm.model.model.decoder.to(llm.device)
        elif "llama" in args.net.lower() or "mixtral" in args.net.lower() or 'qwen' in args.net.lower()  or 'minicpm' in args.net.lower():
            llm.model = llm.model.to(llm.device)
        elif "falcon" in args.net.lower():
            llm.model.transformer = llm.model.transformer.to(llm.device)

    if args.eval_sqnr:
        from more_itertools import batched
        import json
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model)
        from qwen_omni_utils import process_mm_info
        USE_AUDIO_IN_VIDEO = False
        cur_index = 0
        sqnr_total = []
        csv_file_path = args.sqnr_result
        llm.model.eval()
        llm.model_origin.eval()
        with open(args.input_file) as f:
            for lines in batched(f, args.batch_size):
                datas = [json.loads(line) for line in lines]

                conversations = [data["prompt"] for data in datas]

                text = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                audios, images, videos = process_mm_info(
                    conversations, use_audio_in_video=False
                )

                inputs = processor(
                    text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                )
                inputs = inputs.to(llm.device).to(llm.model.dtype)
                
                llm.model_origin.to(llm.device)
                inputs['output_hidden_states'] = True
                output_quant = llm.model(**inputs)
                if "omni" in args.net.lower():
                    output_float = llm.model_origin.thinker(**inputs)
                else:
                    output_float = llm.model_origin.model(**inputs)
                
                all_hidden_states_float = output_float[2]
                all_hidden_states_quant = output_quant[2]
                # 对每个样本记录每一层的 SQNR
                layer_sqnr = [0] * (len(all_hidden_states_float) ) # 初始化每层 SQNR 列表
                
                for layer_index in range(1, len(all_hidden_states_float)):
                    sqnr = compute_sqnr_per_token(all_hidden_states_float[layer_index], all_hidden_states_quant[layer_index])
                    # sqnr = compute_sqnr_per_modality(all_hidden_states_float[layer_index], all_hidden_states_quant[layer_index], audio_mask, image_mask)
                    layer_sqnr[layer_index - 1] = sqnr  # 记录当前层 SQNR 
                
                # sqnr_final = compute_sqnr(output_float[0], output_quant[0])
                # layer_sqnr[layer_index] = sqnr_final
                sqnr_total.append(layer_sqnr)  # 保存当前样本的 SQNR 值
                # 把最终的logits 也计算一下 sqnr
                print(f'cur_index: {cur_index}, sqnr_layers: {layer_sqnr}')
                cur_index += 1
                
                if cur_index >= 32:  # 假设最多处理 32 个样本
                    break

        # 计算所有层的平均 SQNR
        mean_sqnr = [sum(layer)/cur_index for layer in zip(*sqnr_total)]

        # 将结果写入 CSV 文件
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        with open(csv_file_path, mode='w', newline='') as csv_file:
            # 设定表头，只包含每层的 SQNR 值
            fieldnames = [f'layer_{i}_sqnr' for i in range(1, len(mean_sqnr)+1)]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writeheader()  # 写入表头
            
            # 写入每个样本的 SQNR 值
            for layer_sqnr in sqnr_total:
                writer.writerow({f'layer_{i+1}_sqnr': sqnr for i, sqnr in enumerate(layer_sqnr)})
            
            # 写入平均值行
            writer.writerow({f'layer_{i+1}_sqnr': mean_sqnr[i] for i in range(len(mean_sqnr))})
            print({f'layer_{i+1}_sqnr': mean_sqnr[i] for i in range(len(mean_sqnr))})

        print(f'完成 SQNR 计算，结果已写入 {csv_file_path} \n')

    if args.eval_omni_task:
        # ------------------------------------------------------------------
        # Local evaluation branch
        # Purpose: run global LibriSpeech WER scoring before generic task evaluation.
        # Upstream has no ASR WER branch at this location.
        # ------------------------------------------------------------------
        if args.compute_wer:
            results["librispeech_wer"] = evaluate_librispeech_wer(llm, args, logger)
            return results
        from more_itertools import batched
        import json
        from transformers import AutoProcessor
        # ------------------------------------------------------------------
        # Upstream implementation retained for comparison
        # Upstream: processor = AutoProcessor.from_pretrained(args.model)
        # Local: use local_files_only/trust_remote_code and generate through the
        # complete wrapper with the quantized thinker attached for text-only ASR.
        # This changes evaluation wiring only.
        # ------------------------------------------------------------------
        processor = AutoProcessor.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
        from qwen_omni_utils import process_mm_info
        USE_AUDIO_IN_VIDEO = False
        generation_model = getattr(llm, "model_origin", llm.model)
        if hasattr(generation_model, "thinker"):
            generation_model.thinker = llm.model
        generation_model.eval()
        file_index = 0
        allocated = torch.cuda.memory_allocated(llm.device) / 1024**2
        print(f'allocated:  {allocated:.2f} MB')
        
        with open(args.input_file) as f, open(args.output_file, "w", encoding='utf-8') as fw:
            for lines in batched(f, args.batch_size):
                datas = [json.loads(line) for line in lines]
                file_index += 1

                # if file_index < 59:
                #     continue

                allocated = torch.cuda.memory_allocated(llm.device) / 1024**2
                # print(f'allocated_{file_index}:  {allocated:.2f} MB, before load data')

                conversations = [data["prompt"] for data in datas]

                text = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                audios, images, videos = process_mm_info(
                    conversations, use_audio_in_video=False
                )

                inputs = processor(
                    text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                )
                inputs = inputs.to(llm.device).to(llm.model.dtype)
                allocated = torch.cuda.memory_allocated(llm.device) / 1024**2
                # print(f'allocated_{file_index}:  {allocated:.2f} MB, load data done.')
                                
                # ------------------------------------------------------------------
                # Upstream implementation retained for comparison
                # Upstream: text_ids = llm.model.generate(..., max_new_tokens=128)
                # Local: keep wrapper generation, explicit text-only settings, and
                # a configurable output limit while preserving greedy decoding.
                # ------------------------------------------------------------------
                generation_kwargs = {
                    "use_audio_in_video": USE_AUDIO_IN_VIDEO,
                    "return_audio": False,
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                    "num_beams": 1,
                }
                if hasattr(generation_model, "thinker"):
                    generation_kwargs["thinker_max_new_tokens"] = args.max_new_tokens
                else:
                    generation_kwargs["max_new_tokens"] = args.max_new_tokens
                text_ids = generation_model.generate(
                    **inputs,
                    **generation_kwargs,
                )
                allocated = torch.cuda.memory_allocated(llm.device) / 1024**2
                # print(f'allocated_{file_index}:  {allocated:.2f} MB, inference done')
                generated_ids_list = [
                    text_ids[i][len(inputs["input_ids"][i]) :] for i in range(len(datas))
                ]
                
                # generated_ids_list = text_ids[:, inputs.input_ids.size(1):]
                response_text = processor.batch_decode(
                    generated_ids_list,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for data, response, generated_ids in zip(datas, response_text, generated_ids_list):
                    generated_token_count = int(generated_ids.numel())
                    record = {
                        **data,
                        "response": response,
                        "hypothesis": response,
                        "generated_token_count": generated_token_count,
                        "hit_max_new_tokens": generated_token_count >= args.max_new_tokens - 1,
                        "generation_config": {
                            "do_sample": False,
                            "num_beams": 1,
                            "max_new_tokens": args.max_new_tokens,
                            "return_audio": False,
                            "use_audio_in_video": False,
                            "repetition_penalty": 1.0,
                            "no_repeat_ngram_size": None,
                        },
                        "error": None,
                    }
                    json_data = json.dumps(record, ensure_ascii=False)
                    fw.write(json_data + "\n")
                    fw.flush()

    if args.eval_ppl:
        results_ppl = {}
        csv_file_path = args.ppl_result
        # for dataset in ["wikitext2", "c4-new"]:
        for dataset in ["wikitext2"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader, weights_only=False)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=llm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // llm.seqlen
            
            
            if 'Qwen2.5-Omni' in args.model:
                use_cache = llm.model.config.text_config.use_cache
                llm.model.config.text_config.use_cache = False
            elif 'MiniCPM' in args.model:
                use_cache = llm.model.config.use_cache
                llm.model.config.use_cache = False            
            
            llm.model.eval()
            nlls = []
            # nsamples = 1
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * llm.seqlen) : ((i + 1) * llm.seqlen)].to(llm.device)
                if "opt" in args.net.lower():
                    outputs = llm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mixtral" in args.net.lower() or 'qwen' in args.net.lower() or 'minicpm' in args.net.lower():
                    outputs = llm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = llm.model.transformer(batch)

                hidden_states = outputs[0]
                if hidden_states.dtype != llm.model.lm_head.weight.dtype:
                    llm.model.lm_head.to(hidden_states.dtype)
                logits = llm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * llm.seqlen) : ((i + 1) * llm.seqlen)][
                    :, 1:
                ].to(llm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * llm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * llm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            if 'Qwen2.5-Omni' in args.model:
                llm.model.config.text_config.use_cache = use_cache
            elif 'MiniCPM' in args.model:
                llm.model.config.use_cache = use_cache
            
            results_ppl[dataset] = ppl.item()
        # 将结果写入 CSV 文件
        with open(csv_file_path, mode='w', newline='') as csv_file:
            # 设定表头，只包含每层的 SQNR 值
            fieldnames = results_ppl.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writeheader()  # 写入表头
            
            # 写入每个样本的 SQNR 值
            writer.writerow(results_ppl)

        print(f'完成 PPL 计算，结果已写入 {csv_file_path} \n')        
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            llm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit
        )
        results.update(t_results)
        logger.info(results)
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))
    
    if args.tasks_multimodal != "":
        
        if 'Omni' in args.model:
            print(f'--------->>>>>>>>>>>>>>>>>>>> omni model !!!!!!!!!!!')
            from models.LMMClass_Omni import LMMClass
            vlm = LMMClass(args.model)
            vlm.model.thinker.model = llm.model.model
            t_results = eval_multimodal.simple_evaluate(
                vlm,
                tasks=args.tasks_multimodal.split(","),
                num_fewshot=args.num_fewshot,
                limit=None if args.limit_multimodal == 1.0 else args.limit_multimodal
            )
            results.update(t_results['results'])
            logger.info(results)
        else:
            from models.LMMClass import LMMClass
            vlm = LMMClass(args.model, llm.model)
            
            t_results = eval_multimodal.simple_evaluate(
                vlm,
                tasks=args.tasks_multimodal.split(","),
                num_fewshot=args.num_fewshot,
                limit=None if args.limit_multimodal == 1.0 else args.limit_multimodal,
                gen_kwargs="max_new_tokens=128"
            )
            results.update(t_results['results'])
            logger.info(results)
            print(f'tasks_multimodal:  {results}')
    return results


def main_entry(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'infer'], help="training or inference mode")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--output_dir_postfix", type=str, default="", help="post fix for output dir")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--calib_dataset",type=str,default="omnibench",
        choices=["wikitext2", "ptb", "c4", "mix","pile", "omnibench"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--tasks_multimodal", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--auto_scale", action="store_true")
    parser.add_argument("--auto_alpha", action="store_true")
    parser.add_argument("--auto_epochs", action="store_true")
    parser.add_argument("--loss_multi_modal", action="store_true")
    parser.add_argument("--loss_multi_modal_mae", action="store_true")
    parser.add_argument("--loss_multi_modal_mae_alpha", action="store_true")
    parser.add_argument("--ppl_result", default="ppl_result.csv")
    parser.add_argument("--eval_sqnr", action="store_true")
    parser.add_argument("--sqnr_result", default="sqnr_result.csv")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-2)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point",default=False, action="store_true", help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--limit_multimodal", type=float, default=1.0)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    parser.add_argument("--input_file", default="")
    parser.add_argument("--output_file", default="")    
    parser.add_argument("--grad_info_path", default="")    
    parser.add_argument("--eval_omni_task", action="store_true")
    # ------------------------------------------------------------------
    # Local command-line additions
    # Purpose: expose LibriSpeech input construction and offline scoring controls.
    # Upstream has no corresponding ASR argument block.
    # ------------------------------------------------------------------
    parser.add_argument("--compute_wer", action="store_true")
    parser.add_argument("--make_librispeech_jsonl", action="store_true")
    parser.add_argument("--score_librispeech_jsonl", action="store_true")
    parser.add_argument("--librispeech_root", default="/data/liuqinheng/benchmark/LibriSpeech")
    parser.add_argument("--split", default="test-other")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_order", default="sorted", choices=["sorted", "random", "speaker_stratified"])
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--subset_name", default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--asr_prompt", default=ASR_PROMPT)
    parser.add_argument("--asr_system_prompt", default="")
    parser.add_argument("--prompt_label", default="")
    parser.add_argument("--prompt_source", default="CLI arguments")
    parser.add_argument("--method_name", default="dense-bf16")
    parser.add_argument("--model_dtype_label", default="bfloat16")
    parser.add_argument(
        "--normalization",
        default="librispeech_basic",
        choices=["librispeech_basic", "qwen_asr_en"],
    )
    parser.add_argument("--score_output", default="")
    parser.add_argument("--scored_output", default="")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_memory_gib", type=int, default=None)
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
        help="Data type to calculate activation. Options: text-only, vision-only, audio-only, text-vision, text-audio, vision-audio"
    )
    inference_mode = os.getenv('inference_mode', 'merged_scales')
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # Local ASR utility dispatch; upstream continues directly to quantization setup.
    if args.make_librispeech_jsonl:
        make_librispeech_jsonl(args)
        return {}
    if args.score_librispeech_jsonl:
        score_librispeech_jsonl(args)
        return {}
  
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    grad_info = None
    if len(args.grad_info_path) > 0:
        grad_info = torch.load(args.grad_info_path, weights_only=False)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=4) or (args.abits<16 and args.abits>=4):
        args.deactive_amp = True

    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]

    # init logger
    if args.output_dir:
        from datetime import datetime
        current_time = datetime.now()
        formatted_with_ms = current_time.strftime("%m%d-%H%M%S.%f")

        # args.output_dir = f'{args.output_dir}/{args.net}-{args.dataset_type}-{args.epochs}epochs-w{args.wbits}a{args.abits}-{args.output_dir_postfix}-{formatted_with_ms}'
        args.output_dir = f'{args.output_dir}/{args.net}-{args.epochs}epochs-w{args.wbits}a{args.abits}-{args.output_dir_postfix}-{formatted_with_ms}-{inference_mode}'
        print(f'->>>>> output_dir is: {args.output_dir}/mas_parameters.pth ')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    logger.info(f'inference_mode:  {inference_mode}')

    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    llm = LMClass(args)
    llm.seqlen = 2048
    llm.model.eval()
    for param in llm.model.parameters():
        param.requires_grad = False

    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": True,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        llm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}-{args.dataset_type}-{args.nsamples}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}-{args.dataset_type}.pt'

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.net}_{args.dataset_type}_{args.nsamples}.cache'
        print(f'try to load cache from: {cache_dataloader}')
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
                # ------------------------------------------------------------------
                # Upstream: calibration_dataset = prepare_dataset(n_sample=args.nsamples, data_type=args.dataset_type)
                # Local: propagate the configured local LibriSpeech root and split.
                # This changes calibration dataset selection only.
                # ------------------------------------------------------------------
                calibration_dataset = prepare_dataset(
                    n_sample=args.nsamples,
                    data_type=args.dataset_type,
                    librispeech_root=args.librispeech_root,
                    split=args.split,
                )
                processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
                is_minicpm = 'MiniCPM' in args.model
                dataloader = prepare_dataset_before_quant(processor, calibration_dataset, batch_size=args.batch_size, is_minicpm=is_minicpm)
            else:
                dataloader, _ = get_loaders(
                    args.calib_dataset,
                    nsamples=args.nsamples,
                    seed=args.seed,
                    model=args.model,
                    seqlen=llm.seqlen,
                )
            torch.save(dataloader, cache_dataloader)
            
        act_scales = None
        if args.let and (args.resume is None):
            act_scales = torch.load(args.act_scales)
        masquant(
            llm,
            args,
            dataloader,
            act_scales,
            logger,
            grad_info
        )
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in llm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer) or isinstance(module,QuantLlamaDecoderLayerV2) :
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        llm.model.save_pretrained(args.save_dir)  
        llm.tokenizer.save_pretrained(args.save_dir) 
    evaluate(llm, args,logger)


if __name__ == "__main__":
    print(sys.argv)
    main_entry()
