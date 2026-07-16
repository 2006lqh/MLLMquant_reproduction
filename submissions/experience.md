# Reproduction Record

## Project Scope

This record covers Qwen2.5-Omni-3B audio-text evaluation on LibriSpeech
test-other and the MAS/CMC reproduction workflow.

## Dense Precision Terminology

The released Qwen2.5-Omni branch in `models/LMClass.py` passes
`torch_dtype=torch.bfloat16`, enables `device_map="auto"`, disables audio
output, forwards the requested attention implementation, and uses the Thinker
component for this workflow. Therefore new released-loader dense runs use the
method name `dense-bf16` and dtype label `bfloat16`.

The MASQuant paper's original table label remains `Dense FP16`. The released
loader does not prove the paper experiment's actual runtime dtype, so the two
statements must not be conflated.

The retained Dense P0 full2939 log identifies Qwen2.5-Omni, calls `main.py`,
and corresponds to the retained response and summary, but it does not print a
runtime dtype. It is therefore `INFERRED_BF16_NOT_DIRECTLY_LOGGED`, not
runtime-verified BF16. The retained sorted512 score is derived from that P0
response and has the same evidence level. P3 has no retained runtime log or
artifact and its dtype is `NOT VERIFIED`.

## Environment and Attention

The evaluated environment uses the local MASQuant environment with PyTorch,
Transformers, jiwer, and local model and dataset assets. Dtype and attention
backend are independent fields. The released Qwen2.5-Omni loader receives the
requested `--attn_implementation`; the local parser default is `eager`.
Historical P0 records `attn_implementation='eager'` as requested, but its
actual runtime backend is `NOT VERIFIED`. Future runs must record both the
requested backend and the loaded configuration value before generation.

## Dataset and Scoring Protocol

LibriSpeech transcripts are paired with FLAC files by utterance ID. Historical
`librispeech_basic` scoring lowercases, strips punctuation, normalizes
whitespace, strips fixed ASR response prefixes, and aggregates global S/I/D/N.
The full2939 Dense P0 summary records `levenshtein_fallback`; that historical
result was not scored by jiwer. The sorted512 retained summaries record
`jiwer.process_words` with historical version `unknown`.

Future P1 runs use `qwen_asr_en`: symmetric lowercase, punctuation stripping,
and whitespace normalization without prefix stripping, prompt-specific edits,
mr/mister substitutions, manual number fixes, or exclusions for empty and
hit-max responses. This strict mode requires jiwer and logs the actual scorer
and version.

## Future P1 Protocol

P1 system prompt: `You are a speech recognition model.`

P1 user prompt: `Transcribe the English audio into text without any punctuation marks.`

Future P1 Dense runs must use `--method_name dense-bf16`,
`--model_dtype_label bfloat16`, `--prompt_label p1`, and runtime validation of
the Thinker first floating parameter and first `q_proj.weight` as
`torch.bfloat16`. Response records must include model, method, dtype evidence,
paper baseline label, prompt fields, attention requested/actual, Talker state,
`return_audio`, generation configuration, and normalization.

## Source Modifications

The local changes are in `custom_dataset.py`, `generate_act_scale_shift.py`,
`infer_mas.py`, `main.py`, `models/LMClass.py`,
`models/modeling_qwen2_5_omni.py`, `quantize/infer_quant.py`,
`quantize/masquant.py`, and `quantize/svd_utils.py`.

They provide local LibriSpeech loading, ASR JSONL generation and scoring,
text-audio calibration and CMC support, local-only model loading, thinker-only
text generation, and device-safe temporary tensor handling. These are
engineering and benchmark-evaluation adaptations; they do not introduce a new
MAS or CMC mathematical method.

## Artifact Policy

- Experiment artifacts: `/home/zhouyangchengyu/project_origin/submissions/experiments`
- Logs: `/home/zhouyangchengyu/project_origin/submissions/logs`
- Caches: `/home/zhouyangchengyu/project_origin/cache`

Only response, scored, summary, and run-log artifacts are retained by default.
Do not create archived input manifests, comparison artifacts, or diagnostic
artifacts unless explicitly requested. Historical logs are preserved as original
runtime output; they are not rewritten when artifact naming changes.

## Current Artifact Inventory

As of 2026-07-17, six complete full2939 result sets are retained for
LibriSpeech test-other. Each set has a response JSONL, offline scored JSONL,
summary JSON, and matching run log:

- `dense-bf16` P0: historical retained run, `librispeech_basic` scoring and
  inferred dtype evidence only.
- `dense-bf16-fa2` P0 and P1: runtime-observed BF16 and FlashAttention 2.
- `mas-cmc-w4a8-fullrank-audio-fa2` P0 and P1: W4A8 MAS plus audio CMC with
  full rank, runtime-observed BF16 and FlashAttention 2.
- `mas-cmc-w4a8-rank0p2-audio-fa2` P1: W4A8 MAS plus audio-only CMC at rank
  ratio 0.2, runtime-observed BF16 and FlashAttention 2.

The `dense-bf16` P1 run log is retained as historical runtime output, but its
formal response, scored, and summary artifacts are not present. It is not a
completed result set and is intentionally excluded from `result.md`.

The rank0p2 P1 run passed CPU-only offline scoring after an exact 2939-ID
response integrity check. Its score is based on `qwen_asr_en` with
`jiwer.process_words` 4.0.0; no failed, empty, or hit-max responses were
recorded.

## Naming Convention

Experiment artifacts use `<model>__<method>__<dataset>__<subset>__<prompt>__<artifact>`.
Dense Qwen2.5-Omni releases use `dense-bf16`; quantized experiments retain their
actual quantization method names, such as `mas-cmc-w4a8-rank0p2`.

## Maintenance Policy

`result.md` and `experience.md` are updated only after an explicit project
instruction. Future source modifications must retain replaced upstream code
blocks as concise adjacent comments; independent `.old`, `.orig`, and `.bak`
source copies are not used.
