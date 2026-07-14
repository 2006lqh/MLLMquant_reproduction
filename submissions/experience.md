# Reproduction Record

## Project Scope

This record covers Qwen2.5-Omni-3B audio-text evaluation on LibriSpeech
test-other and the MAS/CMC reproduction workflow.

## Environment

The evaluated environment uses the local MASQuant environment with PyTorch,
Transformers, jiwer, and local model and dataset assets.

## Dataset Preparation

LibriSpeech transcripts are paired with FLAC files by utterance ID. Evaluation
inputs preserve the audio path, reference, P0 prompt, split, and sample order.

## Scoring Protocol

Text is normalized with `librispeech_basic`; WER is accumulated globally from
word-level substitutions, insertions, deletions, and reference word counts.

## Experiment Chronology

Dense P0 full and sorted512 baselines were established before the W4A8 CMC
comparison. The no-CMC text-base unified-weight diagnostic exhibited repeated
generation and is retained as a diagnostic rather than a formal MAS result.

## Important Findings

MAS+CMC rank0.2 restored most of the no-CMC diagnostic degradation on sorted512,
but remained above the Dense baseline. No full or rank-sweep follow-up is implied
by this record.

## Source Modifications

The local changes are in `custom_dataset.py`, `generate_act_scale_shift.py`,
`infer_mas.py`, `main.py`, `models/LMClass.py`,
`models/modeling_qwen2_5_omni.py`, `quantize/infer_quant.py`,
`quantize/masquant.py`, and `quantize/svd_utils.py`.

They provide local LibriSpeech loading, ASR JSONL generation and scoring,
text-audio calibration and CMC support, local-only model loading, thinker-only
text generation, and device-safe temporary tensor handling. The replaced upstream
blocks are retained as comments adjacent to active local code. These are
engineering and benchmark-evaluation adaptations; they do not introduce a new
MAS or CMC mathematical method. Prompt and data-selection options define the
benchmark protocol and therefore must be recorded with results.

## Discarded Experiments

Prompt sweeps, rank sweeps, sanity runs, duplicate outputs, and incomplete
outputs are discarded after their verified metrics or diagnostic purpose are
recorded.

## Artifact Layout

- Experiment results: `/home/zhouyangchengyu/project_origin/submissions/experiments`
- Inputs: `/home/zhouyangchengyu/project_origin/submissions/experiments/inputs`
- Logs: `/home/zhouyangchengyu/project_origin/submissions/logs`
- Caches: `/home/zhouyangchengyu/project_origin/cache`

## Naming Convention

Experiment artifacts use `<model>__<method>__<dataset>__<subset>__<prompt>__<artifact>`.
Cache artifacts use `<model>__<method>__<calibration>__<cache-type>`.

## Maintenance Policy

`result.md` and `experience.md` are updated only after an explicit project
instruction. New experiment results, inputs, logs, and caches must use the
artifact layout above. Future source modifications must retain replaced upstream
code blocks as comments adjacent to the active implementation. Independent
`.old`, `.orig`, and `.bak` source copies are not used.
