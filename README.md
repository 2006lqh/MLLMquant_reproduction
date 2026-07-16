# MASQuant Reproduction on Qwen2.5-Omni-3B

This repository documents a local reproduction of MASQuant audio-text quantization on Qwen2.5-Omni-3B. The evaluation task is English automatic speech recognition on the complete LibriSpeech `test-other` split (2,939 utterances), measured by word error rate (WER).

The repository preserves the EfficientAI source tree, calibration caches, formal outputs, scoring records, and concise reproduction notes. It does not claim that the local numbers are an exact reproduction of every paper setting.

## Highlights

- Evaluates Dense BF16 + FlashAttention 2 (FA2) and MAS+CMC W4A8 configurations.
- Uses Qwen2.5-Omni Thinker-only text generation; Talker speech output is disabled.
- Retains one response record and one scored record per utterance for each formal run.
- Scores references and hypotheses symmetrically with `qwen_asr_en` and `jiwer` 4.0.0.
- Separates directly observed results from possible explanations for differences from the paper.

## Method Overview

MASQuant addresses smoothing misalignment in multimodal models. Instead of applying one smooth scale to text, audio, and vision tokens, Modality-Aware Smoothing (MAS) uses a modality-specific scale in shared quantized linear layers.

Cross-Modal Compensation (CMC) keeps a text-smoothed quantized weight as the shared base and approximates the audio-specific residual with low-rank L/R factors. The residual is whitened with audio-token second-order statistics before SVD. During audio prefill, the quantized layer uses the audio scale and the audio L/R path; autoregressive text decoding defaults to the text path.

The local W4A8 implementation is fake quantization: it simulates 4-bit weight and 8-bit activation error in the computation graph rather than exporting a portable 4-bit checkpoint.

## Evaluation Protocol

The LibriSpeech loader matches each FLAC file to its transcript by utterance ID. It constructs a Qwen2.5-Omni chat message containing the audio input and an ASR prompt, applies the local processor, and decodes text greedily. Each run writes:

- `response`: raw decoded output and runtime metadata.
- `scored`: normalized reference/hypothesis text with per-utterance substitutions, insertions, and deletions.
- `summary`: corpus-level WER, runtime configuration, and artifact paths.

`qwen_asr_en` removes Qwen special tokens, applies English text normalization, lowercases, removes punctuation, and collapses whitespace on both reference and hypothesis. `jiwer.process_words` computes S/I/D, and `WER = (S + I + D) / N`, where N is the normalized reference-word count.

Two prompt protocols are retained. P0 is a user-only baseline. P1 adds a speech-recognition system instruction and requests unpunctuated English transcription. P1 is the matched comparison protocol because it is available for Dense, full-rank W4A8, and rank0p2 W4A8.

## Local Results

| Method | Protocol | CMC Rank | WER | Empty | Hit Max |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense BF16 + FA2 | P0 | - | 4.4193% | 0 | 0 |
| Dense BF16 + FA2 | P1 | - | 4.5195% | 0 | 0 |
| MAS+CMC W4A8 audio | P0 | 1.0 | 6.3235% | 2 | 4 |
| MAS+CMC W4A8 audio | P1 | 1.0 | 4.6991% | 0 | 0 |
| MAS+CMC W4A8 audio | P1 | 0.2 | 4.8447% | 0 | 0 |

Under P1, full-rank W4A8 is 0.1796 percentage points above Dense. Rank0p2 is 0.1456 points above full-rank. The P1 runs have no failed samples, empty outputs, or token-limit hits; the P0 full-rank result is not used for the main quantization comparison because it contains empty and hit-limit outputs.

## Relation to the Paper

MASQuant Table 2 reports Qwen2.5-Omni-3B on LibriSpeech `test-other` with 3.9% Dense FP16 WER and 3.6% MASQuant W4A8 WER. The local P1 Dense and full-rank W4A8 results are higher by 0.6195 and 1.0991 percentage points, respectively. The paper evidence is available locally at `EfficientAI/masquant/paper/table_2_qwen_omni.jpg` and online in the [MASQuant paper](https://arxiv.org/html/2603.04800v1).

This is a reference comparison, not a claim of configuration equivalence. The local run uses BF16 + FA2, a fixed P1 prompt, `qwen_asr_en` scoring, and a text-audio calibration cache. The paper table does not disclose the corresponding prompt, normalizer, decoding configuration, CMC rank, calibration mixture, or attention backend. The Dense baseline gap shows that the W4A8 gap cannot be assigned to CMC alone.

## Repository Layout

| Path | Contents |
| --- | --- |
| `EfficientAI/` | EfficientAI source snapshot, including MASQuant code and the paper result table image. |
| `EfficientAI/masquant/` | Local MASQuant entry points, quantization modules, Qwen2.5-Omni integration, and ASR scoring implementation. |
| `cache/` | Persisted MAS parameters, calibration dataloaders, audio white matrix, and rank-specific CMC adapters. |
| `submissions/report.md` | Chinese reproduction report with method, workflow, code mapping, results, and error analysis. |
| `submissions/result.md` | English result record and paper-reference gap analysis. |
| `submissions/experience.md` | English implementation notes covering code paths, cache lifecycle, and protocol boundaries. |
| `submissions/experiments/` | Formal response, scored, and summary artifacts. |
| `submissions/logs/` | Matching inference logs. |

## Reading the Artifacts

The formal artifact names encode model, method, dataset, subset, prompt protocol, and artifact type. For example, a P1 W4A8 full-rank result has matching response, scored, summary, and log files under `submissions/experiments/` and `submissions/logs/`.

The cache is intentionally separated by lifecycle: MAS parameters and an existing low-rank adapter are direct inference inputs; activation dataloaders and the white matrix belong to calibration or adapter construction. Dense inference does not consume MAS/CMC cache.

## References

- [MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2603.04800)
- [EfficientAI MASQuant source tree](EfficientAI/masquant/)

