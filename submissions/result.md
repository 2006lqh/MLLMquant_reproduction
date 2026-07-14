# Experimental Results

## Evaluation Protocol

Model: Qwen2.5-Omni-3B. Dataset: LibriSpeech test-other. The P0 prompt is
`Transcribe the speech into English. Output only the transcription text.`
Scoring uses jiwer 4.0.0 with `librispeech_basic` normalization and global
word-level S/I/D/N aggregation, not sentence averaging. Generation uses greedy
decoding, batch size 1, and `max_new_tokens=256`.

## Completed Experiments

| Model | Method | Dataset | Subset | Prompt | Samples | WER | S | I | D | Edit | Exact | Empty | Hit max | Validity |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-Omni-3B | dense-fp16 | LibriSpeech test-other | full2939 | P0 | 2939 | 5.0018% | 1929 | 335 | 388 | 2652 | - | 0 | 0 | completed |
| Qwen2.5-Omni-3B | dense-fp16 | LibriSpeech test-other | full2939 | P3 | 2939 | 5.4337% | 1942 | 599 | 340 | 2881 | - | 0 | 0 | historical completed result |
| Qwen2.5-Omni-3B | dense-fp16-prompt-diagnostic | LibriSpeech test-other | full2939 | P3 | 2939 | 4.8188% | - | - | - | - | - | - | - | historical prefix diagnostic |
| Qwen2.5-Omni-3B | dense-fp16 | LibriSpeech test-other | sorted512 | P0 | 512 | 8.1211% | 534 | 82 | 76 | 692 | 216 | 0 | 0 | completed subset extracted from full response |
| Qwen2.5-Omni-3B | mas-cmc-w4a8-rank0p2 | LibriSpeech test-other | sorted512 | P0 | 512 | 9.8228% | 636 | 98 | 103 | 837 | 202 | 0 | 0 | completed |
| Qwen2.5-Omni-3B | text-base-unified-weight-without-cmc-diagnostic | LibriSpeech test-other | sorted512 | P0 | 512 | 134.3270% | 1919 | 8931 | 596 | 11446 | 86 | 10 | 75 | diagnostic only; not a formal MAS-only result |

## Core Results Retained

Core response, scored, summary, comparison, input, and run-log files are stored
under `experiments`, `experiments/inputs`, and `logs` using the repository-wide
naming convention.

## Historical Results Not Retained

P3 full, prefix diagnostics, prompt sweeps, sanity runs, and rank sweeps are
recorded above when their metrics were verified. Their raw artifacts were not
retained because later P0 and rank0.2 runs provide the reproducible comparison
protocol.

## Invalid or Incomplete Experiments

The zero-byte P3 sanity response and any targeted16/cross-bit outputs without a
complete response are invalid and are not retained.

Maintenance policy: update this file only after an explicit project instruction.
