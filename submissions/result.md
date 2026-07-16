# Experimental Results

## Scope And Protocol

All completed rows below use Qwen2.5-Omni-3B on the full LibriSpeech
`test-other` split (2939 utterances). The current strict runs use the P1 prompt:

- System: `You are a speech recognition model.`
- User: `Transcribe the English audio into text without any punctuation marks.`

P1 scoring uses `qwen_asr_en` and `jiwer.process_words` 4.0.0. WER is global
word error rate: `(substitutions + insertions + deletions) / reference_words`.
The MASQuant paper's table calls its dense baseline `Dense FP16`; locally
executed dense runs are labeled BF16 only when the released loader/runtime
provides that evidence.

## Retained Complete Results

| Method | Prompt | WER | S | I | D | Edit / Words | Failed | Empty | Hit max | Runtime evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense-bf16 | P0 | 5.0018% | 1929 | 335 | 388 | 2652 / 53021 | 0 | 0 | 0 | historical; dtype inferred, `librispeech_basic`, Levenshtein fallback |
| dense-bf16-fa2 | P0 | 4.4193% | 1683 | 262 | 392 | 2337 / 52882 | 0 | 0 | 0 | BF16 and FA2 runtime-observed; `qwen_asr_en` |
| dense-bf16-fa2 | P1 | 4.5195% | 1869 | 242 | 279 | 2390 / 52882 | 0 | 0 | 0 | BF16 and FA2 runtime-observed; `qwen_asr_en` |
| mas-cmc-w4a8-fullrank-audio-fa2 | P0 | 6.3235% | 2068 | 573 | 703 | 3344 / 52882 | 0 | 2 | 4 | W4A8, audio CMC rank 1.0, BF16 and FA2 runtime-observed |
| mas-cmc-w4a8-fullrank-audio-fa2 | P1 | 4.6991% | 1929 | 316 | 240 | 2485 / 52882 | 0 | 0 | 0 | W4A8, audio CMC rank 1.0, BF16 and FA2 runtime-observed |
| mas-cmc-w4a8-rank0p2-audio-fa2 | P1 | 4.8447% | 2006 | 312 | 244 | 2562 / 52882 | 0 | 0 | 0 | W4A8, audio-only CMC rank 0.2, BF16 and FA2 runtime-observed |

## P1 Comparison

Under the same P1, full2939, `qwen_asr_en` protocol:

- Dense BF16 + FA2: 4.5195% WER.
- MAS+CMC W4A8 full-rank audio: 4.6991% WER, +0.1796 percentage points versus dense.
- MAS+CMC W4A8 rank0p2 audio: 4.8447% WER, +0.3253 percentage points versus dense and +0.1456 percentage points versus full rank.

## Artifact Status

The six table rows are backed by complete response, scored, summary, and
run-log artifacts in `submissions/experiments/` and `submissions/logs/`.
There is one additional retained `dense-bf16` P1 run log without a corresponding
response/scored/summary set; it is not reported as a completed experiment.
Historical P3, sorted512, no-CMC, prompt-sweep, and diagnostic results are not
listed because their complete formal artifact sets are not retained.
