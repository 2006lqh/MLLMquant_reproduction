# Experimental Results

## Scope and Evaluation Protocol

Qwen2.5-Omni-3B was evaluated on all 2,939 utterances in LibriSpeech `test-other`. The loader pairs every FLAC file with its transcript by utterance ID before generation. The retained runs use Thinker-only BF16 text generation, Talker disabled, and FlashAttention 2 recorded at runtime.

P0 is a user-only baseline. P1 adds the system instruction `You are a speech recognition model.` and requests unpunctuated English transcription; P1 is the matched protocol for the Dense and W4A8 comparison. Generation is greedy with one beam, a 256-token cap, and audio output disabled.

References and hypotheses are normalized symmetrically by `qwen_asr_en`, then `jiwer.process_words` 4.0.0 computes substitutions (S), insertions (I), and deletions (D). `WER = (S + I + D) / N`, where N is the normalized reference-word count. W4A8 denotes the local fake-quant path with logical 4-bit weights and 8-bit activations.

## Local Results

| Method | Prompt | W/A | CMC Rank | Attention | WER | S | I | D | N | Empty | Hit Max |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense BF16 + FA2 | P0 | BF16 | none | FA2 | 4.4193% | 1683 | 262 | 392 | 52882 | 0 | 0 |
| Dense BF16 + FA2 | P1 | BF16 | none | FA2 | 4.5195% | 1869 | 242 | 279 | 52882 | 0 | 0 |
| MAS+CMC W4A8 full-rank audio | P0 | 4/8 | 1.0 | FA2 | 6.3235% | 2068 | 573 | 703 | 52882 | 2 | 4 |
| MAS+CMC W4A8 full-rank audio | P1 | 4/8 | 1.0 | FA2 | 4.6991% | 1929 | 316 | 240 | 52882 | 0 | 0 |
| MAS+CMC W4A8 rank0p2 audio | P1 | 4/8 | 0.2 | FA2 | 4.8447% | 2006 | 312 | 244 | 52882 | 0 | 0 |

Each retained row has 2,939 response records and 2,939 scored records with unique IDs. All five summaries report `failed=0`, and each satisfies `edit_distance = S + I + D` and `WER = edit_distance / N`.

## Matched Local Comparison

Under P1, full-rank W4A8 is 0.1796 percentage points above Dense. Rank0p2 is 0.3253 points above Dense and 0.1456 points above full-rank. The reduced rank mainly increases substitutions: 2,006 for rank0p2 versus 1,929 for full-rank. P0 is not a quantization-only comparison: its full-rank run has two empty hypotheses and four hit-limit outputs, while the corresponding P1 run has neither.

## Paper Reference and Gap

MASQuant Table 2 reports Qwen2.5-Omni-3B on LibriSpeech `test-other`: Dense FP16 3.9% WER and MASQuant W4A8 3.6% WER. The local P1 Dense result is 0.6195 points higher; local full-rank W4A8 is 1.0991 points higher; local rank0p2 is 1.2447 points higher. The source is the local table image `EfficientAI/masquant/paper/table_2_qwen_omni.jpg` and the [paper HTML](https://arxiv.org/html/2603.04800v1).

These are reference gaps, not direct method rankings. The paper does not specify in Table 2 the prompt, normalization pipeline, decoding configuration, CMC rank, calibration mixture, or attention backend. The local run also uses BF16 + FA2 and a text-audio calibration cache. A controlled ablation is required before assigning the gap to any one factor.

## Error Evidence

P1 full-rank has 2,485 edits: 1,929 substitutions, 316 insertions, and 240 deletions. Rank0p2 has 2,562 edits: 2,006 substitutions, 312 insertions, and 244 deletions. `1688-142285-0001` changes `mister hale` to `mister howe`; `8461-281231-0034` changes `preceptory` to `presbytory`. These examples describe lexical substitutions only and do not establish a global cause.

The P1 runs have no failures, empty hypotheses, or hit-limit outputs, so the paper gap is not explained by those observable failure modes. Dense P1 already exceeds the paper Dense result, which further indicates that protocol and implementation differences precede the W4A8-specific discrepancy.

## Artifact Chain

`response` contains raw decoded text and runtime metadata. `scored` adds normalized text and per-utterance edit counts. `summary` aggregates WER and records dtype, attention backend, prompt label, quantization metadata, and output paths. This separation permits rescoring without rerunning inference.
