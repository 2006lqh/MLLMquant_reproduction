# Implementation Notes and Findings

## MAS and CMC in the Local Code

MASQuant addresses smoothing misalignment: text, audio, and vision can have incompatible activation ranges, so one shared smooth scale can favor the dominant modality. `quantize/svd_utils.py:trans_scales()` expands the stored text/audio/vision scales for every decoder layer. `quantize/int_linear.py` then selects a scale from the multimodal token mask. The base model remains shared; MAS changes the scaling used before quantized linear computation rather than storing three complete models.

CMC restores the modality-specific weight behavior that would otherwise be lost by a shared quantized base. In `modality_err_low_rank_decomposition()`, the text-smoothed quantized weight is the base and the audio-smoothed weight is the target. With `quant_cmc=0`, the target is not quantized a second time. The transposed difference is whitened, decomposed by SVD, and stored as L/R factors. `infer_quant.py` attaches existing audio L/R factors to the matching quantized linear layers.

The mask is available during multimodal prefill. During one-token autoregressive decode, `int_linear.py` defaults to the text mask. Audio CMC therefore addresses the audio-input portion of ASR; it does not make the later text decode path equivalent to Dense.

## Cache Lifecycle

The local cache has four distinct roles. The activation dataloader provides calibration inputs for activation statistics and MAS optimization. `mas-parameters.pth` is the direct scale input to formal quantized inference. The CMC dataloader provides audio inputs for the white matrix; the white matrix is used when building an adapter, not recomputed during formal inference. A rank-specific low-rank adapter is the direct CMC input during inference.

The retained cache names identify a `text-audio` calibration set with 128 samples. This is narrower than the upstream Qwen2.5-Omni example, which demonstrates `text-audio-vision` calibration. The local ASR task consumes only text and audio, while the model and scale structure still support vision. Dense inference consumes none of the MAS/CMC cache.

## ASR Evaluation Path

`main.py` verifies FLAC/transcript ID correspondence, builds a chat message containing an audio path and an ASR prompt, calls `AutoProcessor.apply_chat_template()`, obtains multimodal inputs through `process_mm_info()`, and invokes `generate()`. It writes response JSONL during generation. Its offline scorer applies `qwen_asr_en` to both sides, calls `jiwer.process_words`, and writes scored JSONL and a summary.

`qwen_asr_en` is a vendored Qwen English ASR normalization sequence: special-token removal, English and basic normalization, lowercase tokenization, punctuation removal, and whitespace cleanup. Using the same normalization on reference and hypothesis prevents formatting differences from being treated as transcription errors, but the chosen normalizer remains part of the evaluation protocol.

## What the Results Establish

With the matched P1 protocol, Dense BF16 + FA2 reaches 4.5195% WER, full-rank W4A8 reaches 4.6991%, and rank0p2 W4A8 reaches 4.8447%. Full-rank is closer to Dense and has 77 fewer edits than rank0p2. The P1 summaries have no failed samples, empty outputs, or token-limit hits, so these comparisons are not driven by obvious generation failure.

P0 demonstrates prompt sensitivity rather than a competing quantization result. Dense P0 is 4.4193%, while full-rank P0 is 6.3235% and contains two empty outputs plus four token-limit hits. P0 and P1 should therefore not be merged into one ranking.

## Relation to the Paper

The MASQuant paper evaluates Qwen2.5-Omni-3B on LibriSpeech `test-other` and reports 3.9% Dense FP16 WER and 3.6% MASQuant W4A8 WER in Table 2. The local P1 values are higher by 0.6195 and 1.0991 percentage points for Dense and full-rank W4A8, respectively. Because the Dense baseline is already higher, the discrepancy is not attributable only to CMC.

Confirmed protocol differences are BF16 versus the paper's FP16 label, local FA2 dispatch, a fixed P1 prompt, `qwen_asr_en` plus `jiwer` 4.0.0 scoring, and text-audio cached calibration. Table 2 does not disclose the paper's corresponding prompt, normalization, decoder settings, CMC rank, calibration mixture, or attention backend. These differences are plausible contributors, not measured causal explanations. The paper reference is `EfficientAI/masquant/paper/table_2_qwen_omni.jpg` and [MASQuant](https://arxiv.org/html/2603.04800v1).
