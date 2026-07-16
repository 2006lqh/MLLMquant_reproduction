# Qwen2.5-Omni-3B MAS+CMC 量化复现报告

## 1. 摘要

本项目在 Qwen2.5-Omni-3B 上复现 MASQuant 的 Audio-Text 评测链路：以 LibriSpeech `test-other` 的 2939 条音频为输入，生成英文转写并计算全局词错误率（WER）。正式结果覆盖 Dense BF16 + FlashAttention 2（FA2）、MAS+CMC W4A8 audio full-rank 和 MAS+CMC W4A8 audio rank0p2。

主比较协议 P1 下，Dense 为 4.5195% WER，full-rank 为 4.6991%，rank0p2 为 4.8447%。论文表 2 对同一模型和 split 报告 Dense FP16 为 3.9%、MASQuant W4A8 为 3.6%。本地结果可验证量化链路和相对 rank 趋势，但尚不能视为论文数值的严格复现；差异分析见第 8 节。

## 2. 任务、协议与评分

### 2.1 评测对象

论文和本项目均使用 LibriSpeech `test-other`，论文以 WER 衡量 Audio-Text 任务。本地加载器扫描 `*.flac` 与 `*.trans.txt`，按 utterance ID 配对；正式全量运行要求得到 2939 个样本。模型只量化 Thinker，即 Qwen2.5-Omni 的 Transformer 文本解码组件；Talker 语音输出组件关闭。

Dense 与量化运行均使用 `torch.bfloat16`、实际记录为 FA2 的注意力后端。W4A8 表示当前 `QuantLinear` 的权重 4 bit、激活 8 bit fake-quant 路径，不是独立导出的 4-bit checkpoint。

### 2.2 P0 与 P1

P0 是 user-only 基线：不传入 system message，用户提示为 `Transcribe the speech into English. Output only the transcription text.`。P1 由 system message `You are a speech recognition model.` 和用户提示 `Transcribe the English audio into text without any punctuation marks.` 组成。

P1 是量化结果的主要比较协议，因为 Dense、full-rank 和 rank0p2 都在该协议下完成。P0 用于观察提示词敏感性，不能把 P0/P1 的差异解释为量化或 CMC 的收益。正式 P1 生成采用贪婪解码、单 beam、最大 256 个新 token，并显式禁用音频输出。

### 2.3 术语与评分

| 术语 | 含义 |
| --- | --- |
| `qwen_asr_en` | 固定的英文 ASR 归一化流程。它对参考转写和模型输出对称执行 Qwen 特殊 token 移除、英文文本归一化、转小写、移除标点和空白压缩。 |
| `jiwer.process_words` | `jiwer` 4.0.0 的逐词对齐函数，统计替换（S）、插入（I）和删除（D）。`WER = (S + I + D) / N`，N 为归一化参考文本词数。 |
| response / scored / summary | response 保存原始生成与运行元数据；scored 保存归一化文本和逐样本 S/I/D；summary 保存全量聚合指标。 |
| Thinker / Talker | Qwen2.5-Omni 的 Transformer 文本解码组件 / 语音输出组件。本实验只运行 Thinker。 |
| FA2 | FlashAttention 2 注意力后端；表中 FA2 表示 summary 的运行时记录确认该后端实际参与推理。 |
| fake-quant | 在计算图中模拟低位权重和激活的量化误差，不等同于导出独立的低位模型文件。 |

## 3. 方法原理

### 3.1 MAS：按模态平滑

论文指出，不同模态的激活范围差异会使单一 smooth scale 被占优模态主导，从而损害较弱模态。MAS 为 text、audio 和 vision 分别学习 scale：在线性层内，激活通道除以当前模态的 scale，对应权重通道乘以该 scale，因此浮点线性映射保持等价，而量化前的数值范围被重新分配。

`quantize/svd_utils.py` 的 `trans_scales()` 将保存的每层 scale 展开至 attention 的 q/k/v/o 投影和 MLP 的 gate/up 投影；down projection 使用单位 scale。`quantize/int_linear.py` 依据 token mask 选择 audio、vision 或 text scale。因此 MAS 不是维护三套模型，而是在共享层中执行按模态的量化输入变换。

### 3.2 CMC：共享 base 的低秩补偿

模态专属 scale 会产生不同的平滑权重，而高效推理需要共享低精度权重。CMC 以文本平滑后的量化权重作为 base，对 audio 平滑目标与该 base 的差构造残差。本地正式结果使用 `quant_cmc=0`：文本 base 被量化，audio 目标不再进行第二次量化。

`modality_err_low_rank_decomposition()` 在残差转置上求低秩 L/R。推理时，audio token 在 prefill 阶段走 audio scale 和 L/R 旁路；自回归单 token decode 没有多模态 mask，默认走 text 路径。CMC 因此主要补偿音频输入阶段，不能覆盖量化激活、共享 base 和后续文本解码带来的全部误差。

### 3.3 白化与 rank

白化矩阵由校准音频 token 的二阶统计构成。代码先在白化后的残差上做 SVD，再恢复 L/R；其目的不是仅最小化权重矩阵误差，而是提高对实际校准输入方向的重构质量。rank0p2 保留约 20% 的可用秩，rank1p0 使用完整可用秩。论文也报告白化后残差的有效秩下降，但矩阵重构更完整不必然等同于更低 WER。

## 4. 本地复现流程

1. `main.py` 读取 `test-other` 的转录和 FLAC 文件，检查音频 ID 与转录 ID 一一对应，并按 ID 排序生成样本。
2. `AutoProcessor.apply_chat_template()` 依据 P0 或 P1 构造对话；`qwen_omni_utils.process_mm_info()` 解析音频输入；随后由 Qwen2.5-Omni `generate()` 生成纯文本转写。
3. Dense 路径以 BF16 Thinker 作为参考。量化路径读取已存在的 `mas-parameters.pth`，经 `trans_scales()` 注入模态 scale，并读取指定 rank 的 audio L/R adapter。
4. 校准链路使用 `text-audio` 的 128 条缓存：activation dataloader 用于生成激活统计和 MAS 参数，CMC dataloader 用于构建 audio white matrix，white matrix 用于分解 adapter。正式推理加载既有 MAS 参数和 adapter，不重建校准统计。
5. 每条生成写入 response；离线评分对 response 重做归一化和 `jiwer` 对齐，写入 scored 与 summary。summary 同时记录样本数、失败数、空输出、达到 token 上限的输出、注意力后端和 dtype。

## 5. 代码实现边界

| 文件 | 实际职责 |
| --- | --- |
| `EfficientAI/masquant/main.py` | LibriSpeech 扫描、P0/P1 消息构造、Qwen 音频生成、JSONL 评分和 WER 汇总。 |
| `quantize/svd_utils.py` | scale 展开、音频二阶统计白化、残差 SVD 和 L/R 构建。 |
| `quantize/int_linear.py` | 依据多模态 mask 选择 scale，并在量化层执行对应的线性计算。 |
| `quantize/infer_quant.py` | 将已有的 audio L/R adapter 绑定到量化线性层。 |
| `models/LMClass.py` | 本地模型和 processor 加载，暴露 Thinker 用于生成。 |

上游公开示例使用三模态 `text-audio-vision` 校准；本地正式 ASR cache 的来源为 `text-audio`。模型结构仍保留三模态 scale 字段，但本任务没有视觉输入。Dense 不依赖 MAS/CMC cache。

## 6. 本地结果

五组可比记录均有 2939 条 response 和 scored，且 sample ID 唯一。它们都满足 `failed=0`、`edit_distance=S+I+D` 和 WER 聚合关系。

| 方法 | Prompt | W/A | CMC Rank | WER | S | I | D | N | Empty | Hit Max |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense BF16 + FA2 | P0 | BF16 | 无 | 4.4193% | 1683 | 262 | 392 | 52882 | 0 | 0 |
| Dense BF16 + FA2 | P1 | BF16 | 无 | 4.5195% | 1869 | 242 | 279 | 52882 | 0 | 0 |
| MAS+CMC W4A8 full-rank audio | P0 | 4/8 | 1.0 | 6.3235% | 2068 | 573 | 703 | 52882 | 2 | 4 |
| MAS+CMC W4A8 full-rank audio | P1 | 4/8 | 1.0 | 4.6991% | 1929 | 316 | 240 | 52882 | 0 | 0 |
| MAS+CMC W4A8 rank0p2 audio | P1 | 4/8 | 0.2 | 4.8447% | 2006 | 312 | 244 | 52882 | 0 | 0 |

P1 下，full-rank 相比 Dense 增加 0.1796 个百分点；rank0p2 相比 Dense 增加 0.3253 个百分点，相比 full-rank 增加 0.1456 个百分点。P0 full-rank 出现 2 个空输出和 4 个 hit-max，因而不用于主量化结论。

## 7. 与论文表 2 的对照

论文的实验设置明确使用 Qwen2.5-Omni-3B、LibriSpeech `test-other` 和 WER；表 2 的 Dense FP16 为 3.9%，MASQuant W4A8 为 3.6%。本地与论文的数值对照如下。论文依据为 `EfficientAI/masquant/paper/table_2_qwen_omni.jpg` 及 [MASQuant 论文表 2](https://arxiv.org/html/2603.04800v1)。

| 比较对象 | 论文 WER | 本地 P1 WER | 本地 - 论文 |
| --- | ---: | ---: | ---: |
| Dense | 3.9%（FP16） | 4.5195%（BF16 + FA2） | +0.6195 个百分点 |
| MASQuant W4A8 | 3.6% | 4.6991%（full-rank） | +1.0991 个百分点 |
| MASQuant W4A8 | 3.6% | 4.8447%（rank0p2） | +1.2447 个百分点 |

## 8. 误差分析：本地 WER 高于论文的可能原因

首先，Dense P1 已比论文 Dense 高 0.6195 个百分点，说明差距不能仅归因于 CMC。P1 的三组正式记录没有失败、空输出或 hit-max，因此 P1 的主要差距也不能简单解释为中断或截断。相反，P0 full-rank 的空输出和 hit-max 证明提示协议确实会显著影响 WER。

以下差异已被确认，但各自对 WER 的贡献尚未通过单变量消融量化：

1. **评分与提示协议未对齐。** 本地固定 P1 和 `qwen_asr_en` + `jiwer` 4.0.0；论文确认使用 WER，但表 2 与实验设置未公开这些提示词、归一化步骤和库版本。英文数字、标点、缩写或特殊 token 的处理会改变词级编辑距离。
2. **校准与 CMC 配置尚不可逐项对照。** 本地复用 `text-audio`、128 条校准 cache，并分别评测 full-rank 和 0.2 rank；论文表只报告最终 MASQuant W4A8 数值，未在表中给出相应的 rank、`quant_cmc` 或校准数据组成。上游示例默认展示三模态校准，因此本地 audio-only ASR cache 不能假定与论文完全相同。
3. **本地评测层是扩展实现。** 上游量化数学保留在 MAS/CMC 模块中，但本地 `main.py` 新增了 LibriSpeech 扫描、Qwen 音频消息构造、response JSONL 与离线 WER 评分。模型版本、processor 行为或生成封装的细小差异都可能累积为 WER 差异。

本地 P1 内部比较仍有明确结论：full-rank 比 rank0p2 少 77 次编辑，主要少 77 次 substitution；但二者均未达到 Dense。要定位相对论文的主要差距，需要在其余条件不变时依次替换精度/后端、提示与归一化、校准组成和 CMC rank。当前结果不足以指定其中任一因素为主因。

## 9. 结论

本地实现完整覆盖了 `test-other` 的 2939 条音频，保留逐条 response、逐条评分和全量 summary。MAS+CMC W4A8 在 P1 下未达到本地 Dense，但 full-rank 明显优于 rank0p2。与论文表 2 相比，本地 Dense 和 W4A8 均偏高；该偏差具有可核验的协议、精度、校准和评测实现差异，尚需受控消融后才能归因。
