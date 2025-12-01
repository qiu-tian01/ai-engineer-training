# Chunking 实验结果分析

## 1. 哪些参数最显著？为什么？

| 分割器 | chunk_size | chunk_overlap | 上下文包含答案 | 回答准确 | 冗余度 |
| --- | --- | --- | --- | --- | --- |
| Sentence | 512 | 50 | 是 | 否 | 1 |
| Token | 128 | 8 | 是 | 是 | 1 |
| Sentence Window | 句级窗口 | - | 是 | 否 | 1 |
| Markdown | 结构自适应 | - | 是 | 是 | 1 |

### chunk_size
- **影响最大。** Token 切分使用 128 token 的短块，精准命中问题所需细节，因此得到“回答准确”；Sentence 切分单块长且含多余细节，LLM 容易分散注意力，被评为“否”。
- **经验**：事实问答倾向 100~200 token；总结/叙事题可放宽至 400~600 token 以保留上下文。

### chunk_overlap
- **作用是弥补边界损失。** Sentence splitter 10% 重叠（50 字符）、Token splitter 6% 重叠（8 token），都足以覆盖跨块句子，所以冗余度评分为 1。
- **调参策略**：先估算 chunk 内平均句子数，再设置 10%~20% 的 overlap，兼顾连续性与索引体积。

### splitter 类型
- **决定语义边界与结构。** Markdown splitter 保留标题、列表结构，回答准确；Sentence Window 在检索时聚焦单句但窗口聚合阶段仍返回多个相似段落，若评估 prompt 偏字面匹配，易被判“否”。
- **实践建议**：先按文档结构选择分割器，再微调 chunk_size/overlap，效果优于盲目堆参。

## 2. chunk_overlap 过大或过小的利弊

- **过小（≈0）**：索引最轻、检索最快，但关键信息跨块时直接丢失。
- **合理区间（10%~20%）**：如本实验，既覆盖边界又保持冗余度 1。
- **过大（≥30%）**：保证语义连续却显著增加嵌入/存储成本，检索会召回大量重复块，影响 rerank 与生成效率（`auto_redundancy_score` 将升至 4~5）。

## 3. 精确检索 vs. 上下文丰富性的权衡

1. **分层 chunking**：以 Token splitter 构建“子块”提升召回，再关联更大的“父块”供生成引用（ParentDocumentRetriever 思路）。
2. **Sentence Window**：句级检索 + 窗口式上下文；若配合更精细的答案判定 prompt，可以兼顾准确与背景信息。
3. **融合检索**：同时运行 Token/Markdown 索引，用 RRF 等算法融合，最后挑选冗余度低的 top-k。
4. **任务自适应调参**：事实问答→小块+小 overlap；复杂推理→Markdown/Sentence Window，并适度扩大 chunk_size。

## LlamaIndex 常用切分方法对比

- **SentenceSplitter**
  - *优点*：遵循自然句界，语义连贯，适合叙事与说明文。
  - *缺点*：长句会让块过长，含噪多，回答易跑题。
  - *场景*：新闻、长文说明、希望保持段落语义的 QA。

- **TokenTextSplitter**
  - *优点*：长度可控，便于对齐向量库限制或保证精确匹配。
  - *缺点*：可能打断句子，需依赖 overlap 补救。
  - *场景*：事实问答、多语言混排、需要严格控制 token 的场合。

- **SentenceWindowNodeParser**
  - *优点*：检索粒度细、生成上下文足，是兼顾精确度与语境的折中方案。
  - *缺点*：实现更复杂，窗口过大仍会引入冗余，需要配合 `MetadataReplacementPostProcessor`。
  - *场景*：法规条款、细节查证等需要句级定位的任务。

- **MarkdownNodeParser**
  - *优点*：理解标题/列表/代码块，能保留文章层级结构。
  - *缺点*：依赖 Markdown 规范，纯文本时优势有限。
  - *场景*：技术文档、API 手册、结构化报告。

以上总结依据 `main.py` 的自动评估结果，详见 `chunking_research/outputs/splitter_eval.csv`。***