# 评估记录

本文件记录 AskDatasphere 历次 RAGAS 评估的参数、结果、优缺点分析与改进方案。

四个指标（Context Recall、Context Precision、Faithfulness、Answer Relevancy）的**计算方式**见 [README.md](README.md#评估-rag-质量)「评估 RAG 质量」章节，本文件只记录每次评估的**结果解读**。四个指标都是越接近 1 越好。

优化方向的技术方案见 [RAG_OPTIMIZATION.md](RAG_OPTIMIZATION.md)。

---

## 第一次基线（2026-10-26，50 样本）

| 项目 | 值 |
|---|---|
| 样本量 | 50（从 400 条评估集取前 50） |
| Top K | 3 |
| 生成 / judge 模型 | claude-sonnet-4-5 |
| Embedding | BAAI/bge-base-en-v1.5 |
| Chunk 大小 | 3200 字符 / 400 重叠 |

| 指标 | 分数 | 低于 0.5 的条数 |
|---|---|---|
| Context Recall | 0.710 | 14 / 50 |
| Context Precision | 0.625 | 23 / 50 |
| Faithfulness | 0.916 | 1 / 50 |
| Answer Relevancy | 0.797 | 9 / 50 |

**结论：** 首次跑通评估流程，确认瓶颈在检索侧（Recall/Precision 偏低，Faithfulness 已很高）。Top K=3 偏小，决定后续调到 5。样本量小，仅作基线参考。

---

## 第二次全量评估（2026-10-28，400 样本）

| 项目 | 值 |
|---|---|
| 样本量 | 400（全量，200 个 chunk × 2 问答对） |
| Top K | 5 |
| 生成 / judge 模型 | claude-sonnet-4-5 |
| Embedding | BAAI/bge-base-en-v1.5 |
| Chunk 大小 | 3200 字符 / 400 重叠 |

| 指标 | 均值 | 低于 0.5 的条数 | 等于 0 的条数 |
|---|---|---|---|
| Context Recall | **0.793** | 77 / 400（19%） | 77 / 400 |
| Context Precision | **0.631** | 165 / 400（41%） | 73 / 400 |
| Faithfulness | **0.909** | 13 / 400（3%） | 2 / 400 |
| Answer Relevancy | **0.796** | 74 / 400（18.5%） | 74 / 400 |

### 结果解读

**Context Recall（0.793）：** 约八成问题检索器找到了必要信息。但 77 条（19%）得分为 0——正确切片完全没被召回，Claude 无从作答，只能输出"信息不足"。

**Context Precision（0.631）：** 四个指标里最弱，165 条（41%）低于 0.5，说明近一半问题的召回结果充斥噪音 chunk，或有用的 chunk 排在靠后。这是当前系统最主要的瓶颈。

**Faithfulness（0.909）：** 四个指标里最高，仅 13 条低于 0.5。系统提示词的约束（"只基于文档片段回答，不足则说信息不足"）效果显著，Claude 基本没有凭空编造。2 条得分为 0 的 case 可能是 Claude 混入了训练数据里的 SAP 知识，值得单独排查。

**Answer Relevancy（0.796）：** 74 条得分恰好为 0（18.5%），与 Context Recall 为 0 的 77 条高度重合——根因是检索失败：检索没找回正确内容时，Claude 输出"信息不足"，这类回答反向生成的问题与原始问题毫无语义关联，导致得分为 0。不是生成模型的问题。

### 低分 case 分析（针对 77 条 recall=0）

逐条排查 recall=0 的问题，导致检索失效的术语/问题类型集中在三类：

1. **精确数值 / 规格事实（约 19 条）**：问具体数字（版本号 Pandas `2.2.2`、范围 `1-366`、线程数 `10`、预览 `20` 行），答案是埋在大段正文或表格里的一句话。dense embedding 不编码精确数字，且事实只占大 chunk 的一句话，chunk 向量被周围无关正文主导，排不进 Top 5。

2. **稀有复合术语**：`auxiliary measure`、`Calendar - Day of Year`、`Semantic Enrichment` 等全库低频术语，语义信号被整块宽泛主题冲淡。例：问"什么是 auxiliary measure"，召回的全是通用 measure 内容，偏偏没有那条定义句。

3. **旧产品名 / 变更日志元问题（4 条，评估集噪音）**：`SAP Data Warehouse Cloud`（SAP Datasphere 旧名）相关问题，如"文档档案里提到的最早版本""2021.7 把 Administration 改名成了什么"，依赖跨多个 changelog 条目检索，不适合 Top-5 向量召回。

**另发现的硬问题——chunk 超出 embedding 模型上限：** `BAAI/bge-base-en-v1.5` 的 `max_seq_length` 是 512 token，而 3200 字符英文 ≈ 800 token，超出部分被 SentenceTransformer 直接截断——每个 chunk 尾部约 1/3 内容（~1200 字符）根本没进入向量。若事实恰好落在 chunk 后半段，它压根没被编码，这不是排序问题而是信息根本没进索引。

### 改进方案（优先级排序）

1. **缩小 chunk（优先做，修硬 bug）**：chunk 降到 ~1200 字符（约 300 token，留足 512 余量），overlap 降到 ~150-200。同时解决"尾部截断丢失"和"语义稀释"。需重建 `chroma_db/`。
2. **混合检索（BM25 + 向量）**：BM25 字面匹配，对精确 token（版本号、`1-366`、低频术语）命中率高，直接补 dense 检索盲区，改善上述第 1、2 类失败。
3. **Reranker（cross-encoder）**：召回阶段用 Top K=10 拉回更多候选，再精选 Top 3-5，直接改善 Context Precision。
4. **语义切块**：按段落 / 标题边界切，而非固定字符数，进一步减少碎片化。

---

<!-- 新评估记录追加在此之上，倒序或顺序自行选择；每条包含：参数表 + 结果解读 + 低分 case 分析 + 改进方案 -->
