# RAG 优化思考

本文档记录 AskDatasphere 的 RAG 质量优化方向，从已落地的改进到待探索的路径。评估结果见 [EVAL_LOG.md](EVAL_LOG.md)，项目主文档见 [README.md](README.md)。

---

## 当前状态

| 优化项 | 状态 | 效果 |
|---|---|---|
| RAGAS 评估链路 | 已落地 | 建立量化基线，驱动后续决策 |
| Chunk 缩小（1500 / 200） | 已落地 | 修复 512 token 截断硬 bug，减少语义稀释 |
| BM25 混合检索 + RRF 融合 | 已落地 | Context Recall 0.793 → 0.858（+8.2%） |
| CrossEncoder Reranker | 已落地 | Context Precision 0.631 → 0.793（+25.7%） |
| 语义切块（按段落/标题边界） | 待做 | 预期改善细粒度 UI 行为类问题的召回 |
| PDF 噪音清洗 | 待做 | 预期改善 Context Precision |
| Agentic RAG（迭代检索） | 待做 | 预期改善约束/否定事实类问题 |
| Graph RAG | 待做 | 预期改善多跳/全局关系类问题 |

---

## 已落地优化

### 评估驱动

优化的前提是能量化。本项目用 [RAGAS](https://docs.ragas.io) 建立了离线评估链路，对检索和生成分别打分，四个核心指标：Context Recall、Context Precision、Faithfulness、Answer Relevancy。三步流程：

1. `gen_eval_dataset.py` 从向量库按 deliverable 比例采样 chunk，用 Claude 生成 Q&A 对
2. `run_eval.py` 对每个 question 调用线上同款检索 + 生成逻辑，补上系统实际产出
3. `evaluate.py` 用 RAGAS 算四个指标，judge 用 Claude，embedding 用本地 bge

### Chunk 策略（1500 / 200）

`BAAI/bge-base-en-v1.5` 的 `max_seq_length` 是 512 token，3200 字符英文 ≈ 800 token，SentenceTransformer 直接截断超出部分——每个 chunk 尾部约 1/3 内容根本没进向量。下调到 1500 字符（约 375 token）同时修复了两个问题：截断 bug 和语义稀释（大 chunk 把精确事实淹没在周围宽泛正文里）。

### 混合检索（BM25 + Dense → RRF）

纯向量检索对精确 token（版本号、数值范围、低频术语）命中率弱，BM25 字面匹配正好互补。实现方式：启动时从 Chroma 全量拉取文档，在内存里建 `BM25Okapi` 倒排索引（`rank_bm25` 库），两路各召回 20 个候选，经 RRF（k=60）融合取 Top 10。

### CrossEncoder Reranker

Bi-Encoder 将问题与文档分别编码为向量后计算距离，速度快但精度有限——向量相似度无法捕捉细粒度匹配关系。CrossEncoder（`BAAI/bge-reranker-base`）将问题和候选文档拼接后一同输入，注意力机制同时作用于两段文本，输出更精确的相关性分数。在 RRF 后的 Top 10 候选上做精排，Context Precision 提升 25.7%。

---

## 待探索优化方向

### 1. 语义切块

当前固定字符切块不考虑自然边界，一段完整的操作说明可能被切到两个 chunk 中间。按段落或标题边界切，保证每个 chunk 是语义自洽的单元，预期减少第三次评估中"细粒度 UI 行为 / 过程描述"类的 recall=0 case（占剩余 51 条的约 65%）。

实现思路：用 `fitz`（PyMuPDF）提取 PDF 的 block 级别结构（每个 block 通常对应一个段落或标题），按 block 边界合并，超出 1500 字符再二次切分。

### 2. PDF 噪音清洗

PDF 文本提取会把表格拍平成逐行的孤立单元格，把页眉页脚（页码、章节标题、"PUBLIC"水印）混入正文。这些噪音稀释切片的语义，是 Context Precision 的潜在拖累项。可在 ingest 阶段过滤：行长度 < 20 字符的行、纯数字行（页码）、包含特定关键字的行（"PUBLIC"、"Document Version"）。

### 3. Agentic RAG（迭代检索）

当前系统是单次"检索 → 生成"，约束/否定事实类问题（"Can I do X?" → "No"）一次召回经常找不到那一句限制性说明。Agentic RAG 在检索和生成之间加入由 LLM 驱动的评估-决策循环：

| | 普通 RAG | Agentic RAG |
|---|---|---|
| 检索次数 | 固定 1 次 | 动态，按需多次 |
| 问题处理 | 原始问题直接检索 | 可分解为子问题分别检索 |
| 结果评估 | 无 | 模型自评估检索质量 |
| 适合问题 | 单跳、事实查询 | 多跳、比较、约束类问题 |

核心能力：**迭代检索**（第一次不足时重写 query 再检索）和 **CRAG**（Corrective RAG，检索后由模型评估相关性，低于阈值则重写 query 或切换策略）。可基于 LangGraph、LlamaIndex Workflows 实现，或直接手写循环。代价是延迟和 Token 消耗高于普通 RAG。

### 4. Graph RAG

由微软于 2024 年提出，架构级升级：先由 LLM 从文档中抽取实体和关系构建知识图谱，再对图中社区进行层次化摘要，检索的是摘要而非原始文本片段。

**两种检索模式：**

- **Local Search**：基于实体及其近邻节点检索，适合"某个功能怎么用"类具体问题，与普通 RAG 效果接近
- **Global Search**：基于全图的社区摘要检索，适合"各模块之间有什么关系"类需要全局视野的问题——这正是普通 RAG 的盲区

**Leiden 算法：社区检测**

构建完知识图谱后，Leiden 算法找出哪些实体之间连接最密集，归为同一"社区"。对本项目的文档跑完 Leiden，可能自动归出：Space / 权限 / 用户角色、DataBuilder / View / Transformation、Catalog / Governance 等社区。

**Global Search 查询流程（Map-Reduce）：**

```
离线 Index 阶段（一次性）：
  文档 → chunk → LLM 抽实体和关系 → 知识图谱
  → Leiden 社区检测 → 每个社区由 LLM 生成摘要

在线 Query 阶段（Global Search）：
  Map：把问题并行发给每个社区摘要，各自独立让 LLM 回答相关要点
  Reduce：汇总所有非空局部答案，再调一次 LLM 合成最终回答
```

Global Search 的 LLM 调用次数 = 社区数量 + 1，成本较高，适合对全局性问题质量要求高的场景。

**依赖：**

```bash
pip install graphrag

python -m graphrag index --init --root ./my_graphrag
# 将文档放入 my_graphrag/input/，修改 settings.yaml 填入 API key 和模型名
python -m graphrag index --root ./my_graphrag

python -m graphrag query --root ./my_graphrag --method global "各模块之间的关系是什么？"
python -m graphrag query --root ./my_graphrag --method local  "如何在 DataBuilder 里创建 View？"
```

对于本项目（约 3000+ chunk），index 阶段的 LLM 调用量相当可观。建议在普通 RAG 对全局性问题回答质量不足时再引入。
