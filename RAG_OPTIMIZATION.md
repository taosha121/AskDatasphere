# RAG 优化思考

本文档记录 AskDatasphere 的 RAG 质量优化方向，从"如何度量"到各条具体改进路径。部分已落地，部分为待探索方向。

项目主文档见 [README.md](README.md)。

---

## 0. 评估驱动（已落地）

优化的前提是能量化——没有度量，任何"改进"都只是感觉。本项目用 [RAGAS](https://docs.ragas.io) 建立了离线评估链路，对检索和生成分别打分。

四个核心指标：

| 指标 | 衡量什么 | 用到的字段 |
|---|---|---|
| Context Recall | 检索是否召回了回答所需的信息 | 召回切片 vs 标准答案 |
| Context Precision | 召回内容是否相关、排序是否靠前 | 召回切片 vs 标准答案 |
| Faithfulness | 答案是否忠于召回内容、有无编造 | 答案 vs 召回切片 |
| Answer Relevancy | 答案是否切题 | 答案 vs 问题 |

评估分三步：

1. **造测试集** —— `gen_eval_dataset.py` 从向量库按 deliverable 比例采样 chunk，用 Claude 为每个 chunk 生成 Q&A 对，得到 `(question, ground_truth)`。
2. **跑真实管线** —— `run_eval.py` 对每个 question 调用线上同款检索 + 生成逻辑，补上系统实际产出的 `answer` 和实际召回的 `retrieved_contexts`。
3. **打分** —— `evaluate.py` 用 RAGAS 算四个指标。judge LLM 用 Claude，embedding 用本地 bge，全程不依赖 OpenAI。

具体命令见 [README.md](README.md#评估rag-质量)。后续每一项优化（换 chunk 策略、加 reranker、混合检索）都以这套指标的变化作为是否有效的判据。

---

## 1. Chunk 划分

当前采用固定窗口切块策略，参数如下：

| 参数 | 值 | 说明 |
|---|---|---|
| Chunk 大小 | 1500 字符（约 375 token） | 单个切片的最大长度，控制在 embedding 模型 512 token 上限内 |
| Overlap | 200 字符 | 相邻切片的重叠区域，防止语义在边界处被截断 |

切块粒度的选择需要在"语义完整性"和"检索精度"之间取得平衡：切块过大，单个切片包含的主题过多，检索时容易引入噪音，而且一旦超过 embedding 模型的输入上限（`bge-base-en-v1.5` 为 512 token，约 2000 字符英文），超出部分会被 SentenceTransformer 直接截断、根本不进向量；切块过小，则上下文不足、信息碎片化。最初设为 3200 字符（约 800 token）偏大——既超了 512 上限导致尾部约 1/3 内容丢失，又让精确事实（版本号、术语、数值）在单个 768 维向量里被稀释。2026-10-28 的全量评估印证了这一点（见 [EVAL_LOG.md](EVAL_LOG.md) 对 77 条 recall=0 的低分 case 分析），据此下调到 1500 字符 / 200 重叠。

PDF 文本提取本身也是切块质量的变量：表格会被拍平成逐行的孤立单元格，页眉页脚（页码、章节标题、"PUBLIC"水印）会混入正文。这些噪音会稀释切片的语义，是 Context Precision 的潜在拖累项，值得在 ingest 阶段清洗。

## 2. Top K 与召回率优化

Top K 控制每次检索返回的候选切片数量，直接影响 RAG 的召回率与上下文质量。

**待完成的优化方向：**

- **召回率评估**：构造一批典型问题，模拟不同 Top K 设置下的检索结果，统计正确答案所在切片是否被召回，从而确定合理的 K 值范围。（评估链路已落地，见第 0 节。）

- **混合检索（Hybrid Search）**：当前使用纯向量检索，擅长语义相似但对精确词匹配较弱。混合检索将向量检索与关键词检索（通常为 BM25 算法）的结果合并，通过 RRF（Reciprocal Rank Fusion，倒数排名融合）等方式合并两者得分。对于 SAP Datasphere 这类包含大量产品术语、UI 元素名称的技术文档，混合检索的召回效果通常明显优于纯向量检索。

- **Rerank（交叉编码器精排）**：Bi-Encoder 将问题与文档分别编码为向量后计算距离，速度快但精度有限。Cross-Encoder 则将问题和候选文档拼接后一同输入模型，注意力机制可同时作用于两段文本，输出的相关性分数更为精确。典型做法是：先用向量检索粗召回 Top 20，再用 Cross-Encoder 对候选重新打分，取 Top 3 送入 LLM。精度显著高于 Bi-Encoder，代价是推理更慢，因此只用于"粗召回之后的精排"阶段。ModelScope 上可用的模型：`BAAI/bge-reranker-base`。

- **双编码器二次比对**：不引入 Cross-Encoder，而是用第二个更强的 Bi-Encoder 对初始候选重新计算相似度，过滤低质量结果。实现成本低于 Cross-Encoder，精度也不及 Cross-Encoder，适合作为延迟敏感或计算资源有限场景下的折中方案。

目前页面已支持手动调节 Top K（默认 5，最大 20），可用于人工观察不同 K 值下的检索质量。

## 3. 幻觉处理

大模型在 RAG 场景下的幻觉主要来源于两类：一是检索到的切片与问题不相关，但模型仍强行作答；二是检索切片不足以完整回答问题，模型用自身参数记忆"补全"了答案。

当前在 Prompt 中采用以下强约束策略：

- **强制仅基于检索内容回答**：System Prompt 明确禁止模型使用检索结果以外的知识，若提供的切片信息不足，要求直接声明无法回答，而非推断或补充。
- **强制引用溯源**：要求模型在每个结论后标注来源编号（如 `[1]`、`[2]`），没有来源的结论在结构上无法输出，从根本上抑制无依据的推断。
- **负面指令优于正面指令**：指令中使用"Do not add any information not explicitly present in the excerpts"，实践中负面约束对抑制幻觉的效果优于单纯的正面引导。

这套约束的效果可由 Faithfulness 指标直接量化——它衡量答案中有多少 claim 能在召回切片里找到依据。

## 4. Graph RAG

由微软于 2024 年提出，是对普通 RAG 的架构级升级。

**核心区别：** 普通 RAG 将文档切成 chunk 存入向量库，检索的是文本片段。Graph RAG 先由 LLM 从文档中抽取实体（如产品名、功能名、角色）和实体间的关系，构建知识图谱，再对图中相互关联的节点"社区"进行层次化摘要。最终检索的是这些摘要，而非原始文本片段。

**两种检索模式：**

- **Local Search**：基于实体及其近邻节点检索，适合"某个功能怎么用"这类具体问题，与普通 RAG 效果接近。
- **Global Search**：基于全图的社区摘要检索，适合"这套文档整体讲了哪些主题"、"各模块之间有什么关系"这类需要全局视野的问题——这正是普通 RAG 的盲区。

**适用场景：** 文档之间存在大量实体引用和概念关联时效果显著，例如 SAP Datasphere 各模块之间的依赖关系、权限角色与功能的映射关系。

**代价：** 构建阶段需要对每个文档 chunk 反复调用 LLM 抽取实体和关系，Token 消耗是普通 RAG ingest 阶段的数倍到数十倍，适合对查询质量要求高、构建成本可接受的场景。

### Leiden 算法：社区检测

构建完知识图谱后，图中有大量实体节点和关系边。Leiden 算法的作用是**社区检测**——找出哪些实体之间连接最密集，将它们归为同一个"社区"。对本项目的文档跑完 Leiden，可能自动归出：

- Space / 权限 / 用户角色（一组）
- DataBuilder / View / Transformation（一组）
- Catalog / Governance（一组）

Leiden 是 Louvain 算法的改进版，目标是最大化"模块度"（modularity）：社区内部连接越密、社区之间连接越稀，得分越高。GraphRAG 在 index 阶段自动调用，无需手动编写。

### Global Search：Map-Reduce 查询流程

Global Search 的本质是两阶段 LLM 调用：

```
离线 Index 阶段（一次性）：
  文档 → chunk → LLM 抽实体和关系 → 知识图谱
  → Leiden 社区检测 → 每个社区由 LLM 生成摘要
  （多层级：Level 0 最细粒度，Level N 最粗粒度）

在线 Query 阶段（Global Search）：
  Map：把问题并行发给每个社区摘要，各自独立让 LLM 回答
       "这个社区的内容和问题相关吗？有哪些要点？"
  Reduce：汇总所有非空的局部答案，再调一次 LLM 合成最终回答
```

Global Search 的 LLM 调用次数 = 社区数量 + 1，成本较高，但覆盖整个图谱，适合全局性问题。

### 依赖库与使用方式

微软官方提供 `graphrag` 包，底层依赖 `graspologic`（Leiden 算法）和 `networkx`（图结构操作）。

```bash
pip install graphrag

# 初始化配置模板
python -m graphrag index --init --root ./my_graphrag

# 将文档放入 my_graphrag/input/（支持 .txt / .csv）
# 修改 my_graphrag/settings.yaml：填入 API key、base_url、模型名

# 建索引（较慢，每个 chunk 都需调用 LLM 抽取实体）
python -m graphrag index --root ./my_graphrag

# 查询
python -m graphrag query --root ./my_graphrag --method global "各模块之间的关系是什么？"
python -m graphrag query --root ./my_graphrag --method local  "如何在 DataBuilder 里创建 View？"
```

对于本项目（约 1500 个 chunk），index 阶段的 LLM 调用量相当可观（实体抽取 + 社区摘要）。建议在普通 RAG 对全局性问题回答质量不足时再引入。

## 5. Agentic RAG

将 RAG 从单次"检索 → 生成"流程升级为由 Agent 驱动的多步骤推理循环，核心思路是让模型在检索和生成之间具备判断和决策能力。

**与普通 RAG 的对比：**

| | 普通 RAG | Agentic RAG |
|---|---|---|
| 检索次数 | 固定 1 次 | 动态，按需多次 |
| 问题处理 | 原始问题直接检索 | 可分解为子问题分别检索 |
| 结果评估 | 无 | 模型自评估检索质量 |
| 适合问题 | 单跳、事实查询 | 多跳、比较、推理类问题 |

**关键能力：**

- **问题分解**：复杂问题（如"DataBuilder 和 DataLayer 在权限模型上有什么区别？"）先被拆成若干子问题，分别检索后再汇总作答。
- **迭代检索**：第一次检索结果不足时，模型根据已有信息重写 query，发起第二次检索，直到信息充分。
- **自我纠正（CRAG）**：Corrective RAG，每次检索后由模型评估返回切片与问题的相关性得分，低于阈值则判定为"检索失败"，自动重写 query 或切换到其他检索策略（如 Web Search）再试。

**实现路径：** 可基于 LangGraph、LlamaIndex Workflows 等框架实现 Agent 循环，或像本项目一样手写循环——本质是在"检索 → 评估 → 决策 → 再检索"之间加一个由 LLM 驱动的控制流。

**适用场景：** 用户问题多样、需要跨多个文档进行推理的场景，代价是延迟和 Token 消耗都高于普通 RAG。
