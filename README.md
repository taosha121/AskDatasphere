# AskDatasphere

基于 SAP Datasphere 官方文档构建的 RAG 问答系统，提供 Chunk 浏览和 Q&A 两个功能页面。

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | OpenUI5 1.120 (CDN，无需构建) |
| 后端 | FastAPI + uvicorn |
| 向量数据库 | Chroma (本地持久化) |
| Embedding 模型 | BAAI/bge-base-en-v1.5 (via ModelScope) |
| 生成模型 | Anthropic Claude (claude-sonnet-4-5) |
| 评估 | RAGAS (Claude 做 judge，bge 做 embedding) |

## 目录结构

```
AskDatasphere/
├── backend/
│   └── main.py          # FastAPI：/api/chunks、/api/ask、静态文件托管
├── frontend/
│   ├── index.html       # UI5 CDN bootstrap
│   ├── manifest.json    # UI5 路由配置
│   ├── Component.js     # UIComponent 根组件
│   ├── view/
│   │   ├── Shell.view.xml    # 根视图，持有 NavContainer
│   │   ├── Chunks.view.xml   # Chunk 浏览页
│   │   └── Chat.view.xml     # Q&A 页
│   └── controller/
│       ├── Shell.controller.js
│       ├── Chunks.controller.js
│       └── Chat.controller.js
├── rag/
│   ├── ingest.py                  # 一次性脚本：PDF → Chroma
│   ├── gen_eval_dataset.py        # 采样 chunk，用 Claude 生成评估问答对
│   ├── run_eval.py                # 对评估问题跑真实 RAG，补全答案与召回
│   ├── evaluate.py                # 用 RAGAS 对上一步结果打分
│   └── __init__.py
├── pdfs/                # SAP Datasphere 官方 PDF（已加入 .gitignore，需自行下载）
├── chroma_db/           # Chroma 向量库（运行 ingest 后生成）
├── RAG_OPTIMIZATION.md  # RAG 质量优化思考（评估、chunk、reranker、Graph/Agentic RAG）
└── pyproject.toml
```

## 环境配置

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # 核心运行依赖
pip install -e ".[eval]"    # 另需评估时（RAGAS + langchain）
```

`.env` 文件（放在项目根目录）：

```
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_BASE_URL=http://localhost:6655/anthropic   # 公司内网代理地址
```

## 构建向量库

只需运行一次：

```bash
python -m rag.ingest
```

从 `pdfs/` 读取所有 PDF，提取纯文本（标题优先取 PDF 内嵌 metadata，缺失时从首页正文的 "Document Version → ©" 之间提取），按 3200 字符切块（400 字符重叠），写入 `chroma_db/`。重复运行是幂等的，已存在的 chunk 会自动跳过。

## 启动服务

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000`。服务启动时会预加载 embedding 模型，第一次启动较慢（约 10-30 秒）。

## 功能说明

### Chunk 浏览页

按索引范围（Start / End）浏览 Chroma 中存储的所有文档切块，支持查看完整文本。

### Q&A 页

- 左栏：输入问题 → Ask → 显示 Claude 的回答
- 右栏：显示检索命中的 Top K 文档块 + 最终发给 Claude 的完整 Prompt
- Top K 可在页面上调节（默认 5，最大 20）

## 评估 RAG 质量

用 RAGAS 对检索和生成分别打分（judge 用 Claude，embedding 用本地 bge，不依赖 OpenAI）。三步：

```bash
# 1. 造测试集：采样 chunk，用 Claude 生成问答对 → eval_dataset.jsonl
python -m rag.gen_eval_dataset --sample 200

# 2. 跑真实 RAG：补上系统实际答案与召回 → eval_runs.jsonl
python -m rag.run_eval --limit 50          # 先小跑；去掉 --limit 为全量

# 3. 打分：输出四指标 → eval_scores.csv（每条明细）+ eval_summary.json（聚合）
python -m rag.evaluate --limit 50          # --limit 要与上一步一致
```

输出两个文件：`eval_scores.csv` 每行一条问题，含召回切片、系统答案、标准答案和该条四个指标分数，用于定位低分 case；`eval_summary.json` 是本次跑的参数（样本数、judge/embedding 模型、时间）加四指标聚合均值。

四个指标：Context Recall、Context Precision（检索质量）、Faithfulness、Answer Relevancy（生成质量）。各指标含义及后续优化方向见 [RAG_OPTIMIZATION.md](RAG_OPTIMIZATION.md)。

> 评估依赖较重（RAGAS + langchain），需先 `pip install -e ".[eval]"`。langchain 1.x 与 RAGAS 0.2.x 不兼容，相关版本已在 pyproject 的 `eval` extra 中钉定。

## RAG 优化思考

Chunk 划分、Top K 与混合检索、Rerank、幻觉抑制、Graph RAG、Agentic RAG 等优化方向的分析，独立整理在 [RAG_OPTIMIZATION.md](RAG_OPTIMIZATION.md)。

---

## 原始文档说明

本项目的语料库来源于 [SAP Help Portal — SAP Datasphere](https://help.sap.com/docs/SAP_DATASPHERE)，均为公开文档，无需登录即可访问。

### 文档来源

语料全部来自各章节"Download PDF"下载的官方 PDF（`pdfs/`，已加入 `.gitignore`，不随代码库分发，需自行下载）。ingest 时从每个 PDF 提取真实标题作为 deliverable，当前涵盖：

- Acquiring, Preparing, and Modeling Data with SAP Datasphere
- Administering SAP Datasphere
- Accessing SAP Datasphere via the Command Line
- Consuming Data Exposed by SAP Datasphere
- Data Sharing Cockpit - Data Providers Guide
- Feature Scope Description - SAP Datasphere
- Getting Started with SAP Datasphere
- Getting Started With SAP Datasphere, SAP BW Bridge
- Governing and Publishing Data in the Catalog
- Integrating Data and Managing Spaces in SAP Datasphere
- SAP Datasphere Business Content
- SAP Datasphere Security Recommendations
- Security Guide
- What's New in SAP Datasphere

### 版本说明

SAP Datasphere 文档随产品持续迭代更新。重新下载所获取的 PDF 会与本仓库当前使用的版本有所差异（新增、修改或删除的页面），属于正常现象。本仓库的 chunk 基于下载时的文档版本，不保证与最新版本一致。

---

## 评估基线记录

### 第一次基线（2026-10-26，50 样本）

| 项目 | 值 |
|---|---|
| 样本量 | 50（从 400 条评估集取前 50） |
| Top K | 3 |
| 生成 / judge 模型 | claude-sonnet-4-5 |
| Embedding | BAAI/bge-base-en-v1.5 |

| 指标 | 分数 | 低于 0.5 的条数 |
|---|---|---|
| Context Recall | 0.710 | 14 / 50 |
| Context Precision | 0.625 | 23 / 50 |
| Faithfulness | 0.916 | 1 / 50 |
| Answer Relevancy | 0.797 | 9 / 50 |

---

### 第二次全量评估（2026-10-28，400 样本）

| 项目 | 值 |
|---|---|
| 样本量 | 400（全量，200 个 chunk × 2 问答对） |
| Top K | 5 |
| 生成 / judge 模型 | claude-sonnet-4-5 |
| Embedding | BAAI/bge-base-en-v1.5 |

| 指标 | 均值 | 低于 0.5 的条数 | 等于 0 的条数 |
|---|---|---|---|
| Context Recall | **0.793** | 77 / 400（19%） | 77 / 400 |
| Context Precision | **0.631** | 165 / 400（41%） | 73 / 400 |
| Faithfulness | **0.909** | 13 / 400（3%） | 2 / 400 |
| Answer Relevancy | **0.796** | 74 / 400（18.5%） | 74 / 400 |

---

### 四个指标详解

#### Context Recall（0.793）——检索有没有找回必要信息

**计算方式：** judge LLM 把 `ground_truth` 拆成若干命题，逐条检查 `retrieved_contexts` 里是否有支撑内容，统计覆盖比例。

**本次结果解读：** 整体 0.793，说明约八成的问题检索器找到了必要信息。但有 77 条（19%）得分为 0——这些问题的正确切片完全没被召回，Claude 无从作答，只能输出"信息不足"。

**改进方向：**
- 引入混合检索（BM25 + 向量）：纯向量检索对精确术语（如产品名称、API 名）匹配较弱，BM25 能弥补
- Query expansion / HyDE：先让模型生成假想答案再检索，提升语义覆盖

---

#### Context Precision（0.631）——召回内容是否有用、排序是否合理

**计算方式：** judge LLM 判断每个召回 chunk 对回答 `ground_truth` 是否真正相关，再按排名加权——有用的 chunk 排越靠前分越高。

**本次结果解读：** 0.631 是四个指标里最弱的，且 165 条（41%）低于 0.5，说明近一半问题的召回结果里充斥着噪音 chunk，或者有用的 chunk 排在靠后位置。这是当前系统最主要的瓶颈。

**改进方向：**
- 加 Reranker（cross-encoder）：召回阶段用 Top K=10 拉回更多候选，再用 reranker 精选 Top 3–5 送给 Claude，能大幅提升 precision
- 混合检索同样有帮助（减少无关切片混入）

---

#### Faithfulness（0.909）——答案有没有编造

**计算方式：** judge LLM 从 `answer` 里提取所有事实性陈述，逐条检查 `retrieved_contexts` 里是否有依据，统计有依据的比例。

**本次结果解读：** 0.909 是四个指标里最高的，仅 13 条低于 0.5。系统提示词的约束（"只基于文档片段回答，不足则说信息不足"）效果显著，Claude 基本没有凭空编造内容。2 条得分为 0 的 case 值得单独排查，可能是 Claude 回答时混入了训练数据里的 SAP 知识而非召回内容。

**改进方向：** 当前已足够好，优先级低。如需进一步提升，可在 Prompt 里加更明确的引用要求（已有 `[1] [2]` 引用，可结合验证逻辑）。

---

#### Answer Relevancy（0.796）——答案是否切题

**计算方式：** embedding 模型从 `answer` 反向生成多个假想问题，计算这些假想问题与原始 `question` 的余弦相似度均值。此指标不依赖 `ground_truth`，只看答案和问题语义是否匹配。

**本次结果解读：** 74 条得分恰好为 0（占 18.5%），与 Context Recall 为 0 的 77 条高度重合——这不是巧合。当检索器完全没找回正确内容时，Claude 输出"信息不足，无法回答"，这类回答的反向生成问题和原始问题毫无语义关联，导致 answer_relevancy = 0。所以这 74 条低分的根因是检索失败，而非生成模型的问题。

**改进方向：** 与 Context Recall 相同——提升检索覆盖率，答案质量会随之改善，answer_relevancy 也会跟着上来。

---

### 综合结论与优先改进方向

瓶颈在检索，不在生成。Faithfulness 0.909 证明 Prompt 约束有效，Claude 忠于召回内容；但 Context Precision 0.631 / Context Recall 0.793 都有明显提升空间，19% 的问题检索完全失败，41% 的问题召回噪音过多。

**优先级排序：**

1. **Reranker（最高优先级）**：在当前架构上加 cross-encoder reranker，改动最小、收益最大，直接改善 Context Precision
2. **混合检索（BM25 + 向量）**：对精确术语类问题提升 Context Recall，尤其是产品功能名、API 名等
3. **语义切块**：当前按固定字符数切块，改成按段落 / 标题边界切可减少切块内容碎片化
4. **Top K 动态调整**：根据问题复杂度动态决定召回数量，而非固定值

详细技术方案见 [RAG_OPTIMIZATION.md](RAG_OPTIMIZATION.md)。
