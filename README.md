# AskDatasphere

基于 SAP Datasphere 官方文档构建的 RAG 问答系统，提供 Chunk 浏览和 Q&A 两个功能页面。

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | OpenUI5 1.120 (CDN，无需构建) |
| 后端 | FastAPI + uvicorn |
| 向量数据库 | Chroma (本地持久化) |
| Embedding 模型 | BAAI/bge-base-en-v1.5 (via ModelScope) |
| 检索策略 | Dense（Chroma）+ BM25（rank_bm25）→ RRF 融合 → CrossEncoder 精排 |
| Reranker | BAAI/bge-reranker-base (via ModelScope) |
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

从 `pdfs/` 读取所有 PDF，提取纯文本（标题优先取 PDF 内嵌 metadata，缺失时从首页正文的 "Document Version → ©" 之间提取），按 1500 字符切块（200 字符重叠，约 375 token，控制在 embedding 模型 512 token 上限内），写入 `chroma_db/`。重复运行是幂等的，已存在的 chunk 会自动跳过。

## 启动服务

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000`。服务启动时会预加载 embedding 模型和 reranker 模型，并从 Chroma 全量拉取所有 chunk 构建 BM25 倒排索引，第一次启动较慢（约 30-60 秒）。

## 功能说明

### Chunk 浏览页

按索引范围（Start / End）浏览 Chroma 中存储的所有文档切块，支持查看完整文本。

### Q&A 页

- 左栏：输入问题 → Ask → 显示 Claude 的回答
- 右栏：显示检索命中的 Top K 文档块 + 最终发给 Claude 的完整 Prompt
- Top K 可在页面上调节（默认 5，最大 20）
- 检索采用混合策略：Dense（Chroma 向量检索）和 BM25 各召回 20 个候选，经 RRF 融合后取 Top 10，再由 CrossEncoder reranker 精排，最终返回 Top K

### Eval 报告页

上传 `eval_scores.csv`（由 `python -m rag.evaluate` 生成）后，页面展示：

- 顶部四个指标卡片（Context Recall / Context Precision / Faithfulness / Answer Relevancy），绿色 ≥ 0.75、黄色 ≥ 0.50、红色 < 0.50
- 明细表：列出全部评估问题及每条四个指标得分，支持按任意指标排序（升序 / 降序）
- 点击任意行展开详情弹窗，可查看完整问题、系统答案、标准答案和各项得分

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

四个指标分两组：Context Recall、Context Precision 衡量检索质量，Faithfulness、Answer Relevancy 衡量生成质量。四个指标都是越接近 1 越好。各指标的计算方式如下（历次评估结果与解读见 [EVAL_LOG.md](EVAL_LOG.md)，优化方向见 [RAG_OPTIMIZATION.md](RAG_OPTIMIZATION.md)）：

| 指标 | 衡量什么 | 计算方式 |
|---|---|---|
| **Context Recall** | 检索有没有找回必要信息 | judge LLM 把 `ground_truth` 拆成若干命题，逐条检查 `retrieved_contexts` 里是否有支撑内容，统计覆盖比例 |
| **Context Precision** | 召回内容是否有用、排序是否合理 | judge LLM 判断每个召回 chunk 对回答 `ground_truth` 是否真正相关，再按排名加权——有用的 chunk 排越靠前分越高 |
| **Faithfulness** | 答案有没有编造 | judge LLM 从 `answer` 里提取所有事实性陈述，逐条检查 `retrieved_contexts` 里是否有依据，统计有依据的比例 |
| **Answer Relevancy** | 答案是否切题 | embedding 模型从 `answer` 反向生成多个假想问题，计算这些假想问题与原始 `question` 的余弦相似度均值；不依赖 `ground_truth` |

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

## 评估记录

历次评估的参数、结果、每次的优缺点分析与改进方案，统一记录在 [EVAL_LOG.md](EVAL_LOG.md)。指标的计算方式见上文「评估 RAG 质量」章节，优化方向的技术方案见 [RAG_OPTIMIZATION.md](RAG_OPTIMIZATION.md)。
