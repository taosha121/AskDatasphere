# SAP Datasphere RAG Playground

基于 SAP Datasphere 官方文档构建的 RAG 问答系统，提供 Chunk 浏览和 Q&A 两个功能页面。

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | OpenUI5 1.120 (CDN，无需构建) |
| 后端 | FastAPI + uvicorn |
| 向量数据库 | Chroma (本地持久化) |
| Embedding 模型 | BAAI/bge-base-en-v1.5 (via ModelScope) |
| 生成模型 | Anthropic Claude (claude-sonnet-4-5) |

## 目录结构

```
DS-doc/
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
│   ├── ingest.py        # 一次性脚本：HTML → Chroma
│   └── __init__.py
├── html_docs/           # SAP Help Portal 原始 HTML 文档
├── chroma_db/           # Chroma 向量库（运行 ingest 后生成）
├── app.py               # 旧 Streamlit 界面（已废弃，保留备用）
└── pyproject.toml
```

## 环境配置

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
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

从 `html_docs/` 读取所有 HTML，按 3200 字符切块（400 字符重叠），写入 `chroma_db/`。目前共 112 条记录。

## 启动服务

```bash
.venv/bin/uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000`。服务启动时会预加载 embedding 模型，第一次启动较慢（约 10-30 秒）。

## 功能说明

### Chunk 浏览页

按索引范围（Start / End）浏览 Chroma 中存储的所有文档切块，支持查看完整文本。

### Q&A 页

- 左栏：输入问题 → Ask → 显示 Claude 的回答
- 右栏：显示检索命中的 Top K 文档块 + 最终发给 Claude 的完整 Prompt
- Top K 可在页面上调节（默认 3，最大 20）

## RAG 优化思考

### 1. Chunk 划分

当前采用固定窗口切块策略，参数如下：

| 参数 | 值 | 说明 |
|---|---|---|
| Chunk 大小 | 3200 字符 | 单个切片的最大长度 |
| Overlap | 400 字符 | 相邻切片的重叠区域，防止语义在边界处被截断 |

切块粒度的选择需要在"语义完整性"和"检索精度"之间取得平衡：切块过大，单个切片包含的主题过多，检索时容易引入噪音；切块过小，单个切片上下文不足，回答时信息碎片化。当前参数是针对 SAP Help Portal HTML 文档的初始设定，后续可结合实际召回效果调整。

### 2. Top K 与召回率优化

Top K 控制每次检索返回的候选切片数量，直接影响 RAG 的召回率与上下文质量。

**待完成的优化方向：**

- **召回率评估**：构造一批典型问题，模拟不同 Top K 设置下的检索结果，统计正确答案所在切片是否被召回，从而确定合理的 K 值范围。

- **混合检索（Hybrid Search）**：当前使用纯向量检索，擅长语义相似但对精确词匹配较弱。混合检索将向量检索与关键词检索（通常为 BM25 算法）的结果合并，通过 RRF（Reciprocal Rank Fusion，倒数排名融合）等方式合并两者得分。对于 SAP Datasphere 这类包含大量产品术语、UI 元素名称的技术文档，混合检索的召回效果通常明显优于纯向量检索。

- **Rerank（交叉编码器精排）**：Bi-Encoder 将问题与文档分别编码为向量后计算距离，速度快但精度有限。Cross-Encoder 则将问题和候选文档拼接后一同输入模型，注意力机制可同时作用于两段文本，输出的相关性分数更为精确。典型做法是：先用向量检索粗召回 Top 20，再用 Cross-Encoder 对候选重新打分，取 Top 3 送入 LLM。精度显著高于 Bi-Encoder，代价是推理更慢，因此只用于"粗召回之后的精排"阶段。ModelScope 上可用的模型：`BAAI/bge-reranker-base`。

- **双编码器二次比对**：不引入 Cross-Encoder，而是用第二个更强的 Bi-Encoder 对初始候选重新计算相似度，过滤低质量结果。实现成本低于 Cross-Encoder，精度也不及 Cross-Encoder，适合作为延迟敏感或计算资源有限场景下的折中方案。

目前页面已支持手动调节 Top K（默认 3，最大 20），可用于人工观察不同 K 值下的检索质量。

### 3. 幻觉处理

大模型在 RAG 场景下的幻觉主要来源于两类：一是检索到的切片与问题不相关，但模型仍强行作答；二是检索切片不足以完整回答问题，模型用自身参数记忆"补全"了答案。

当前在 Prompt 中采用以下强约束策略：

- **强制仅基于检索内容回答**：System Prompt 明确禁止模型使用检索结果以外的知识，若提供的切片信息不足，要求直接声明无法回答，而非推断或补充。
- **强制引用溯源**：要求模型在每个结论后标注来源编号（如 `[1]`、`[2]`），没有来源的结论在结构上无法输出，从根本上抑制无依据的推断。
- **负面指令优于正面指令**：指令中使用"Do not add any information not explicitly present in the excerpts"，实践中负面约束对抑制幻觉的效果优于单纯的正面引导。

---

## 原始文档说明

从 SAP Help Portal 下载的 HTML 文档，涵盖 17 个 deliverable，包括：

- Acquiring, Preparing, and Modeling Data with SAP Datasphere
- Administering SAP Datasphere
- Getting Started with SAP Datasphere
- Integrating Data and Managing Spaces
- Security Guide / Security Recommendations
- 等（详见 `html_docs/` 目录）
