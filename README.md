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
│   ├── ingest.py        # 一次性脚本：HTML + PDF → Chroma
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

从 `html_docs/` 读取所有 HTML、从 `pdfs/` 读取所有 PDF，按 3200 字符切块（400 字符重叠），写入 `chroma_db/`。重复运行是幂等的，已存在的 chunk 会自动跳过。

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

### 4. Graph RAG

由微软于 2024 年提出，是对普通 RAG 的架构级升级。

**核心区别：** 普通 RAG 将文档切成 chunk 存入向量库，检索的是文本片段。Graph RAG 先由 LLM 从文档中抽取实体（如产品名、功能名、角色）和实体间的关系，构建知识图谱，再对图中相互关联的节点"社区"进行层次化摘要。最终检索的是这些摘要，而非原始文本片段。

**两种检索模式：**

- **Local Search**：基于实体及其近邻节点检索，适合"某个功能怎么用"这类具体问题，与普通 RAG 效果接近。
- **Global Search**：基于全图的社区摘要检索，适合"这套文档整体讲了哪些主题"、"各模块之间有什么关系"这类需要全局视野的问题——这正是普通 RAG 的盲区。

**适用场景：** 文档之间存在大量实体引用和概念关联时效果显著，例如 SAP Datasphere 各模块之间的依赖关系、权限角色与功能的映射关系。

**代价：** 构建阶段需要对每个文档 chunk 反复调用 LLM 抽取实体和关系，Token 消耗是普通 RAG ingest 阶段的数倍到数十倍，适合对查询质量要求高、构建成本可接受的场景。

#### Leiden 算法：社区检测

构建完知识图谱后，图中有大量实体节点和关系边。Leiden 算法的作用是**社区检测**——找出哪些实体之间连接最密集，将它们归为同一个"社区"。对本项目的文档跑完 Leiden，可能自动归出：

- Space / 权限 / 用户角色（一组）
- DataBuilder / View / Transformation（一组）
- Catalog / Governance（一组）

Leiden 是 Louvain 算法的改进版，目标是最大化"模块度"（modularity）：社区内部连接越密、社区之间连接越稀，得分越高。GraphRAG 在 index 阶段自动调用，无需手动编写。

#### Global Search：Map-Reduce 查询流程

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

#### 依赖库与使用方式

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

对于本项目（112 个 chunk），index 阶段约需数十到上百次 LLM 调用（实体抽取 + 社区摘要）。建议在文档量增大、普通 RAG 对全局性问题回答质量不足时再引入。

### 5. Agentic RAG

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

---

## 原始文档说明

本项目的语料库来源于 [SAP Help Portal — SAP Datasphere](https://help.sap.com/docs/SAP_DATASPHERE)，均为公开文档，无需登录即可访问。

### 文档来源

**HTML 文档（`html_docs/`）**

从各章节页面逐页抓取，涵盖以下 deliverable：

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
- Video Tutorials
- What's New in SAP Datasphere

**PDF 文档（`pdfs/`，已加入 `.gitignore`）**

从各章节的"Download PDF"按钮下载，与 HTML 内容对应，但通常包含更完整的页面数量。PDF 文件体积较大，不随代码库分发，需自行从上述入口下载。

### 版本说明

SAP Datasphere 文档随产品持续迭代更新。重新运行抓取脚本所获取的内容会与本仓库当前存储的版本有所差异（新增、修改或删除的页面），属于正常现象。本仓库的 HTML 快照基于抓取时的文档版本，不保证与最新版本一致。
