# AskDatasphere Roadmap

## Phase 1: RAG 优化与评估

### 1.1 评估集构建
- 用 Claude 对每个 chunk 批量生成 2-3 个问题，构建带 ground truth 的 Q&A 数据集
- 目标：300-500 条标注数据

### 1.2 评估框架接入
- 接入 RAGAS，跑基础指标：faithfulness、answer relevancy、context precision、context recall
- 建立 baseline 数字

### 1.3 Chunk 策略对比
- 当前：固定窗口（3200 chars，400 overlap）
- 对比：语义切块（按段落/标题边界切）
- 用评估集量化两种策略的差异

### 1.4 Reranker
- 在现有 embedding 检索后加 cross-encoder 重排（如 `BAAI/bge-reranker-base`）
- 对比加 reranker 前后的 context precision

---

## Phase 2: 知识图谱 / Graph RAG

### 2.1 实体关系抽取
- 从 SAP Datasphere 文档中抽取核心实体：Space、Dataset、View、Pipeline、User、Permission、Connection 等
- 用 Claude 或 spaCy 抽取三元组

### 2.2 知识图谱构建
- 工具：Neo4j（本地 Docker）或 NetworkX（轻量）
- 存储抽取出的三元组，建立实体关系图

### 2.3 Graph RAG 实现
- 查询时先在图上做实体识别和关系遍历，再结合向量检索
- 重点测试多跳问题（如"Space 的管理员可以做哪些操作"）

### 2.4 效果对比
- 用 Phase 1 的评估集，对比 naive RAG vs Graph RAG 的指标差异
- 重点关注多跳问题上的提升

---

## 当前状态

| 项目 | 状态 |
|---|---|
| HTML 文档 ingest | 完成（112 chunks） |
| PDF 文档 ingest | 完成（1518 chunks，共 1630） |
| 评估集 | 待构建 |
| RAGAS 接入 | 待做 |
| 知识图谱 | 待做 |
