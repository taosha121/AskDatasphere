"""
Streamlit 聊天界面。

运行方式：
    streamlit run app.py

页面行为：
  - 用户在底部输入框输入问题
  - 先从 Chroma 向量库检索最相关的 5 个文档片段（RAG 检索阶段）
  - 将检索结果拼入 prompt，调用 Claude API 生成回答（生成阶段）
  - 回答下方提供可折叠的"Retrieved sources"，展示命中了哪些文档

对话历史保存在 st.session_state.messages，刷新页面后清空。
"""
import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from rag.retriever import retrieve, format_context

# 从 .env 文件加载 ANTHROPIC_API_KEY
load_dotenv()

CLAUDE_MODEL = "claude-sonnet-4-5"

# system prompt 限定助手只基于提供的文档片段回答，避免 Claude 凭自身知识编造
SYSTEM_PROMPT = (
    "You are a helpful assistant specializing in SAP Datasphere. "
    "Answer questions based on the provided documentation excerpts. "
    "If the excerpts do not contain sufficient information, say so honestly. "
    "Be concise and precise."
)

# ---------- 页面初始化（每次脚本重跑都会执行，Streamlit 的运行机制） ----------

st.set_page_config(page_title="SAP Datasphere Assistant", page_icon="📖")
st.title("SAP Datasphere Assistant")
st.caption("Powered by SAP Help Portal docs + Claude")

# session_state 在同一浏览器 session 内跨脚本重跑保持不变，用于存储对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 把历史消息渲染到页面（用户和助手交替显示）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 处理新输入 ----------

# st.chat_input 在用户按回车时返回输入文本，否则返回 None
if prompt := st.chat_input("Ask about SAP Datasphere…"):
    # 立即把用户消息追加到历史并渲染，给用户即时反馈
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 第一步：RAG 检索——将问题向量化，从 Chroma 找出最相关的 5 个 chunk
        with st.spinner("Searching docs…"):
            chunks = retrieve(prompt)
            context = format_context(chunks)

        # 第二步：构造发给 Claude 的消息，把检索结果作为上下文前缀
        user_msg = f"Documentation excerpts:\n\n{context}\n\n---\n\nQuestion: {prompt}"

        # Anthropic client 每次请求时创建（轻量，无连接池开销）
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        # 第三步：调用 Claude 生成回答
        with st.spinner("Generating answer…"):
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                # 只传当前轮次，不传历史——每轮都带完整文档上下文，历史轮次意义不大
                messages=[{"role": "user", "content": user_msg}],
            )
        answer = response.content[0].text
        st.markdown(answer)

        # 展示命中的文档来源，方便用户核查
        with st.expander("Retrieved sources"):
            for i, c in enumerate(chunks, 1):
                meta = c["meta"]
                st.markdown(
                    f"**[{i}]** `{meta.get('deliverable', '')}` — "
                    f"{meta.get('topic_title', '')}  \n"
                    f"`{meta.get('source_file', '')}`"
                )

    # 将助手回答追加到历史，下次脚本重跑时会渲染出来
    st.session_state.messages.append({"role": "assistant", "content": answer})
