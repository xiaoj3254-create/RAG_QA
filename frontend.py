"""Streamlit 前端：纯 HTTP 客户端，通过 requests 调用 FastAPI 接口。

⚠️ 前后端分离：本模块零导入 main/RAG 内部类，全部通过 REST 接口与后端交互。

页面结构：
- 侧边栏：API 连通状态、文档上传/删除、RAG 参数面板、会话新建/切换/删除；
- 主区域：当前会话提示、对话气泡渲染；助手回答下方两个折叠面板——
  "来源与置信度"（来源片段 + 置信度分数）和"调试信息"
  （改写后 Query、Multi-Query 子查询、Rerank 打分明细）；底部聊天输入框。

状态管理：会话历史从后端 SQLite 拉取（含助手消息的 meta 面板数据），
切换会话时自动加载；删除会话后清空本地 session_state（含选择器 widget state）。

会话命名：新建会话可自定义名称，默认名取「会话 N」（N 为现有同名会话最大编号 + 1），
用户手填重名时自动补 `(2)`/`(3)` 后缀，从源头避免下拉框出现多个同名项。
"""
import os

import requests
import streamlit as st

# FastAPI 后端地址（可通过环境变量或侧边栏覆盖）
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG 问答系统", layout="wide")

BACKEND = {"api_base": API_BASE}


def api(method: str, path: str, **kwargs):
    """统一 HTTP 请求封装，返回 (status_code, json_dict)。异常返回 (None, {error})"""
    url = f"{BACKEND['api_base']}{path}"
    try:
        r = requests.request(method, url, timeout=120, **kwargs)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"text": r.text}
    except Exception as e:
        return None, {"error": str(e)}


def load_history(session_id: str):
    """从后端拉取会话历史（持久化在 SQLite）。"""
    if not session_id:
        return []
    code, resp = api("GET", f"/api/session/{session_id}/history")
    if code == 200:
        return resp.get("messages", [])
    return []


# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("📚 RAG 控制台")

    # 1. API 连通状态
    st.subheader("🔌 服务状态")
    api_base_input = st.text_input("API 地址", value=API_BASE)
    if api_base_input != BACKEND["api_base"]:
        BACKEND["api_base"] = api_base_input
        st.rerun()
    code, health = api("GET", "/health")
    if code == 200:
        st.session_state.setdefault("connected", True)
        st.success(f"✅ API 已连接（{BACKEND['api_base']}）")
    elif code is None:
        st.session_state["connected"] = False
        st.error(f"⚠️ API 未连接：{health.get('error', '网络异常')}")
        st.stop()
    else:
        st.error(f"⚠️ API 异常：HTTP {code}")
        st.stop()

    # 2. 文件上传
    st.subheader("📄 文档上传")
    files = st.file_uploader(
        "支持 .pdf / .txt / .md / .json / .csv，可多选",
        type=["pdf", "txt", "md", "json", "csv"],
        accept_multiple_files=True,
    )
    if st.button("上传并索引", disabled=not files):
        for f in files or []:
            code, resp = api(
                "POST", "/api/doc/upload",
                files={"file": (f.name, f.getvalue(), f.type)},
            )
            if code == 200:
                st.success(f"✅ {f.name}：{resp.get('chunk_count', 0)} 个文本块")
            else:
                st.error(f"❌ {f.name}：{resp.get('detail', resp.get('error', code))}")
    code, resp = api("GET", "/api/doc/list")
    docs = resp.get("docs", [])
    if docs:
        for d in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"- {d['file_name']}（chunks={d['chunk_count']}）")
            with col2:
                if st.button("🗑️", key=f"del_doc_{d['doc_id']}", help="删除该文档"):
                    dc, dr = api("DELETE", f"/api/doc/{d['doc_id']}")
                    if dc == 200:
                        st.success(f"已删除 {d['file_name']}")
                    else:
                        st.error(f"删除失败：{dr.get('detail', dr.get('error', dc))}")
                    st.rerun()
    else:
        st.info("暂无已上传文档")

    # 3. RAG 参数面板
    # 注意：chunk_size/chunk_overlap 只影响之后上传文档的分块方式，不会重切已入库的向量；
    # 且以下参数每次提问都会随请求发送并临时覆盖后端 .env 默认值。
    st.subheader("⚙️ RAG 参数")
    chunk_size = st.slider("chunk_size", 200, 1000, 500, step=100)
    chunk_overlap = st.slider("chunk_overlap", 0, 200, 100, step=50)
    top_k = st.slider("top_k", 1, 8, 3)
    enable_mq = st.toggle("开启 Multi-Query", value=True)
    enable_rer = st.toggle("开启 Reranker", value=True)

    # 4. 会话列表
    st.subheader("💬 会话列表")

    # 先取会话列表：既用于渲染选择器，也用于生成不重名的默认会话名
    code, resp = api("GET", "/api/session")
    sessions = resp.get("sessions", [])
    session_ids = [s["session_id"] for s in sessions]
    session_names = {s["session_id"]: s["session_name"] for s in sessions}

    # 默认名「会话 N」：N 取现有同名会话的最大编号 + 1，保证默认名不与已有会话重名
    used_nums = [
        int(n[3:]) for n in session_names.values()
        if n.startswith("会话 ") and n[3:].isdigit()
    ]
    default_name = f"会话 {max(used_nums) + 1 if used_nums else 1}"

    # 新建会话：名称可自定义。key 带版本号，创建成功后自增 → 输入框自动清空，
    # 避免同一个名字被反复提交（否则新会话又会重名）。
    name_gen = st.session_state.setdefault("name_gen", 0)
    new_name = st.text_input(
        "新会话名称", value="", placeholder=default_name,
        key=f"new_session_name_{name_gen}",
    )
    if st.button("➕ 新建会话"):
        name = (new_name or "").strip() or default_name
        # 用户手填的名字若与已有会话相同，补 (2)/(3) 后缀，从源头避免重名
        if name in session_names.values():
            i = 2
            while f"{name} ({i})" in session_names.values():
                i += 1
            name = f"{name} ({i})"

        code, resp = api("POST", "/api/session", json={"session_name": name})
        if code == 200:
            sid = resp["session_id"]
            # 必须同时写选择器的 widget state：否则它仍持有旧值，
            # 下一轮会把 session_id 覆盖回旧会话，表现为「新建了但没切过去」
            st.session_state["session_selector"] = sid
            st.session_state["session_id"] = sid
            st.session_state["loaded_sid"] = sid
            st.session_state["history"] = []
            st.session_state["name_gen"] = name_gen + 1
            st.rerun()
        else:
            st.error(f"新建失败：{resp.get('detail', resp.get('error', code))}")

    if sessions:
        # 选择器的 key 固定，保证选择结果跨 rerun 稳定保留；
        # 若它持有的是已被删除的会话，Streamlit 会因「值不在选项内」报错，先归位到第一项
        if st.session_state.get("session_selector") not in session_ids:
            st.session_state["session_selector"] = session_ids[0]

        sel = st.selectbox(
            "选择会话",
            session_ids,
            format_func=lambda x: session_names.get(x, x),
            key="session_selector",
        )
        # 以选择器为准：切换后立即生效（主区按 session_id 变化重载历史）
        st.session_state["session_id"] = sel

        # 删除当前选中的会话
        if st.button("🗑️ 删除当前会话", key="del_session"):
            dc, dr = api("DELETE", f"/api/session/{sel}")
            if dc == 200:
                st.success("会话已删除")
                # 清空前端会话状态（含选择器 widget state），避免继续引用已删除的 session_id
                for k in ("session_id", "history", "loaded_sid", "session_selector"):
                    st.session_state.pop(k, None)
                st.rerun()
            else:
                st.error(f"删除失败：{dr.get('detail', dr.get('error', dc))}")
    else:
        st.info("暂无会话，请先新建")


# ==================== 主区域 ====================
st.title("🤖 RAG 智能问答")

# 初始化会话状态
if "history" not in st.session_state:
    st.session_state["history"] = []
sid = st.session_state.get("session_id")

# 切换会话时自动加载历史
if sid and sid != st.session_state.get("loaded_sid"):
    st.session_state["history"] = load_history(sid)
    st.session_state["loaded_sid"] = sid

# 显式展示当前会话：仅靠侧边栏选择器时，会话名相同就无法判断是否切换成功
current_name = session_names.get(sid) if sid else None
if current_name:
    st.caption(f"当前会话：**{current_name}** · {len(st.session_state['history'])} 条消息")
else:
    st.caption("当前未选择会话，请在左侧新建或选择一个会话")

# 渲染对话气泡
for msg in st.session_state["history"]:
    role = msg.get("role")
    content = msg.get("content")
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        # 助手回答的折叠面板：来源 + 调试信息
        if role == "assistant" and "meta" in msg:
            meta = msg["meta"]
            with st.expander("📎 来源与置信度"):
                for s in meta.get("sources", []):
                    st.markdown(f"- **[{s.get('index')}]** {s.get('source')}")
                    preview = s.get("content_preview", "")
                    if preview:
                        st.caption(preview)
                st.markdown(f"**置信度：{meta.get('confidence')}**")
            with st.expander("🔍 调试信息"):
                st.write("改写后 Query：", meta.get("rewritten_query"))
                st.write("Multi-Query 子查询：")
                st.json(meta.get("sub_queries", []))
                st.write("Rerank 打分：")
                st.json(meta.get("rerank_scores", []))

# 聊天输入
question = st.chat_input("输入你的问题...")

if question and sid:
    # 追加用户消息到界面
    st.session_state["history"].append({"role": "user", "content": question})

    payload = {
        "session_id": sid,
        "question": question,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "enable_multi_query": enable_mq,
        "enable_reranker": enable_rer,
    }
    code, resp = api("POST", "/api/chat", json=payload)

    if code == 200:
        st.session_state["history"].append({
            "role": "assistant",
            "content": resp.get("answer", "（无回答）"),
            "meta": {
                "sources": resp.get("sources", []),
                "confidence": resp.get("confidence"),
                "rewritten_query": resp.get("rewritten_query"),
                "sub_queries": resp.get("sub_queries"),
                "rerank_scores": resp.get("rerank_scores"),
            },
        })
    else:
        err = resp.get("detail", resp.get("error", f"HTTP {code}"))
        st.error(f"⚠️ 请求失败：{err}")
        st.session_state["history"].append({
            "role": "assistant",
            "content": f"⚠️ 请求失败：{err}",
        })

    st.rerun()
elif question and not sid:
    st.warning("请先在侧边栏新建或选择一个会话")