"""FastAPI 后端服务层：串起 RAG 核心 + SQLite 元数据存储 + 文档上传。

- /api/doc/*：文档上传与列表
- /api/chat：问答（保存会话消息）
- /api/session/*：会话管理
- /health：前端连通性探测
- /docs：FastAPI 自动生成的接口文档

设计要点：全局单例 RAGChain（避免 Chroma/LLM 重复实例化）；
前端参数通过 chat 请求体临时覆盖（不污染全局配置）。
"""
import os
import threading
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

import config
from utils.logger import setup_logger
from utils import file_loader
from db import sqlite_db
from main import RAGChain, RAGConfig, Retriever

logger = setup_logger("api", config.LOG_FILE)

app = FastAPI(title="RAG QA System", version="1.0.0")


# ----------------------------------------------------------------------
# 全局单例 RAGChain（避免 Chroma/LLM 重复实例化）
# ----------------------------------------------------------------------
_rag: Optional[RAGChain] = None
_config_lock = threading.Lock()  # 保护临时覆盖运行配置，避免并发请求互相干扰


def get_rag() -> RAGChain:
    """返回全局单例 RAGChain；首次调用时加锁构造，避免并发首请求各自 new 一个实例并互相覆写。"""
    global _rag
    if _rag is None:
        with _config_lock:
            if _rag is None:  # 双检锁：锁内再次判空，防止重复构造
                cfg = RAGConfig()
                _rag = RAGChain(cfg)
                # 用空列表建图（后续通过上传实时追加；Chroma 会加载已有持久化数据）
                _rag.index_documents([])
    return _rag


@app.on_event("startup")
def startup():
    """服务启动：初始化 SQLite 建表 + 预热 RAG 单例。"""
    sqlite_db.init_db()
    get_rag()
    logger.info("RAG QA 服务已启动。接口文档: /docs")


# ----------------------------------------------------------------------
# 请求模型
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    question: str
    # 前端参数面板可选的临时覆盖
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    enable_multi_query: Optional[bool] = None
    enable_reranker: Optional[bool] = None


class CreateSessionRequest(BaseModel):
    session_name: str = "新会话"


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    sub_queries: List[str] = []
    rewritten_query: str = ""
    rerank_scores: List[Dict[str, Any]] = []


# ----------------------------------------------------------------------
# 健康检查
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------------------------------------------------
# 文档相关
# ----------------------------------------------------------------------
@app.post("/api/doc/upload")
async def upload_doc(file: UploadFile = File(...)):
    """上传并解析文档（txt/md/pdf），写入 Chroma 并记录元数据到 SQLite。"""
    # filename 可能为空（某些客户端不携带），兜底占位名
    filename = file.filename or "unnamed"
    # 大小限制校验
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(413, f"文件超过 {config.MAX_UPLOAD_MB}MB 限制")

    # 后缀校验（与 utils/file_loader.py 的 SUPPORTED_EXTS 保持一致）
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".json", ".csv"}:
        raise HTTPException(400, "仅支持 txt/md/pdf/json/csv 文件")

    # 写临时文件后走 file_loader 解析（复用 PDF/编码兜底逻辑）
    # tmp 目录固定在项目根下，避免按 CHROMA_DB_PATH 推断依赖进程 CWD 而分叉
    tmp_dir = Path(config.ROOT_DIR) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fpath = tmp_dir / f"{os.urandom(4).hex()}_{filename}"
    fpath.write_bytes(content)

    try:
        docs = file_loader.load_files([fpath])
    finally:
        fpath.unlink(missing_ok=True)  # 用完即删

    if not docs:
        raise HTTPException(400, "文件解析为空或损坏，无法索引")

    # 分块 + 追加到 Chroma；retriever 重关联与建图是单例状态变更，
    # 与 chat 的配置覆盖/恢复共用锁，避免并发竞态
    rag = get_rag()
    # 预生成 doc_id，注入每个 chunk 的 metadata，建立「SQLite 文档 ↔ Chroma 向量」关联，
    # 后续删除文档时可按 doc_id 精准定位并删除对应向量。
    doc_id = uuid.uuid4().hex
    with _config_lock:
        chunks = rag.processor.split_documents(docs)
        for c in chunks:
            c.metadata["doc_id"] = doc_id
        rag.processor.create_vector_store(chunks)
        rag.retriever = Retriever(rag.processor.vector_store, rag.config)
        rag._build_graph()

    # 记录元数据到 SQLite
    sqlite_db.add_uploaded_doc_meta(filename, len(chunks), doc_id=doc_id)
    return {"doc_id": doc_id, "file_name": filename, "chunk_count": len(chunks)}


@app.get("/api/doc/list")
def list_docs():
    """获取已上传文档元数据列表。"""
    return {"docs": sqlite_db.list_uploaded_docs()}


@app.delete("/api/doc/{doc_id}")
def delete_doc(doc_id: str):
    """删除已上传文档：同时清理 Chroma 向量与 SQLite 元数据。

    先删 Chroma 向量（按 doc_id 过滤），再删 SQLite 记录；
    若 Chroma 删除失败仍继续删元数据，避免残留死记录。
    """
    doc = sqlite_db.get_uploaded_doc(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")

    rag = get_rag()
    # 向量与元数据分两步，互不阻塞：即使 Chroma 删除异常，元数据也要清掉
    # 传入 file_name 作为 source 兜底匹配，兼容早期未注入 doc_id 的历史 chunk
    with _config_lock:
        deleted_chunks = rag.processor.delete_documents_by_doc_id(doc_id, source=doc["file_name"])
    sqlite_db.delete_uploaded_doc(doc_id)
    logger.info("删除文档: %s (chunks_removed=%d)", doc["file_name"], deleted_chunks)
    return {
        "doc_id": doc_id,
        "file_name": doc["file_name"],
        "deleted_chunks": deleted_chunks,
    }


# ----------------------------------------------------------------------
# 问答
# ----------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """问答：内部调用 RAGChain.query，并将问答消息保存到 SQLite。

    运行参数（chunk/top_k/开关）是进程内单例的临时覆盖：互斥锁保护覆盖+查询全程，
    保证并发请求不互相污染；finally 里快照恢复，避免一个会话的参数泄漏给下一个会话。
    """
    rag = get_rag()

    # 会话必须先存在：外键已开启，直接写入会抛 IntegrityError，这里显式校验并返回 404
    if not any(s["session_id"] == req.session_id for s in sqlite_db.get_all_sessions()):
        raise HTTPException(404, "会话不存在，请先创建会话")

    # 覆盖+查询全程加锁，避免并发请求各自盖掉对方的运行参数
    with _config_lock:
        # 快照当前配置，finally 恢复，防止临时覆盖跨请求泄漏
        _snapshot = (
            rag.processor.config.chunk_size,
            rag.processor.config.chunk_overlap,
            rag.retriever.config.top_k,
            rag.config.enable_multi_query,
            rag.config.enable_reranker,
        )
        try:
            chunk_cfg_changed = (req.chunk_size is not None
                                 and rag.processor.config.chunk_size != req.chunk_size)
            overlap_cfg_changed = (req.chunk_overlap is not None
                                   and rag.processor.config.chunk_overlap != req.chunk_overlap)
            if req.chunk_size is not None:
                rag.processor.config.chunk_size = req.chunk_size
            if req.chunk_overlap is not None:
                rag.processor.config.chunk_overlap = req.chunk_overlap
            # chunk 参数变了必须重建 splitter，否则改 config 不生效
            if chunk_cfg_changed or overlap_cfg_changed:
                rag.processor.rebuild_text_splitter()
            if req.top_k is not None:
                rag.retriever.config.top_k = req.top_k
            if req.enable_multi_query is not None:
                rag.config.enable_multi_query = req.enable_multi_query
                rag.retriever.config = rag.config
            if req.enable_reranker is not None:
                rag.config.enable_reranker = req.enable_reranker

            # 从会话历史构建 chat_history（供 RAG 多轮改写）
            messages = sqlite_db.get_session_messages(req.session_id)
            chat_history = [{"role": m["role"], "content": m["content"]} for m in messages]

            result = rag.query(req.question, chat_history)
        except ValueError as e:
            raise HTTPException(500, str(e))
        finally:
            # 恢复配置快照：临时覆盖不跨请求残留
            rag.processor.config.chunk_size, rag.processor.config.chunk_overlap, \
                rag.retriever.config.top_k, rag.config.enable_multi_query, \
                rag.config.enable_reranker = _snapshot
            # 若本次请求重建过 splitter，恢复配置后必须同步重建，否则 splitter 与配置脱节
            if chunk_cfg_changed or overlap_cfg_changed:
                rag.processor.rebuild_text_splitter()

    # 保存问答消息到 SQLite（助手消息附带 meta，前端刷新历史仍可看来源/置信度/调试信息）
    sqlite_db.save_message(req.session_id, "user", req.question)
    sqlite_db.save_message(req.session_id, "assistant", result["answer"], meta={
        "sources": result["sources"],
        "confidence": result["confidence"],
        "rewritten_query": result["rewritten_query"],
        "sub_queries": result["sub_queries"],
        "rerank_scores": result["rerank_scores"],
    })

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "sub_queries": result["sub_queries"],
        "rewritten_query": result["rewritten_query"],
        "rerank_scores": result["rerank_scores"],
    }


# ----------------------------------------------------------------------
# 会话相关
# ----------------------------------------------------------------------
@app.post("/api/session")
def create_session(req: CreateSessionRequest):
    """新建会话，返回 session_id。"""
    sid = sqlite_db.create_session(req.session_name)
    return {"session_id": sid, "session_name": req.session_name}


@app.get("/api/session")
def list_sessions():
    """获取全部会话列表。"""
    return {"sessions": sqlite_db.get_all_sessions()}


@app.get("/api/session/{session_id}/history")
def session_history(session_id: str):
    """获取会话历史消息。"""
    msgs = sqlite_db.get_session_messages(session_id)
    # 会话不存在时返回 404
    if not msgs and not any(s["session_id"] == session_id for s in sqlite_db.get_all_sessions()):
        raise HTTPException(404, "会话不存在")
    return {"session_id": session_id, "messages": msgs}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """删除会话及其全部历史消息。"""
    deleted = sqlite_db.delete_session(session_id)
    if not deleted:
        raise HTTPException(404, "会话不存在")
    return {"session_id": session_id, "deleted": True}