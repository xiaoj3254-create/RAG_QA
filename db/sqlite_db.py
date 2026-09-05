"""SQLite 元数据存储：sessions / messages / uploaded_docs 三张表。

职责：只存储会话与文档的元数据，不存储向量（向量保存在 Chroma）。
启动时调用 init_db() 自动建表（幂等）。

零第三方依赖，仅使用 sqlite3 标准库，保证模块可独立运行。
"""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import config
from utils.logger import setup_logger

logger = setup_logger("sqlite", config.LOG_FILE)


def _connect() -> sqlite3.Connection:
    """建立连接，返回类字典行。

    并发安全：FastAPI 同步接口跑在线程池，多个请求可能同时写库。
    - timeout 允许在锁定短暂阻塞后自动重试，避免直接抛 \"database is locked\"；
    - PRAGMA foreign_keys=ON 真正启用外键约束（防孤儿消息）；
    - 只读连接同时开启 WAL，减少读写互斥。
    """
    conn = sqlite3.connect(config.SQLITE_DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """幂等初始化：创建三张表（目录不存在时自动创建）。"""
    # 确保 db 目录存在
    import os
    os.makedirs(os.path.dirname(config.SQLITE_DB_PATH) or ".", exist_ok=True)

    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions(
                session_id TEXT PRIMARY KEY,
                session_name TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,          -- 'user' | 'assistant'
                content TEXT NOT NULL,
                meta TEXT,                   -- JSON：来源/置信度/调试信息（助手回答）
                created_at REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            CREATE TABLE IF NOT EXISTS uploaded_docs(
                doc_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                upload_time REAL NOT NULL
            );
            """
        )
        # 旧库迁移：messages 表没有 meta 列时补齐（刷新历史后来源/置信度面板仍可用）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
        if "meta" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN meta TEXT")
            logger.info("已为 messages 表补充 meta 列（旧库迁移）")
    logger.info("SQLite 初始化完成: %s", config.SQLITE_DB_PATH)


# ----------------------------------------------------------------------
# 会话
# ----------------------------------------------------------------------
def create_session(session_name: str = "新会话") -> str:
    """新建会话，返回 session_id。"""
    sid = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions(session_id, session_name, created_at) VALUES(?,?,?)",
            (sid, session_name, time.time()),
        )
    logger.info("创建会话: %s (%s)", sid, session_name)
    return sid


def get_all_sessions() -> List[Dict[str, Any]]:
    """返回全部会话（按创建时间倒序）。"""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    """删除会话及其全部消息。

    messages 表外键未声明 ON DELETE CASCADE，因此先删消息再删会话，
    避免外键约束报错。返回 True 表示确实删除了一条会话记录；False 表示会话不存在。
    """
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        cur = conn.execute(
            "DELETE FROM sessions WHERE session_id=?", (session_id,)
        )
        deleted = cur.rowcount > 0
    if deleted:
        logger.info("删除会话: %s", session_id)
    return deleted


# ----------------------------------------------------------------------
# 消息
# ----------------------------------------------------------------------
def save_message(session_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """保存一条问答消息（role: user / assistant）。

    meta: 助手回答的附带信息（来源/置信度/调试信息），序列化后存入 meta 列，
    前端刷新历史后仍能展示来源与调试面板。
    """
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content, meta, created_at) VALUES(?,?,?,?,?)",
            (session_id, role, content, json.dumps(meta, ensure_ascii=False) if meta else None, time.time()),
        )


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """获取某个会话的全部历史消息（按 id 升序），meta 反序列化为 dict。"""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, session_id, role, content, meta, created_at FROM messages"
            " WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    msgs = [dict(r) for r in rows]
    for m in msgs:
        if m.get("meta"):
            try:
                m["meta"] = json.loads(m["meta"])
            except (TypeError, ValueError) as e:
                logger.warning("消息 meta 解析失败，忽略: %s", e)
                m.pop("meta", None)
        else:
            # 无 meta（用户消息或旧数据）时不返回该键，避免前端按 \"meta in msg\" 误判为空面板
            m.pop("meta", None)
    return msgs


# ----------------------------------------------------------------------
# 上传文档元数据
# ----------------------------------------------------------------------
def add_uploaded_doc_meta(file_name: str, chunk_count: int, doc_id: Optional[str] = None) -> str:
    """写入上传文档元数据，返回 doc_id。

    doc_id 可选：由调用方（api.py）预先生成，以便在写入 Chroma 前
    把 doc_id 注入 chunk 元数据，建立「SQLite 文档 ↔ Chroma 向量」的关联。
    """
    if doc_id is None:
        doc_id = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            "INSERT INTO uploaded_docs(doc_id, file_name, chunk_count, upload_time)"
            " VALUES(?,?,?,?)",
            (doc_id, file_name, chunk_count, time.time()),
        )
    logger.info("记录上传文档: %s (chunks=%d)", file_name, chunk_count)
    return doc_id


def get_uploaded_doc(doc_id: str) -> Optional[Dict[str, Any]]:
    """按 doc_id 查询上传文档元数据；不存在返回 None。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM uploaded_docs WHERE doc_id=?", (doc_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_uploaded_doc(doc_id: str) -> bool:
    """删除上传文档元数据记录。返回 True 表示确实删除了一条；False 表示不存在。"""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM uploaded_docs WHERE doc_id=?", (doc_id,)
        )
        deleted = cur.rowcount > 0
    if deleted:
        logger.info("删除上传文档元数据: %s", doc_id)
    return deleted


def list_uploaded_docs() -> List[Dict[str, Any]]:
    """返回全部已上传文档元数据（按上传时间倒序）。"""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM uploaded_docs ORDER BY upload_time DESC"
        ).fetchall()
    return [dict(r) for r in rows]