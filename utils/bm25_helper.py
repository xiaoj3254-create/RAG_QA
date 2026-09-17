"""BM25 稀疏检索辅助模块：rank-bm25 + jieba 中文分词 + RRF 多路融合。

提供两个能力：
1. BM25Helper —— 从 Chroma 语料构建并维护 BM25 稀疏检索索引（懒加载 + 脏标记重建）；
2. rrf_fuse —— Reciprocal Rank Fusion，将多路检索结果按名次融合为一个排序列表。

设计要点：
- 同一份语料：索引直接取自 Chroma collection（与向量检索同一批 chunk），
  双路检索面向同一数据集，不会出现语料不一致；
- 索引懒加载：首次检索时构建，之后复用；文档增删后由 DocumentProcessor
  调用 invalidate() 标记失效，下次检索按需重建（并用 chunk 数做脏检测兜底）；
- 并发安全：索引构建在锁内完成，调用方持有锁内快照在锁外检索，
  即使期间索引被重建，快照内 bm25 与 docs 仍严格按行对齐；
- 全链路降级：jieba / rank-bm25 缺失、索引构建失败、检索异常均返回空列表，
  主链路自动回退纯向量检索，不影响问答可用性。
"""

import threading
from typing import List, Dict, Optional

from langchain_core.documents import Document

from utils.logger import setup_logger
import config

logger = setup_logger("bm25", config.LOG_FILE)

# 依赖懒探测：缺失时模块仍可导入，search() 直接返回空列表（降级纯向量）
try:
    import jieba
    jieba.setLogLevel(60)  # 关闭 jieba 初始化日志噪音
    _JIEBA_READY = True
except ImportError:
    _JIEBA_READY = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_READY = True
except ImportError:
    _BM25_READY = False


def tokenize(text: str) -> List[str]:
    """中文分词（jieba），过滤空白 token。

    无 jieba 时退化为英文按空格切分。
    """
    if not _JIEBA_READY:
        return [t for t in text.split() if t.strip()]
    return [t.strip() for t in jieba.lcut(text) if t.strip()]


class BM25Helper:
    """BM25 索引管理器：从 Chroma 语料构建稀疏检索索引（懒加载 + 脏标记重建）。

    由 DocumentProcessor 持有（语料的写入方负责失效标记），Retriever 只读检索。
    """

    def __init__(self, vector_store):
        self._vector_store = vector_store
        self._lock = threading.Lock()   # 保护索引构建（并发请求下只建一次）
        self._bm25 = None               # BM25Okapi 实例
        self._docs: List[Document] = [] # 与 BM25 语料按行对齐的 Document 列表
        self._chunk_count = -1          # 建索引时的语料规模，用于脏检测

    # ---------- 索引管理 ----------

    def invalidate(self) -> None:
        """文档增删后调用：标记索引失效，下次检索按需重建。"""
        with self._lock:
            self._bm25 = None
            self._chunk_count = -1

    def _ensure_index(self):
        """确保索引可用（懒构建 + chunk 数脏检测）。

        返回锁内取得的 (bm25, docs) 快照；索引不可用时返回 (None, None)。
        调用方在锁外使用快照检索：即使期间索引被 invalidate/重建，
        快照内部 bm25 与 docs 仍严格按行对齐，不会错位。
        """
        with self._lock:
            collection = getattr(self._vector_store, "_collection", None)
            if collection is None:
                return None, None
            current_count = collection.count()
            # 索引已构建且语料规模未变 → 直接复用
            if self._bm25 is not None and current_count == self._chunk_count:
                return self._bm25, self._docs
            if current_count == 0:
                self._bm25 = None
                self._chunk_count = 0
                return None, None

            data = collection.get(include=["documents", "metadatas"])
            corpus = data.get("documents") or []
            metadatas = data.get("metadatas") or []
            tokenized = [tokenize(doc) for doc in corpus]
            self._bm25 = BM25Okapi(tokenized)
            self._docs = [
                Document(page_content=content, metadata=meta or {})
                for content, meta in zip(corpus, metadatas)
            ]
            self._chunk_count = current_count
            logger.info("BM25 索引已构建：%d 个 chunk", current_count)
            return self._bm25, self._docs

    # ---------- 检索 ----------

    def search(self, query: str, k: int = 5) -> List[Document]:
        """BM25 稀疏检索，返回按相关性降序的 top-k Document。

        得分为 0 的 chunk 会被过滤（避免无关结果混入融合）；
        任何异常返回空列表，主链路回退纯向量。
        """
        if not (_BM25_READY and _JIEBA_READY):
            return []
        if not query or not query.strip():
            return []
        try:
            bm25, docs = self._ensure_index()
            if bm25 is None:
                return []
            scores = bm25.get_scores(tokenize(query))
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [docs[i] for i in top if scores[i] > 0]
        except Exception as e:
            logger.warning("BM25 检索失败，返回空结果: %s", e)
            return []


def rrf_fuse(result_lists: List[List[Document]], rrf_k: int = 60,
             top_n: int = 10) -> List[Document]:
    """Reciprocal Rank Fusion：融合多路排序结果。

    每路结果按名次贡献 1/(rrf_k + rank) 分，累加后按总分降序取 top_n。
    以 page_content 为去重键（与 retrieve_multi 的去重口径一致），
    被多路同时命中的 chunk 自然获得更高排名——这正是融合的价值所在。

    Args:
        result_lists: 多路检索结果（每路为按相关性降序的 Document 列表）
        rrf_k: 平滑常数（业界默认 60，抑制排名靠后结果的贡献）
        top_n: 融合后返回的候选数
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            docs.setdefault(key, doc)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [docs[key] for key, _ in ranked[:top_n]]
