"""BM25 稀疏检索辅助模块：rank-bm25 + jieba 中文分词 + RRF 融合。

设计要点：
- 语料直接取自 Chroma collection（与向量检索同一份 chunk），双路检索面向同一数据集
- 索引懒加载：首次检索时构建，之后复用；文档增删后由 DocumentProcessor 调用
  invalidate() 标记失效，下次检索按需重建（同时用 chunk 数做脏检测兜底）
- 纯向量检索对"第6条"这类编号/关键词查询不敏感（语义相似度对数字编号无区分度），
  BM25 恰好补足字面精确匹配；两路结果经 RRF 融合后兼顾语义相关性与关键词命中
- 全链路降级：jieba/rank-bm25 缺失、索引构建失败、检索异常均返回空列表，
  主链路自动回退纯向量检索，不影响问答可用性
"""

import re
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


# ---------- 法条编号归一化 ----------
# 用户查询习惯写"第6条"，而法条原文是"第六条"——阿拉伯数字与汉字数字不一致
# 会导致 BM25 分词完全错开。检索前统一把 第N条/款/章/项 的阿拉伯数字转为汉字。
_CN_DIGITS = "零一二三四五六七八九"


def _arabic_to_cn(num: int) -> str:
    """1~999 的整数转中文数字（法条编号实际范围；超出原样返回）。"""
    if num <= 0 or num >= 1000:
        return str(num)
    digits = [int(d) for d in str(num)]
    if len(digits) == 1:
        return _CN_DIGITS[num]
    if len(digits) == 2:
        tens, ones = digits
        prefix = "十" if tens == 1 else _CN_DIGITS[tens] + "十"
        return prefix + (_CN_DIGITS[ones] if ones else "")
    hundreds, tens, ones = digits
    result = _CN_DIGITS[hundreds] + "百"
    if tens == 0 and ones == 0:
        return result
    if tens == 0:
        return result + "零" + _CN_DIGITS[ones]
    result += _CN_DIGITS[tens] + "十"
    if ones:
        result += _CN_DIGITS[ones]
    return result


_LEGAL_NUM_RE = re.compile(r"第(\d+)([条款章项])")


def _normalize_legal_numbers(text: str) -> str:
    """第6条 → 第六条（同样处理 款/章/项），消除数字风格差异。"""
    return _LEGAL_NUM_RE.sub(
        lambda m: "第" + _arabic_to_cn(int(m.group(1))) + m.group(2), text
    )


def tokenize(text: str) -> List[str]:
    """法条编号归一化 + 中文分词（jieba），过滤空白 token。

    无 jieba 时退化为英文按空格切分。
    """
    if not _JIEBA_READY:
        return [t for t in text.split() if t.strip()]
    return [t.strip() for t in jieba.lcut(_normalize_legal_numbers(text)) if t.strip()]


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

    def _ensure_index(self) -> bool:
        """确保索引可用（懒构建 + chunk 数脏检测）。返回索引是否可用。"""
        with self._lock:
            collection = getattr(self._vector_store, "_collection", None)
            if collection is None:
                return False
            current_count = collection.count()
            # 索引已构建且语料规模未变 → 直接复用
            if self._bm25 is not None and current_count == self._chunk_count:
                return True
            if current_count == 0:
                self._bm25 = None
                self._chunk_count = 0
                return False

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
            return True

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
            if not self._ensure_index():
                return []
            scores = self._bm25.get_scores(tokenize(query))
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [self._docs[i] for i in top if scores[i] > 0]
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
