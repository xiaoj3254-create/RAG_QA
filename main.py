"""
RAG 检索增强生成系统 - 完整实现

本模块实现了生产级别的 RAG 系统，包括：
- 文档加载和处理（支持 Chroma 磁盘持久化）
- 向量存储和检索（Multi-Query 多路召回 + 硅基流动 rerank 重排）
- 上下文感知的问答生成（幻觉自检 + LangGraph 条件分支重试）
- 来源引用和置信度评估

⚠️ Embeddings 说明：
- 默认使用简单的 Fake Embeddings（用于演示）
- 如需高质量结果，请在 .env 中配置 OPENAI_API_KEY（硅基流动密钥），通过 OpenAI 兼容协议调用硅基流动的 BAAI/bge-m3 向量模型
- 使用 SimpleEmbeddings 时程序会打印警告，生产环境必须配置真实 Embedding

⚠️ LLM / Embedding 说明：
- 统一走 OpenAI 兼容协议接硅基流动（SiliconFlow），默认生成模型 deepseek-ai/DeepSeek-V4-Flash，向量模型 BAAI/bge-m3；
  未配置密钥时自动进入演示降级模式，仍可跑通检索、来源、子查询、重排、幻觉校验全流程
"""

import re
from typing import List, Dict, Any, Optional, TypedDict, Literal
from dataclasses import dataclass

import config
from dotenv import load_dotenv

# LangChain 核心导入
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

# 文本处理
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 向量存储（Chroma 在 DocumentProcessor 中懒加载；此兜底用于安装失败时降级）
from langchain_core.vectorstores import InMemoryVectorStore

# LangGraph
from langgraph.graph import StateGraph, START, END

# 项目内部模块
from utils.logger import setup_logger
from utils import file_loader, reranker_helper, bm25_helper

logger = setup_logger("rag", config.LOG_FILE)

# 加载环境变量（密钥校验后移到 deepseek_model，无密钥时进入降级模式）
load_dotenv()


# 初始化生成模型（懒加载单例：生成/改写/多查询/幻觉检测共用同一实例）
_model: Any = None


def deepseek_model():
    """懒加载生成 LLM 单例；无有效密钥时抛出 ValueError。"""
    global _model
    if _model is None:
        if not config.DEEPSEEK_API_KEY or config.DEEPSEEK_API_KEY.startswith("your_"):
            raise ValueError("未设置有效的 DEEPSEEK_API_KEY")
        _model = init_chat_model(model=config.DEEPSEEK_MODEL,
                                model_provider="openai",
                                base_url=config.OPENAI_API_BASE,
                                api_key=config.DEEPSEEK_API_KEY)
    return _model


# ==================== 简单 Embeddings（用于演示）====================

class SimpleEmbeddings(Embeddings):
    """
    简单的 Embeddings 实现（用于演示）

    使用简单的文本哈希生成确定性伪随机向量，适合演示目的。
    生产环境请通过硅基流动（OpenAI 兼容协议）或 HuggingFace 使用真实语义 Embedding。
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        """简单的文本嵌入（基于字符哈希）"""
        import hashlib
        # 使用文本的hash生成伪随机但确定的向量
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # 扩展到目标维度
        vector = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            # 归一化到 [-1, 1]
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            vector.append(value)

        return vector


# 选择 Embeddings
def get_embeddings():
    """根据环境选择合适的 Embeddings"""
    # 只判断"配置了真实密钥"，不在源码里比对具体 key 值（避免密钥泄漏进代码库）
    if config.OPENAI_API_KEY and not config.OPENAI_API_KEY.startswith("your_"):
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info("使用硅基流动 Embeddings（OpenAI 兼容协议）: %s（base: %s）",
                        config.OPENAI_EMBEDDING_MODEL, config.OPENAI_API_BASE)
            # timeout=30：embed_documents 批量嵌入可能较慢，10s 不够
            return OpenAIEmbeddings(
                model=config.OPENAI_EMBEDDING_MODEL,
                base_url=config.OPENAI_API_BASE,
                api_key=config.OPENAI_API_KEY,
                timeout=30,
                # 硅基流动等非 OpenAI 厂商不接受 langchain 默认的 tokenize 分片后
                # 提交的 token 整数数组（会 400 The parameter is invalid）。
                # 关闭后直接提交原始文本字符串，兼容多路接入。
                check_embedding_ctx_length=False,
            )
        except ImportError:
            logger.warning("langchain_openai 未安装，使用简单 Embeddings")

    # 兜底演示：必须打印生产警告
    logger.warning(
        "⚠️ 未检测到有效的 OPENAI_API_KEY，使用 SimpleEmbeddings 兜底演示。"
        "生产环境必须配置真实 Embedding，否则检索质量无法保证！"
    )
    return SimpleEmbeddings()


# ==================== 配置类 ====================

@dataclass
class RAGConfig:
    """RAG 系统配置（默认值全部来自 config 模块）"""

    # 模型配置
    temperature: float = config.TEMPERATURE

    # 分块配置
    chunk_size: int = config.CHUNK_SIZE
    chunk_overlap: int = config.CHUNK_OVERLAP

    # 检索配置
    top_k: int = config.TOP_K
    search_type: str = "similarity"  # similarity, mmr

    # 生成配置
    max_tokens: int = config.MAX_TOKENS

    # 幻觉重试（防死循环）
    max_retry: int = config.MAX_RETRY

    # 功能开关
    enable_multi_query: bool = config.ENABLE_MULTI_QUERY
    enable_reranker: bool = config.ENABLE_RERANKER
    multi_query_num: int = config.MULTI_QUERY_NUM
    reranker_model_name: str = config.RERANKER_MODEL_NAME

    # 混合检索（BM25 稀疏 + 向量稠密，RRF 融合）
    enable_hybrid_search: bool = config.ENABLE_HYBRID_SEARCH
    bm25_candidate_k: int = config.BM25_CANDIDATE_K


# ==================== 状态定义 ====================

class RAGState(TypedDict):
    """RAG 流程状态"""
    query: str                          # 用户查询（含改写结果）
    chat_history: List[Dict[str, str]]  # 对话历史
    documents: List[Document]           # 检索到的文档
    context: str                        # 格式化的上下文
    answer: str                         # 生成的回答
    sources: List[Dict[str, Any]]       # 来源信息
    confidence: float                   # 置信度评分
    # ---- 新增字段 ----
    sub_queries: List[str]              # multi-query 生成的子查询
    rerank_scores: List[Dict[str, Any]] # 重排打分明细（供调试面板）
    is_hallucination: bool              # 幻觉校验结果
    retry_count: int                    # 已重试次数
    rewritten_query: str                # 改写后的查询（供调试面板）
    hallucination_feedback: str         # 幻觉重试时的纠偏反馈（供 generate 节点使用）


# ==================== 文档处理模块 ====================

class DocumentProcessor:
    """文档处理器：加载、分块、向量化（Chroma 磁盘持久化）"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        self.embeddings = get_embeddings()  # 使用智能选择的 Embeddings
        self.vector_store = None
        self._create_vector_store()  # 启动即创建/加载一次，避免重复实例化
        # BM25 稀疏检索索引（与向量库同一份语料；归 DocumentProcessor 持有，
        # 因为增删文档后由此负责失效标记；Retriever 只读检索）
        self.bm25 = bm25_helper.BM25Helper(self.vector_store)

    def _create_vector_store(self) -> None:
        """创建 Chroma 持久化向量库（加载已有数据，不重建）；失败回退内存向量库。"""
        try:
            from langchain_chroma import Chroma
            self.vector_store = Chroma(
                collection_name="rag_docs",
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_DB_PATH,
            )
            logger.info("Chroma 向量库就绪: %s", config.CHROMA_DB_PATH)
            # 启动即校验：持久化向量维度与当前 Embedding 是否一致
            self._check_embedding_compatibility()
        except Exception as e:
            # Chroma 安装失败/初始化异常时降级为内存向量库，保证演示可跑通
            logger.warning("⚠️ Chroma 初始化失败，回退 InMemoryVectorStore（无持久化）: %s", e)
            self.vector_store = InMemoryVectorStore(self.embeddings)

    def _check_embedding_compatibility(self) -> None:
        """检测已持久化向量维度与当前 Embedding 是否一致。

        常见踩坑：先用 SimpleEmbeddings(384 维) 写入 Chroma，之后配置 OPENAI_API_KEY
        重启后改用硅基流动 BAAI/bge-m3(1024 维)。维度不一致会导致每次检索都报错，
        而检索节点会静默吞掉异常返回空结果——表现为"什么都搜不到"，极难排查。
        这里显式检测并在日志中大声警告；同时清空该 collection，让新上传的文档
        以新维度重建索引（向量只是派生数据，源文档仍在，重新上传即可恢复）。
        """
        store = getattr(self.vector_store, "_collection", None)  # _collection 是 chromadb.Collection 对象
        if store is None:
            return  # 内存向量库无持久化数据，跳过

        try:
            if store.count() == 0:  # count() 返回集合里面文档总条数
                return
            existing = store.get(include=["embeddings"], limit=1)["embeddings"]
            # existing 可能是 numpy 数组或列表，len() 才是安全的判空（不该对数组做布尔求值）
            if existing is None or len(existing) == 0:
                return
            stored_dim = len(existing[0])
            current_dim = len(self.embeddings.embed_query("维度探测"))
            if stored_dim == current_dim:
                return

            logger.error(
                "⚠️ Embedding 维度不兼容：库中已有的向量维度为 %d，当前 Embedding 生成 %d 维。"
                "这是 SimpleEmbeddings 与硅基流动 BAAI/bge-m3 切换导致的。仅删除向量无法生效，"
                "因为 collection 的 HNSW 维度元数据仍停留在旧维度；这里直接删除并重建 collection，"
                "请重新上传文档以新维度建立索引。",
                stored_dim, current_dim,
            )
            # 仅 store.delete(ids) 不足：chromadb 的 collection 维度元数据不会随之更新，
            # 检索仍会报“expecting embedding with dimension of X”。必须 drop 后重建。
            store._client.delete_collection(store._name)  # type: ignore[attr-defined]
            from langchain_chroma import Chroma
            # 重建空 collection，让实例句柄指向新集合（旧的 _collection 已失效）
            self.vector_store = Chroma(
                collection_name="rag_docs",
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_DB_PATH,
            )
            logger.warning(
                "已重建 collection 'rag_docs'（旧维度 %d → 新维度 %d），请重新上传文档建立索引。",
                stored_dim, current_dim,
            )
        except Exception as e:
            logger.warning("Embedding 兼容性自检失败，跳过: %s", e)

    def load_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """从文本创建文档"""
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {"source": f"doc_{i}"}
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def rebuild_text_splitter(self) -> None:
        """按当前 config 的 chunk_size/chunk_overlap 重建 text_splitter。

        前端通过 API 覆盖 chunk 参数后调用，否则改 config 不生效（原 splitter 存的是初始值）。
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        logger.info("已按 chunk_size=%s / chunk_overlap=%s 重建 text_splitter",
                    self.config.chunk_size, self.config.chunk_overlap)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为小块"""
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]):
        """向已存在的向量存储追加文档（不新建实例，防重复实例化）"""
        if documents:
            self.vector_store.add_documents(documents)
        return self.vector_store

    def delete_documents_by_doc_id(self, doc_id: str, source: Optional[str] = None) -> int:
        """从 Chroma 中删除指定文档的所有 chunk，返回删除数量。

        删除策略（两级匹配，兼容历史数据）：
        1. 优先按 doc_id 精确匹配（新上传的文档在入库时已注入 doc_id）；
        2. 若按 doc_id 未命中，且提供了 source（文件名），则按 source 包含文件名兜底匹配
           （兼容早期上传、未注入 doc_id 的历史 chunk）。
        """
        store = getattr(self.vector_store, "_collection", None)
        if store is None:
            logger.warning("向量库无 _collection（可能是内存兜底），跳过删除")
            return 0
        try:
            # 第一级：按 doc_id 精确匹配
            result = store.get(where={"doc_id": doc_id}, include=[])
            ids = result.get("ids", []) or []

            # 第二级：doc_id 未命中时，按 source 兜底匹配（兼容历史数据）
            # 注意：chromadb 的 $contains 仅用于列表字段，不支持字符串子串匹配，
            # 因此拉取全部 metadata 在 Python 端做子串过滤。
            if not ids and source:
                all_data = store.get(include=["metadatas"])
                ids = [
                    cid for cid, meta in zip(all_data["ids"], all_data["metadatas"])
                    if meta and source in (meta.get("source") or "")
                ]
                if ids:
                    logger.info("doc_id=%s 无匹配，改用 source 兜底匹配到 %d 个 chunk", doc_id, len(ids))

            if ids:
                store.delete(ids=ids)
                logger.info("已从 Chroma 删除 doc_id=%s 的 %d 个 chunk", doc_id, len(ids))
                self.bm25.invalidate()  # 语料已变更，BM25 索引待重建
            return len(ids)
        except Exception as e:
            logger.warning("Chroma 删除失败 doc_id=%s: %s", doc_id, e)
            return 0

    def process(self, documents: List[Document], doc_id: Optional[str] = None) -> int:
        """Document 入库统一入口：注入 doc_id → 分块 → 追加向量化，返回 chunk 数。

        若传入 doc_id，会注入每个 Document 的 metadata，建立 SQLite 文档 ↔ Chroma 向量关联。
        api.py 上传流程与 index_documents 纯文本分支均通过此入口入库。
        """
        if doc_id:
            for doc in documents:
                doc.metadata["doc_id"] = doc_id

        logger.info("分割文档...")
        chunks = self.split_documents(documents)
        logger.info("生成了 %d 个文本块", len(chunks))

        logger.info("写入向量存储...")
        self.create_vector_store(chunks)
        logger.info("向量存储更新完成")
        self.bm25.invalidate()  # 语料已变更，BM25 索引待重建
        return len(chunks)

    def process_paths(self, paths: List[str], doc_id: Optional[str] = None) -> int:
        """文件路径入口：解析文件 -> 分块 -> 追加向量化，返回 chunk 数。

        解析后复用 process() 统一入库（支持 doc_id 注入）。
        """
        docs = file_loader.load_files(paths)
        if not docs:
            logger.warning("没有成功解析到任何文件")
            return 0
        return self.process(docs, doc_id=doc_id)


# ==================== 检索模块 ====================

class Retriever:
    """检索器：向量稠密检索 + BM25 稀疏检索，RRF 融合（可降级为纯向量）"""

    def __init__(self, vector_store, config: RAGConfig, bm25: Optional[bm25_helper.BM25Helper] = None):
        self.vector_store = vector_store
        self.config = config
        # BM25 索引由 DocumentProcessor 持有并随增删失效；未传入时自建（独立使用场景）
        self.bm25 = bm25 or bm25_helper.BM25Helper(vector_store)

    def _vector_search(self, query: str, k: int) -> List[Document]:
        """向量单路检索（异常返回空列表，不中断融合）"""
        try:
            return self.vector_store.similarity_search(query=query, k=k)
        except Exception as e:
            logger.warning("向量检索失败 q=%s: %s", query, e)
            return []

    def _hybrid_search(self, query: str) -> List[List[Document]]:
        """对单条查询执行 向量 + BM25 双路召回，返回结果列表（供 RRF 融合）"""
        lists = [self._vector_search(query, self.config.top_k)]
        bm25_docs = self.bm25.search(query, k=self.config.bm25_candidate_k)
        if bm25_docs:
            lists.append(bm25_docs)
        return lists

    def retrieve(self, query: str) -> List[Document]:
        """单路检索（混合开关开启时为向量+BM25 双路 RRF 融合）"""
        if not self.config.enable_hybrid_search:
            return self._vector_search(query, self.config.top_k)

        lists = self._hybrid_search(query)
        if len(lists) <= 1:
            return lists[0] if lists else []
        fused = bm25_helper.rrf_fuse(
            lists, rrf_k=config.RRF_K, top_n=self.config.top_k * 2
        )
        logger.info("混合检索（单路）：%d 路融合 → %d 个候选", len(lists), len(fused))
        return fused

    def retrieve_multi(self, queries: List[str]) -> List[Document]:
        """多查询召回。

        混合模式：每个子查询做 向量+BM25 双路召回，所有结果统一 RRF 融合——
        被多个子查询/多路同时命中的 chunk 排名自然靠前（跨查询共识）。
        纯向量模式：保留原有逐条检索、按内容去重合并逻辑。
        """
        if self.config.enable_hybrid_search:
            all_lists: List[List[Document]] = []
            for q in queries:
                all_lists.extend(self._hybrid_search(q))
            if not all_lists:
                return []
            fused = bm25_helper.rrf_fuse(
                all_lists, rrf_k=config.RRF_K, top_n=self.config.top_k * 3
            )
            logger.info("混合检索（多路）：%d 个子查询 × 2 路 = %d 路结果，RRF 融合 → %d 个候选",
                        len(queries), len(all_lists), len(fused))
            return fused

        # 纯向量多路召回（原逻辑）
        seen = set()
        merged: List[Document] = []
        for q in queries:
            for doc in self._vector_search(q, self.config.top_k):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    merged.append(doc)
            # 候选足够多时提前终止，避免无意义检索
            if len(merged) >= self.config.top_k * 3:
                break
        return merged

# ==================== 生成模块 ====================

class Generator:
    """生成器：基于上下文生成回答（含查询改写、置信度、幻觉检测）"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm: Any = None  # 懒加载生成模型，无密钥时保持 None 进入降级模式

        # RAG 提示模板
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的问答助手。请基于提供的上下文信息回答用户的问题。

重要规则：
1. 只使用提供的上下文信息来回答问题
2. 如果上下文中没有相关信息，请诚实地说"根据提供的信息，我无法回答这个问题"
3. 回答要准确、简洁、有条理
4. 在回答末尾标注信息来源

上下文信息：
{context}
"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{query}")
        ])

        # 查询改写提示
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询优化专家。请根据对话历史，将用户的问题改写为一个独立、完整的查询。

如果问题本身已经很清晰完整，直接返回原问题。
只返回改写后的查询，不要添加任何解释。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "原始问题：{query}\n\n请改写为独立完整的查询：")
        ])

        # 幻觉检测提示
        self.hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是幻觉检测器。请判断"回答"是否与"上下文"冲突，或者编造了上下文中不存在的内容。
只输出 JSON：{{"is_hallucination": true/false, "reason": "简短理由"}}"""),
            ("human", """上下文：{context}

问题：{query}

回答：{answer}""")
        ])

    def _llm_available(self) -> bool:
        """判断是否有可用的生成 LLM（无密钥时进入降级模式）。"""
        return bool(config.DEEPSEEK_API_KEY) and not config.DEEPSEEK_API_KEY.startswith("your_")

    def _get_llm(self):
        """懒加载获取 LLM 实例（单例）。"""
        if not self._llm_available():
            raise RuntimeError("未配置有效的 DEEPSEEK_API_KEY，无法调用 LLM")
        if self.llm is None:
            self.llm = deepseek_model()
        return self.llm

    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """根据对话历史改写查询（降级模式返回原查询）"""
        if not chat_history or not self._llm_available():
            return query

        # 转换对话历史格式
        messages = []
        for msg in chat_history[-4:]:  # 只用最近4轮对话
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        chain = self.rewrite_prompt | self._get_llm() | StrOutputParser()
        return chain.invoke({"query": query, "chat_history": messages})

    def generate(self, query: str, context: str, chat_history: List[Dict[str, str]] = None,
                 feedback: str = None) -> str:
        """生成回答。无 LLM 可用时返回降级文案，仍把检索到的上下文透出。

        feedback: 幻觉重试时传入纠偏提示，让 LLM 在相同的 context 下换一种更忠实的方式作答，
        避免重试变成对同一输入的空转（仅靠随机采样）。
        """
        if not self._llm_available():
            return (
                "⚠️ 演示模式：未配置 DEEPSEEK_API_KEY，无法调用 LLM 生成回答。\n\n"
                "已检索到相关上下文片段：\n" + (context or "（无）")[:600]
            )

        messages = []
        if chat_history:
            for msg in chat_history[-4:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        if feedback:
            messages.append(SystemMessage(
                content="重要：上一版回答因与上下文不符被幻觉校验驳回，请重新作答。"
                        f"纠偏要求：{feedback}。必须严格只使用【上下文信息】中的内容，"
                        "不得推断上下文之外的细节，如果上下文没有相关内容就明确说明无法回答。"
            ))

        chain = self.rag_prompt | self._get_llm() | StrOutputParser()
        return chain.invoke({
            "query": query,
            "context": context,
            "chat_history": messages
        })

    def evaluate_confidence(self, query: str, context: str, answer: str) -> float:
        """评估回答的置信度（无 LLM 时返回 0.5 兜底）"""
        if not self._llm_available():
            return 0.5

        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """评估以下回答的置信度。考虑：
1. 回答是否基于提供的上下文
2. 信息的相关性和准确性
3. 回答的完整性

只返回一个0到1之间的数字，表示置信度。"""),
            ("human", """上下文：{context}

问题：{query}

回答：{answer}

置信度（0-1）：""")
        ])

        chain = eval_prompt | self._get_llm() | StrOutputParser()
        try:
            raw = chain.invoke({
                "context": context,
                "query": query,
                "answer": answer
            })
            # DeepSeek V4 等推理模型会在正式输出前附 <reasoning>/思考块，
            # float("1</…>1") 会抛异常导致恒回退 0.5；先剥离标记再按正则提取数字。
            clean = re.sub(r"<[/]?[^>]+>", "", raw)
            match = re.search(r"\d+(?:\.\d+)?", clean)
            score = float(match.group()) if match else 0.0
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.warning("置信度评估异常，回退 0.5: %s", e)
            return 0.5

    def detect_hallucination(self, query: str, context: str, answer: str) -> bool:
        """幻觉检测：LLM 判断回答是否与上下文冲突/编造内容。

        Returns:
            True = 存在幻觉（需重试生成）；False = 无幻觉。
            异常时保守返回 False（不触发无谓重试）。
        """
        if not answer or not self._llm_available():
            return False

        chain = self.hallucination_prompt | self._get_llm() | StrOutputParser()
        try:
            raw = chain.invoke({"context": context, "query": query, "answer": answer})
            text = raw.strip().lower()
            # 解析 JSON 输出中的 is_hallucination 字段
            return '"is_hallucination": true' in text or text.startswith("true")
        except Exception as e:
            logger.warning("幻觉检测调用失败，默认不重试: %s", e)
            return False


# ==================== RAG 链整合 ====================

class RAGChain:
    """RAG 链：整合所有组件的 LangGraph 状态流程

    图结构（7 节点 + 条件边）：
    START → process_query → multi_query_node → retrieve → rerank_node → generate → hallucination_check_node
        ↓ 条件边
    is_hallucination == True  → 跳回 generate 重新生成（最多 retry max_retry 次）
    is_hallucination == False → evaluate → END
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.processor = DocumentProcessor(self.config)
        self.retriever = Retriever(self.processor.vector_store, self.config,
                                   bm25=self.processor.bm25)
        self.generator = Generator(self.config)
        self.graph = None

    def index_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """索引文档（兼容文本列表；若传入的是文件路径列表则走 file_loader）"""
        if texts and all(isinstance(p, str) and (p.endswith((".txt", ".md", ".pdf", ".json", ".csv"))) for p in texts):
            # 视为文件路径列表
            chunk_count = self.processor.process_paths(texts)
            logger.info("已从文件索引 %d 个文本块", chunk_count)
        else:
            # 纯文本列表：load_documents 转 Document 后走统一入库入口
            documents = self.processor.load_documents(texts, metadatas)
            self.processor.process(documents)

        # 重新关联 retriever 与最新的向量库（Chroma 模式下始终同一实例）
        self.retriever = Retriever(self.processor.vector_store, self.config,
                                   bm25=self.processor.bm25)
        self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 状态图（线性主链 + 幻觉条件边重试）"""

        def _format_context_and_sources(docs: List[Document]) -> Dict[str, Any]:
            """把文档列表格式化为 context 与 sources（retrieve/rerank 复用）"""
            context_parts = []
            sources = []
            for i, doc in enumerate(docs):
                context_parts.append(f"[文档 {i+1}] {doc.page_content}")
                sources.append({
                    "index": i + 1,
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:100] + "..."
                })
            return {"context": "\n\n".join(context_parts), "sources": sources}

        def process_query(state: RAGState) -> RAGState:
            """处理查询：改写查询（如有对话历史）"""
            query = state["query"]
            chat_history = state.get("chat_history", [])

            rewritten = query
            if chat_history:
                try:
                    rewritten = self.generator.rewrite_query(query, chat_history)
                    logger.info("查询改写：%s -> %s", query, rewritten)
                except Exception as e:
                    logger.warning("查询改写失败，使用原查询: %s", e)
            state["query"] = rewritten
            state["rewritten_query"] = rewritten
            return state

        def multi_query_node(state: RAGState) -> RAGState:
            """Multi-Query 节点：生成多个角度的子查询（开关关闭时仅用原查询）"""
            query = state["query"]
            sub_queries = [query]
            if self.config.enable_multi_query:
                try:
                    sub_queries = reranker_helper.multi_query_generate(
                        query, self.generator.llm or self.generator._get_llm(),
                        self.config.multi_query_num
                    )
                    logger.info("multi-query 子查询：%s", sub_queries)
                except Exception as e:
                    logger.warning("multi-query 生成失败，回退原查询: %s", e)
                    sub_queries = [query]
            state["sub_queries"] = sub_queries
            return state

        def retrieve_documents(state: RAGState) -> RAGState:
            """检索相关文档：多查询开关开启时多路召回，否则单路"""
            query = state["query"]
            try:
                if self.config.enable_multi_query:
                    docs = self.retriever.retrieve_multi(state.get("sub_queries") or [query])
                else:
                    docs = self.retriever.retrieve(query)
            except Exception as e:
                logger.error("向量库检索异常，返回空结果: %s", e)
                docs = []
            logger.info("检索到 %d 个相关文档", len(docs))

            state["documents"] = docs
            formatted = _format_context_and_sources(docs)
            state["context"] = formatted["context"]
            state["sources"] = formatted["sources"]
            return state

        def rerank_node(state: RAGState) -> RAGState:
            """Reranker 节点：对检索结果重排序（优先硅基流动 /rerank API，失败时保持原顺序）"""
            if self.config.enable_reranker and state.get("documents"):
                try:
                    docs = reranker_helper.rerank_documents(
                        state["query"],
                        state["documents"],
                        self.config.top_k,
                        self.config.reranker_model_name,
                    )
                    state["documents"] = docs
                    # 重排打分明细（供前端调试面板）
                    state["rerank_scores"] = [
                        {
                            "index": i + 1,
                            "score": doc.metadata.get("rerank_score"),
                            "source": doc.metadata.get("source", "unknown"),
                        }
                        for i, doc in enumerate(docs)
                    ]
                    formatted = _format_context_and_sources(docs)
                    state["context"] = formatted["context"]
                    state["sources"] = formatted["sources"]
                    logger.info("重排完成，返回 top-%d：%s", len(docs), state["rerank_scores"])
                except Exception as e:
                    logger.warning("重排异常，使用原顺序: %s", e)
            return state

        def generate_answer(state: RAGState) -> RAGState:
            """生成回答（LLM 异常时返回友好提示，保证流程不中断）。

            幻觉重试时传入 feedback，让 LLM 在相同 context 下换一种更难幻觉的方式作答。
            """
            try:
                state["answer"] = self.generator.generate(
                    query=state["query"],
                    context=state["context"],
                    chat_history=state.get("chat_history", []),
                    feedback=state.get("hallucination_feedback"),
                )
            except Exception as e:
                logger.error("回答生成失败: %s", e)
                state["answer"] = "抱歉，生成回答时发生错误，请稍后重试。"
            logger.info("回答生成完成（第 %d 次尝试）", state.get("retry_count", 0) + 1)
            return state

        def hallucination_check_node(state: RAGState) -> RAGState:
            """幻觉校验节点：判断回答是否存在幻觉；达到最大重试次数强制放行（防死循环）"""
            retries = state.get("retry_count", 0)
            if retries >= self.config.max_retry:
                # 已达最大重试上限，强制放行：is_hallucination 置 False，避免无限循环
                logger.warning("已达最大重试 %d 次，停止重试，放行回答", self.config.max_retry)
                state["is_hallucination"] = False
            else:
                state["is_hallucination"] = self.generator.detect_hallucination(
                    query=state["query"],
                    context=state["context"],
                    answer=state["answer"]
                )
                if state["is_hallucination"]:
                    # 下次条件边会跳回 generate 重新生成
                    state["retry_count"] = retries + 1
                    # 给重试生成一段纠偏反馈，避免相同输入的空转重试
                    state["hallucination_feedback"] = (
                        f"上一版回答（第 {retries + 1} 次尝试）被判为幻觉/与上下文冲突。"
                        "请重新生成：只引用【上下文信息】里明确出现的内容；"
                        "上下文没有的依据不要编造；若确实回答不了就直说。"
                    )
                    logger.info(
                        "检测到幻觉，准备第 %d/%d 次重试",
                        retries + 1, self.config.max_retry
                    )
            return state

        def evaluate_response(state: RAGState) -> RAGState:
            """评估回答置信度"""
            confidence = self.generator.evaluate_confidence(
                query=state["query"],
                context=state["context"],
                answer=state["answer"]
            )
            state["confidence"] = confidence
            logger.info("置信度评估：%.2f", confidence)
            return state

        # 构建图
        graph = StateGraph(RAGState)

        # 原有节点 + 新增节点
        graph.add_node("process_query", process_query)
        graph.add_node("multi_query_node", multi_query_node)
        graph.add_node("retrieve", retrieve_documents)
        graph.add_node("rerank_node", rerank_node)
        graph.add_node("generate", generate_answer)
        graph.add_node("hallucination_check_node", hallucination_check_node)
        graph.add_node("evaluate", evaluate_response)

        # 线性主链
        graph.add_edge(START, "process_query")
        graph.add_edge("process_query", "multi_query_node")
        graph.add_edge("multi_query_node", "retrieve")
        graph.add_edge("retrieve", "rerank_node")
        graph.add_edge("rerank_node", "generate")
        graph.add_edge("generate", "hallucination_check_node")

        # 条件边：幻觉为真且未超上限 → 跳回 generate 重新生成；否则 → evaluate
        def route_after_hallucination(state: RAGState) -> str:
            return "generate" if state.get("is_hallucination") else "evaluate"

        graph.add_conditional_edges(
            "hallucination_check_node",
            route_after_hallucination,
            {"generate": "generate", "evaluate": "evaluate"},
        )
        graph.add_edge("evaluate", END)

        self.graph = graph.compile()

    def query(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """执行查询，返回回答 + 来源 + 置信度 + 调试信息"""
        if not self.retriever:
            raise ValueError("请先调用 index_documents() 索引文档")

        logger.info("=%s=", "=" * 60)
        logger.info("问题：%s", question)
        logger.info("=%s=", "=" * 60)

        initial_state = {
            "query": question,
            "chat_history": chat_history or [],
            "documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            # 新增字段
            "sub_queries": [],
            "rerank_scores": [],
            "is_hallucination": False,
            "retry_count": 0,
            "rewritten_query": question,
            "hallucination_feedback": None,
        }

        result = self.graph.invoke(initial_state)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "sub_queries": result.get("sub_queries", []),
            "rewritten_query": result.get("rewritten_query", question),
            "rerank_scores": result.get("rerank_scores", []),
        }


