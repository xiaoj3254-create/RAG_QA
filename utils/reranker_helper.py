"""Reranker 工具：Multi-Query 多查询生成 + Cross-Encoder 重排序。

提供两个能力：
1. multi_query_generate —— 调用 LLM 从多个角度生成检索子查询，扩大召回覆盖；
   失败时回退为 [原始查询]，保证主链路可用；
2. rerank_documents —— 对召回文档做 query-doc 精打分重排，提升 top-K 相关性。

重排采用三级降级链，任何一级失败自动落入下一级，主链路不中断：
1. 本地 sentence-transformers CrossEncoder（首选：无网络依赖、无 API 配额与费用，
   单例缓存，进程内只加载一次；模型需已本地缓存或可联网下载）；
2. 硅基流动 /v1/rerank HTTP API（本地模型未安装 / 加载失败 / 推理失败时的兜底；
   配置的模型名不可用时自动切换平台可用的 BAAI/bge-reranker-v2-m3）；
3. 保持召回原顺序返回 docs[:top_n]。

设计要点：本地 sentence-transformers 依赖 torch，属重型依赖，函数内懒加载，
未安装时直接跳过第 1 级改用云端 API，不影响主链路。
"""

import threading
from typing import List, Optional

import requests

import config
from langchain_core.documents import Document
from utils.logger import setup_logger

logger = setup_logger("reranker", config.LOG_FILE)

# 本地 CrossEncoder 单例缓存：按 model_name 键存，进程内只加载一次，
# 避免每次问答重复实例化模型（含首次联网下载）。
_local_model_cache: dict = {}
# 加载失败的模型名集合：本地重排是首选路径，若依赖缺失或模型下载失败，
# 缓存失败结果可避免"每次问答都重试一次加载/下载"造成请求阻塞，
# 后续请求直接走云端 API 兜底（重启进程可重置，便于装好依赖后重新启用）。
_local_model_failed: set = set()
_local_model_lock = threading.Lock()  # 串行化首次加载，防止并发请求重复实例化模型


def multi_query_generate(original_query: str, llm, num_queries: int = 3) -> List[str]:
    """调用 LLM 从不同角度生成 num_queries 个检索子查询。

    Args:
        original_query: 原始问题。
        llm: LangChain ChatModel 实例。
        num_queries: 生成的子查询数量。

    Returns:
        子查询列表；失败时回退为 [original_query]，保证主链路可用。
    """
    prompt = (
        f"你是检索查询专家。针对原始问题，从不同角度生成 {num_queries} 个用于向量检索的子查询，"
        f"使其相互补充、覆盖更多信息。\n"
        f"只输出每行一个查询，不要编号，不要解释。\n\n"
        f"原始问题：{original_query}"
    )
    try:
        resp = llm.invoke(prompt)
        lines = [l.strip() for l in str(resp.content).splitlines() if l.strip()]
        # 过滤编号/列表前缀
        queries = [l for l in lines if not l[0].isdigit() and not l.startswith(("-", "*", "·"))]
        # 保底：保证子查询里包含原始问题；数量不足时用原问题补足
        merged = [original_query] + queries + [original_query] * num_queries
        return list(dict.fromkeys(merged))[:num_queries] if merged else [original_query]
    except Exception as e:
        logger.warning("multi-query 生成失败，回退原始查询: %s", e)
        return [original_query]


def rerank_documents(
    query: str,
    docs: List[Document],
    top_n: int,
    model_name: Optional[str] = None,
) -> List[Document]:
    """对检索结果重排序，返回前 top_n 个文档。

    重排策略从高到低，任何一级失败都降级：
    1. 本地 sentence-transformers CrossEncoder（首选，单例缓存；模型需已在本地
       缓存或可联网下载）；
    2. 硅基流动 /v1/rerank HTTP API（本地不可用时的兜底，零本地下载）；
    3. 原顺序 docs[:top_n]。

    Args:
        query: 查询文本。
        docs: 召回阶段的文档列表。
        top_n: 重排后返回的数量。
        model_name: 重排模型名（本地模型名，同时作为 API 候选模型名）。
            留空则取 config.RERANKER_MODEL_NAME（即 .env 配置），**不硬编码默认模型**，
            避免绕过调用方时与 .env 配置不一致。

    Returns:
        重排后的文档列表；全部失败时降级返回 docs[:top_n]。
    """
    if not docs:
        return docs

    # 运行时解析（而非形参默认值）：确保读到的始终是当前 config 值
    model_name = model_name or config.RERANKER_MODEL_NAME

    ranked = _rerank_local(query, docs, top_n, model_name)
    if ranked is not None:
        return ranked

    ranked = _rerank_via_api(query, docs, top_n, model_name)
    if ranked is not None:
        return ranked

    logger.warning("本地与云端重排均不可用，保持召回原顺序返回")
    return docs[:top_n]


def _rerank_via_api(
    query: str,
    docs: List[Document],
    top_n: int,
    model_name: str,
) -> Optional[List[Document]]:
    """走硅基流动 /v1/rerank HTTP API 重排（第二级兜底）。

    无有效 key 或请求失败返回 None，交由上层降级到原顺序。模型候选为
    [配置的 model_name, BAAI/bge-reranker-v2-m3] 去重后逐个尝试，
    规避「配置了平台不存在的模型名」导致整体失败。
    """
    key = config.OPENAI_API_KEY
    if not key or key.startswith("your_"):
        return None

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    candidates = list(dict.fromkeys([model_name, "BAAI/bge-reranker-v2-m3"]))
    endpoint = config.OPENAI_API_BASE.rstrip("/") + "/rerank"
    last_err: Optional[str] = None

    for m in candidates:
        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                # 不传 top_n：请求全量打分，本地再截取，保证打分覆盖全部候选文档
                json={"model": m, "query": query, "documents": [d.page_content for d in docs]},
                timeout=30,
            )
        except Exception as e:
            last_err = f"请求失败: {e}"
            continue

        if resp.status_code != 200:
            last_err = f"HTTP {resp.status_code}: {resp.text[:120]}"
            continue

        try:
            data = resp.json()
        except ValueError:
            last_err = "响应不是合法 JSON"
            continue

        results = data.get("results") or []
        if not results:
            last_err = "响应无 results 字段"
            continue

        # index -> relevance_score 映射，按分数降序重排原文档
        score_map = {r.get("index"): r.get("relevance_score") for r in results}
        ranked = sorted(
            range(len(docs)),
            key=lambda i: score_map.get(i) if score_map.get(i) is not None else 0.0,
            reverse=True,
        )
        out = []
        for i in ranked[:top_n]:
            score = score_map.get(i)
            docs[i].metadata["rerank_score"] = round(float(score), 4) if score is not None else None
            out.append(docs[i])
        logger.info("硅基流动 rerank(%s) 完成，返回 top-%d", m, len(out))
        return out

    logger.warning("硅基流动 rerank 不可用: %s", last_err)
    return None


def _get_local_reranker(model_name: str):
    """获取本地 CrossEncoder 单例（懒加载 + 按模型名缓存）。

    首次调用时加载 sentence-transformers 并实例化模型写入模块级缓存；
    后续调用直接返回缓存实例，避免重复加载/下载。

    依赖缺失或加载失败返回 None（并记入失败缓存，避免后续请求反复重试），
    此时上层会自动降级到云端 rerank API。
    """
    if model_name in _local_model_cache:
        return _local_model_cache[model_name]
    if model_name in _local_model_failed:
        return None  # 此前已判定不可用，直接走云端兜底
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        logger.warning("sentence-transformers 未安装，重排降级云端 API: %s", e)
        _local_model_failed.add(model_name)
        return None
    with _local_model_lock:
        # 双检：等锁期间可能已被并发请求加载完成
        if model_name in _local_model_cache:
            return _local_model_cache[model_name]
        try:
            model = CrossEncoder(model_name)  # CPU 可推理；模型需已在本地缓存或可联网下载
            _local_model_cache[model_name] = model
            logger.info("本地重排模型已加载并缓存: %s", model_name)
            return model
        except Exception as e:
            logger.error("本地重排模型加载失败，重排降级云端 API: %s", e)
            _local_model_failed.add(model_name)
            return None


def _rerank_local(
    query: str,
    docs: List[Document],
    top_n: int,
    model_name: str,
) -> Optional[List[Document]]:
    """本地 sentence-transformers CrossEncoder 重排（首选路径）。

    模型未安装 / 加载失败 / 推理异常时返回 None，交由上层降级到云端 API。
    """
    model = _get_local_reranker(model_name)
    if model is None:
        return None

    try:
        import numpy as np
        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
        # 形状防御：部分 Cross-Encoder 返回 (n,1) 二维数组或标量，统一拍平成
        # 一维，否则 zip+sort+float() 会因数组比较/转换报错
        scores = np.asarray(scores).reshape(-1).tolist()
        ranked = list(zip(docs, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        # 打分写入 metadata，供前端调试面板展示
        for d, s in ranked:
            d.metadata["rerank_score"] = round(float(s), 4)
        logger.info("本地 CrossEncoder rerank(%s) 完成，返回 top-%d", model_name, top_n)
        return [d for d, _ in ranked[:top_n]]
    except Exception as e:
        logger.error("本地重排失败，降级云端 API: %s", e)
        return None
