 # RAG 检索增强生成问答系统（前后端分离 · 可 Docker 部署）

> 基于 LangChain + LangGraph 的 RAG 问答系统：Streamlit 前端 + FastAPI 后端 + Chroma 持久化向量库 + SQLite 会话元数据。

---

## 1. 项目简介

本项目从一份可运行的 RAG Demo 升级为生产级完整应用，覆盖 **文档解析 → 向量检索 → 生成 → 幻觉自检** 全链路，并引入多种检索增强策略（Multi-Query 多路召回、Cross-Encoder 重排、基于 LLM 的幻觉自检与条件分支重试）。

| 维度 | 说明 |
|---|---|
| 检索增强 | Multi-Query 多路召回 + Cross-Encoder 重排 + 查询改写 |
| 可靠性 | LLM 幻觉自检 + LangGraph 条件分支重试（限次防死循环） |
| 持久化 | Chroma 向量库磁盘持久化 + SQLite 会话/文档元数据 |
| 架构 | Streamlit(纯 HTTP 客户端) ↔ FastAPI REST ↔ RAG 核心 |
| 部署 | Docker 容器化，前后端一键启动 |
| 降级容错 | 无 API 密钥 / 无 GPU 时仍可完整演示全流程 |

## 2. 系统架构

```
+───────────────────+      HTTP/JSON      +──────────────────────────+
│  Streamlit 前端    │ ──────────────────> │  FastAPI 后端 (api.py)    │
│  (frontend.py)     │  <───────────────── │  /api/* · 自动生成 /docs  │
│  纯 HTTP 客户端     │                    +───────────┬──────────────+
+───────────────────+                                │ 调用
                                                     ▼
                                          ┌─────────────────────┐
                                          │  RAG 核心 (main.py)   │
                                          │  LangChain+LangGraph  │
                                          └──────┬────────┬──────┘
                                                 │        │
                           Chroma 向量库(磁盘持久化) │        │ SQLite(会话/文档元数据)
```

**LangGraph 状态图流程（7 节点 + 条件边）**：

```
START → process_query → multi_query_node → retrieve → rerank_node → generate → hallucination_check_node
                                                                                  │
          is_hallucination==True → 回 generate 重试(≤ max_retry 次) ───────────────┤
                                                                                  ▼
          is_hallucination==False → evaluate → END
```

### LangGraph 条件边幻觉重试逻辑

1. `hallucination_check_node` 用 LLM 判断回答是否与上下文冲突或编造内容，输出 `is_hallucination`。
2. 条件边 `route_after_hallucination`：为 `True` 则跳回 `generate` 重新生成；`False` 则进入 `evaluate`。
3. **防死循环**：`retry_count ≥ max_retry`（默认 2）时强制置 `False` 放行，不无谓重试。

## 3. 功能特性

- ✅ **多格式解析**：`.pdf / .txt / .md / .json / .csv` 文档上传（pypdf 提取 PDF，编码自动回退 utf-8/gbk）
- ✅ **文档管理**：上传文档自动分块入库，支持删除（同步清理 Chroma 向量与 SQLite 元数据）
- ✅ **多轮对话 & 查询改写**：结合对话历史消除指代歧义，支持新建/切换/删除会话
- ✅ **Multi-Query 多路召回**：LLM 从多角度生成子查询，扩大召回覆盖
- ✅ **Cross-Encoder 重排**：对召回结果打分重排，提升 top 相关性
- ✅ **幻觉自检重试**：条件分支重生成，降低幻觉回答
- ✅ **来源引用 & 置信度打分**：回答附带来源文档片段与置信度
- ✅ **调试面板**：展示改写后 Query、Multi-Query 子查询、Rerank 打分明细（JSON 树形）
- ✅ **数据持久化**：Chroma 向量库 + SQLite 会话/文档元数据
- ✅ **前后端分离**：Streamlit 为零依赖 HTTP 客户端，不耦合 RAG 内部
- ✅ **异常容错**：LLM/向量库/文件解析异常全部捕获；无密钥自动降级

## 4. 环境部署

### 4.1 配置 `.env`

```bash
cp .env.example .env   # Linux/macOS
copy .env.example .env # Windows
# 填入真实密钥（不填则以"降级演示模式"运行）
```

关键配置项：
```ini
DEEPSEEK_API_KEY=your_DEEPSEEK_API_KEY_here      # 必填才可生成真实回答（硅基流动密钥）
OPENAI_API_KEY=your_siliconflow_api_key_here     # 必填才可使用高质量 Embedding（硅基流动密钥，与上者通常相同）
CHUNK_SIZE=500         # 分块大小
CHUNK_OVERLAP=100      # 分块重叠
TOP_K=3                # 检索返回 top-k
MAX_RETRY=2            # 幻觉最大重试次数（防死循环）
ENABLE_MULTI_QUERY=1   # 功能开关
ENABLE_RERANKER=1
```

### 4.2 本地运行（推荐虚拟环境）

```bash
python -m venv venv
venv\Scripts\activate   # Windows；Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# 终端 1：启动 FastAPI 后端（自动生成 /docs 接口文档）
uvicorn api:app --reload --port 8000

# 终端 2：启动 Streamlit 前端
streamlit run frontend.py --server.port 8501
```

浏览器访问 `http://localhost:8501`（前端）、`http://localhost:8000/docs`（接口文档）。

> `main.py` 为 RAG 核心模块（被 api.py 引用），无独立入口；如需在本地以脚本方式自测，可写独立脚本 `from main import RAGChain, RAGConfig` 后调用。

### 4.3 Docker 部署

```bash
docker build -t rag-qa-system .
docker run -p 8000:8000 -p 8501:8501 --env-file .env rag-qa-system
```

容器内同时启动 uvicorn(8000) 与 streamlit(8501)。

## 5. 模块说明

| 模块 | 职责 |
|---|---|
| `main.py` | **RAG 核心**：`DocumentProcessor`(Chroma 持久化、`delete_documents_by_doc_id` 两级匹配删除)、`Retriever`(单路/多路召回)、`Generator`(生成/改写/置信度/幻觉检测)、`RAGChain`(LangGraph 7 节点条件边图)、`RAGState(TypedDict)` |
| `config.py` | 全局配置：读取 `.env`，集中管理路径、超参、模型名 |
| `api.py` | FastAPI 后端：`/api/doc/upload`、`/api/doc/list`、`/api/doc/{doc_id}`(DELETE)、`/api/chat`、`/api/session`、`/api/session/{id}/history`、`/api/session/{id}`(DELETE)、`/health` |
| `frontend.py` | Streamlit 前端（纯 HTTP 客户端，requests 调后端） |
| `utils/file_loader.py` | 文件解析：PDF/TXT/MD → `Document`，异常捕获 |
| `utils/reranker_helper.py` | Multi-Query 生成 + 重排（硅基流动 /rerank API 优先，本地 CrossEncoder 兜底） |
| `utils/logger.py` | 统一日志（控制台 + 可选文件） |
| `db/sqlite_db.py` | SQLite 三表：`sessions`/`messages`/`uploaded_docs`（仅元数据，不存向量） |
| `db/chroma_db/` | Chroma 向量库持久化目录 |

### 5.1 文档与会话管理

**文档删除（`DELETE /api/doc/{doc_id}`）**：
- 上传时生成 `doc_id` 并注入每个 chunk 的 metadata，建立 SQLite 文档与 Chroma 向量的关联。
- 删除时采用**两级匹配**：优先按 `doc_id` 精确删除 Chroma 向量；若未命中（兼容早期未注入 doc_id 的历史数据），则按文件名兜底匹配 source 字段删除。
- 向量删除后同步清理 SQLite 元数据记录，确保两侧一致。

**会话删除（`DELETE /api/session/{session_id}`）**：
- 先删除该会话的全部消息记录，再删除会话本身（messages 表外键未声明 CASCADE，需显式清理）。
- 前端删除后自动清空当前会话状态，避免引用已删除的 session_id。

## 6. RAG 优化策略

1. **查询改写（Query Rewrite）**：多轮对话中，"它" "那个" 等指代通过 LLM 改写为独立完整查询，提升检索命中率（`Generator.rewrite_query`）。
2. **Multi-Query 多路召回**：一个问题生成多个角度子查询，分别检索后去重合并，扩大召回覆盖，缓解单查询意图单一问题（`reranker_helper.multi_query_generate`）。
3. **重排（Reranker）**：召回（Bi-Encoder）追求速度和召回率但打分粗；重排对 query-doc 精打分，提升 top-K 相关性。优先走硅基流动 `/rerank` API（模型 `BAAI/bge-reranker-v2-m3`，零本地下载），不可用时降级本地 CrossEncoder（`reranker_helper.rerank_documents`）。
4. **幻觉自检条件分支**：生成后用 LLM 检测回答是否与上下文冲突或编造；发现幻觉则通过 **LangGraph 条件边** 跳回重生成，并有 `max_retry` 上限防死循环。

## 7. 调优实验

| 参数 | 影响 | 经验值 |
|---|---|---|
| `chunk_size` | 太小→上下文碎片化丢失语义；太大→引入无关内容。中文长文建议 300-500 | 300~500 |
| `chunk_overlap` | 重叠保证跨块语义连贯，防止关键句被硬切分 | 50~100 |
| `top_k` | 太少→漏答；太多→上下文膨胀干扰模型。配合重排后 top_k 可适当加大 | 3~5（+重排） |
| `multi_query_num` | 子查询越多召回越广，但 LLM 调用成本↑ | 3 |
| `max_retry` | 幻觉重试上限，过高则耗时/成本↑ | 2~3 |

建议实验方法：固定其他参数，单一变量扫描，用来源引用率 / 置信度 / 人工评判对比效果，记录成表格用于效果评估。

## 8. 项目局限与未来改进

- **Embedding 质量**：默认 SimpleEmbeddings（哈希）仅演示，生产需通过硅基流动（OpenAI 兼容协议）或 HuggingFace 使用真实语义向量。
- **重排模型**：当前重排走硅基流动 `/rerank` API（`BAAI/bge-reranker-v2-m3`），不依赖本地模型下载；如需完全离线部署，需预下载 CrossEncoder 模型或换 CPU 友好的小模型。
- **查询策略可扩展**：可加入 MMR（最大边际相关）去冗余、HyDE（假设文档）增强召回。
- **会话与记忆**：当前用最近 4 轮做改写；可升级为向量化长期记忆或总结压缩。
- **评估**：可接入 RAGAS 测评框架、加入 golden pair 数据集做自动化效果评测。
- **并发与性能**：Streamlit 单进程；生产可采用 Celery/异步任务、缓存检索结果。
- **安全**：可加入文件类型白名单校验、敏感词过滤、上下文注入防护。