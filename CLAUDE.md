## 项目背景

已有一份可运行的 RAG Demo 代码（main.py）：

- 使用 LangChain + LangGraph；已有 `RAGChain / DocumentProcessor / Retriever / Generator` 类；
- 当前使用 `InMemoryVectorStore`（内存向量库，重启丢失）；
- 内置简单演示用 `SimpleEmbeddings`；
- 通过硬编码样本文档 SAMPLE_DOCUMENTS，控制台演示问答；
- LangGraph 目前是线性串行流程：`process_query → retrieve → generate → evaluate`；
- 支持查询改写、多轮对话、来源输出、置信度打分；
- **限制：没有持久化、没有文件解析、没有API、没有Web界面、没有reranker/multi‑query/幻觉校验。**

**目标：改造升级为生产级完整 RAG 项目，不要全盘重写原有核心类，尽量复用已有业务逻辑，在此基础上扩展。** 项目定位：大模型应用方向完整项目；前后端分离；可 Docker 部署；Streamlit Web UI；FastAPI 后端；Chroma持久化向量库；SQLite存储会话元数据。

> ❗重要约束：不要抛弃原有 `RAGChain、DocumentProcessor、Retriever、Generator`，在原有类基础上扩展，而不是全部重写一套。

## 整体系统架构

```
Streamlit Web前端 <--> FastAPI REST服务 <--> RAG核心模块(改造后的main.py)
                              ↓
                ┌────────────┴────────────┐
                ↓                         ↓
        Chroma向量库(磁盘持久化)      SQLite数据库(会话/文档元数据)
                ↓
        DeepSeek LLM(硅基流动·OpenAI兼容协议) / 硅基流动 BAAI/bge-m3 Embeddings / 硅基流动 rerank API
```

## 项目目录结构（严格遵守）

```
rag_qa_system/
├── .env                      # 环境变量 DEEPSEEK_API_KEY, OPENAI_API_KEY, 路径配置
├── Dockerfile                # 容器部署
├── requirements.txt          # 完整依赖列表
├── README.md                 # 项目文档：架构、功能、部署、优化点、实验说明
├── main.py                   # ✅原有RAG核心代码，在此基础上改造，不要重写全部类
├── config.py                 # 全局配置：读取.env，定义路径、默认超参
├── api.py                    # FastAPI服务层
├── frontend.py               # Streamlit前端（调用FastAPI http接口，不直接调用RAG内部）
├── db/
│   ├── sqlite_db.py          # SQLite：会话表、消息表、上传文档元数据表
│   └── chroma_db/            # Chroma持久化向量库目录（git忽略）
└── utils/
    ├── file_loader.py        # PDF / TXT / MD 文件解析工具，输出LangChain Document列表
    ├── reranker_helper.py    # 重排序（硅基流动 rerank API 优先，本地CrossEncoder兜底）、Multi‑Query生成工具函数
    └── logger.py             # 统一日志工具
```

## 需要完成的改造&新增模块清单

### 1. config.py

职责：读取 `.env`，集中管理全部路径与参数

- CHROMA_DB_PATH = "./db/chroma_db"

- SQLITE_DB_PATH = "./db/rag.db"

- 默认RAG参数：chunk_size, chunk_overlap, top_k

- reranker模型名称、LLM生成模型名称（DeepSeek）、Embedding模型名称（BAAI/bge-m3）

  > 所有硬编码常量迁移到 config，代码各处导入 config 使用。

### 2. utils/file_loader.py

输入：本地文件路径 / 上传文件二进制；支持 `.txt .md .pdf .json .csv` 输出：`List[langchain_core.documents.Document]`，携带 source 元数据。

- PDF 使用 PyPDFLoader(pypdf) 提取文本；txt/md 编码自动回退（utf-8/gbk/latin-1）；json 走 JSONLoader(jq)；csv 走 CSVLoader；
- 增加异常捕获：损坏文件、空文件处理。

### 3. utils/reranker_helper.py

两个功能函数：

1. `multi_query_generate(original_query:str, llm, num_queries=3) -> list[str]` 输入原始问题，调用LLM生成多个角度的子查询，用于多查询召回；返回子查询列表。
2. `rerank_documents(query:str, docs:List[Document], top_n:int, model_name:str) -> List[Document]` 对检索得到文档做重排序：优先走硅基流动 /rerank API（模型 `BAAI/bge-reranker-v2-m3`），不可用时降级本地 sentence-transformers CrossEncoder，返回重排后的文档列表。

### 4. utils/logger.py

封装logging，统一日志格式；输出控制台+可选日志文件。

### 5. db/sqlite_db.py

SQLite三张表：

1. `sessions`：session_id(主键)、session_name、created_at
2. `messages`：id、session_id、role(user/assistant)、content、created_at
3. `uploaded_docs`：doc_id、file_name、chunk_count、upload_time 提供CRUD函数：

- create_session、get_all_sessions
- save_message、get_session_messages(session_id)
- add_uploaded_doc_meta、list_uploaded_docs

> 只负责元数据存储；向量依旧保存在Chroma，SQLite不存向量。

### 6. main.py 【重点改造，禁止完全重写原有类】

> 保留原有 `RAGConfig、DocumentProcessor、Retriever、Generator、RAGChain、RAGState(TypedDict)`

#### 修改点

1. DocumentProcessor：把原来 `InMemoryVectorStore` 替换为 **Chroma持久化向量库**

```
from langchain_chroma import Chroma
# persist_directory 使用 config.CHROMA_DB_PATH
```

1. 扩展 LangGraph 状态节点，**新增3个节点，并且增加条件边** 原有节点：`process_query、retrieve、generate、evaluate` 新增节点：

- `multi_query_node`：生成多条子查询，做多查询检索召回
- `rerank_node`：对retrieve输出文档做Cross‑Encoder重排序
- `hallucination_check_node`：幻觉校验节点：接收 query、context、answer，LLM判断回答是否存在幻觉/与上下文冲突；输出布尔值 `is_hallucination`

**LangGraph条件流转逻辑：**

```
START → process_query → multi_query_node → retrieve → rerank_node → generate → hallucination_check_node
        ↓
if is_hallucination == True → 跳回 generate 重新生成回答（最多重试2次）
if is_hallucination == False → evaluate → END
```

> 在RAGState TypedDict增加字段：`is_hallucination:bool, retry_count:int, sub_queries:List[str]`

1. Embedding逻辑：优先通过硅基流动（OpenAI 兼容协议）调用 BAAI/bge-m3；没有密钥回退SimpleEmbeddings；打印警告提示生产不要使用SimpleEmbeddings。
2. 完善异常捕获：LLM调用异常、向量库异常捕获。
3. RAGChain.index_documents() 支持接收文件路径列表，内部调用 utils.file_loader，不再只接收文本字符串。

### 7. api.py FastAPI接口层

> FastAPI 调用改造后的 RAGChain 对象，同时读写 sqlite_db.py；自动生成 `/docs` 接口文档。 接口清单：

1. `POST /api/doc/upload`：接收上传文件；解析文档；调用 rag.index_documents；保存文档元数据到sqlite；
2. `GET /api/doc/list`：获取已上传文档元数据列表
3. `POST /api/chat`：入参 session_id、question；内部调用 rag.query；保存问答消息到sqlite；返回 `{"answer","sources","confidence","sub_queries"}`
4. `GET /api/session`：获取全部会话列表
5. `POST /api/session`：新建会话，返回session_id
6. `GET /api/session/{session_id}/history`：获取会话历史消息

### 8. frontend.py Streamlit前端

> ⚠️ 重要：Streamlit只做HTTP客户端，通过requests调用FastAPI接口，**不要直接导入调用RAGChain内部类，实现前后端分离。**

页面布局：

- 侧边栏：
  1. API服务连通性状态提示
  2. 文件上传组件（支持多文件 pdf/txt/md），上传按钮调用上传接口
  3. RAG参数配置面板：chunk_size、chunk_overlap、top_k；开关：是否开启multi‑query、是否开启reranker
  4. 会话列表：新建会话、切换会话
- 主区域：
  1. Chat对话气泡渲染用户/助手消息
  2. 助手回答下方折叠面板：展示来源文档片段、置信度分数
  3. 调试折叠面板：展示改写后query、multi‑query生成的子查询、reranker打分信息
  4. 输入框发送问题

### 9. Dockerfile & requirements.txt

- Dockerfile：python基础镜像，拷贝代码，启动脚本同时启动uvicorn(fastapi后台)+streamlit前端
- requirements.txt：整理全部依赖 langchain、langgraph、langchain‑chroma、fastapi、uvicorn、streamlit、pypdf2、sentence‑transformers、python‑dotenv

### 10. README.md

需要包含：

1. 项目简介
2. 系统架构文本/ASCII图
3. 功能特性列表
4. 环境部署步骤：本地运行方式、docker运行方式；.env配置示例
5. 模块说明
6. RAG优化策略说明：Multi‑Query、Reranker、幻觉自检条件分支、查询改写
7. 调优实验说明：不同chunk_size、top‑k对效果的影响
8. 项目局限与未来改进方向

## 输出要求告诉 Claude

1. 输出完整可运行代码，按上面目录拆分各个文件；每个文件标明文件名；
2. **最大限度复用原始main.py已有类结构，不要全部推倒重写；只做扩展与替换向量存储；**
3. 代码中写必要注释；关键逻辑（LangGraph条件边幻觉重试）写注释；
4. .env提供模板 `.env.example`；

## 补充给 Claude 的提示（防止踩坑）

> 注意：
>
> 1. LangGraph幻觉重试要有最大重试次数，避免死循环；
> 2. Chroma实例注意不要重复实例化；
> 3. Streamlit和FastAPI分离，前端不要耦合RAG内部逻辑；
> 4. SimpleEmbeddings只作为兜底演示，代码打印警告，提示生产使用真实Embedding；
> 5. SQLite数据库启动自动建表；
> 6. 文件上传要做大小简单限制。

