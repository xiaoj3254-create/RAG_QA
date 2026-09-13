# Changelog

## 2026-09-13 — 重排降级链顺序调整（本地优先）

| # | 改动 | 文件 | 说明 |
|---|------|------|------|
| 1 | 重排降级链改为"本地优先" | [utils/reranker_helper.py](utils/reranker_helper.py) | `rerank_documents` 由「API 优先 → 本地兜底」改为 **① 本地 CrossEncoder → ② 硅基流动 `/rerank` API → ③ 原顺序**。`_rerank_local` 返回类型由 `List[Document]` 改为 `Optional[List[Document]]`：模型不可用或推理异常返回 `None` 交由上层降级，不再自行截断为 `docs[:top_n]`（否则会吞掉云端兜底机会） |
| 2 | 本地模型加载失败结果缓存 | [utils/reranker_helper.py](utils/reranker_helper.py) | 新增 `_local_model_failed` 集合。本地成为首选路径后，若依赖缺失或模型下载失败，原实现会在每次问答都重试一次加载/下载并阻塞请求；现失败即记忆，后续请求直接走云端兜底（重启进程可重置，便于装好依赖后恢复本地优先） |
| 3 | 同步注释与文档 | 多文件 | 模块 docstring、`main.py` 模块头与 `rerank_node` docstring、`README.md`（功能特性/模块表/优化策略/局限）、`项目分析文档.md`（技术栈表、目录树、函数解析、流程图、改进清单）统一改为本地优先表述 |

### 验证情况

- ✅ `py_compile` 语法检查通过（含 `main.py`、`reranker_helper.py`）
- ✅ 三场景实测（venv，屏蔽真实 API key 禁止联网）：本地+云端均不可用 → 原顺序 `docs[:3]`；本地可用（注入假模型）→ 按本地分数倒序重排且 `rerank_score` 正确写回 metadata；本地推理抛异常 → 降级云端 → 原顺序
- ✅ 临时验证脚本已删除

### 建议提交信息

```
refactor(rerank): prefer local CrossEncoder, fall back to cloud /rerank API

- swap fallback order in rerank_documents: local -> SiliconFlow API -> original order
- _rerank_local returns Optional[List[Document]] so cloud fallback is reachable
- cache failed local model loads to avoid retrying download on every request
```

---

## 2026-09-13 — 移除法条编号归一化

| # | 改动 | 文件 | 说明 |
|---|------|------|------|
| 1 | 删除法条编号归一化逻辑 | [utils/bm25_helper.py](utils/bm25_helper.py) | 移除 `_arabic_to_cn` / `_normalize_legal_numbers` / `_LEGAL_NUM_RE` / `_CN_DIGITS` 及 `import re`；`tokenize()` 恢复为直接 jieba 分词，不再做 `第6条`→`第六条` 转换。BM25 与向量双路 + RRF 融合检索本身不变 |

### 验证情况

- ✅ `py_compile` 语法检查通过
- ✅ `tokenize('刑法第6条是什么')` → `['刑法', '第', '6', '条是', '什么']`，不再转换
- ✅ 全项目无 `_normalize_legal_numbers` / `_arabic_to_cn` 残留引用

### 建议提交信息

```
refactor: remove legal article number normalization from BM25 tokenizer

- drop _arabic_to_cn/_normalize_legal_numbers and re import from utils/bm25_helper.py
- tokenize() now uses plain jieba segmentation; hybrid retrieval pipeline unchanged
```

---

## 2026-09-12 — BM25 混合检索代码审查修复（3 处，3 个文件）

对"BM25 混合检索"改动做 code-review 发现的 3 个问题（2 名验证员独立确认，均为 minor），全部已修复。问题 1、2 的竞态在当前 FastAPI 单例 + `_config_lock` 全串行接线下不可达，属模块自身线程安全承诺未兑现的加固项；问题 3 无功能影响，属配置读取路径不一致。

### 修复内容

| # | 问题 | 文件 | 修复 |
|---|------|------|------|
| 1 | `BM25Helper.search()` 在锁释放后读 `_bm25`/`_docs`，而索引重建时先赋 `_bm25` 后赋 `_docs`——并发下可读到"新索引+旧文档"的错位组合（静默错误结果），或 `_bm25=None` 触发 AttributeError | [utils/bm25_helper.py](utils/bm25_helper.py) | `_ensure_index()` 改为返回锁内取得的 `(bm25, docs)` 快照；`search()` 锁外只用快照检索，快照内部严格按行对齐，invalidate/重建不再影响进行中的检索 |
| 2 | `_get_local_reranker` 对模块级缓存 check-then-act 无锁，并发首调用会各自 `CrossEncoder()` 加载 GB 级模型，一份被丢弃，与"进程内只加载一次"的注释意图不符 | [utils/reranker_helper.py](utils/reranker_helper.py) | 新增 `_local_model_lock`，锁内二次检查缓存（双检锁），并发首调用只加载一次 |
| 3 | `retrieve`/`retrieve_multi` 中 `rrf_k=config.RRF_K` 走模块级配置，相邻的 `bm25_candidate_k`/`top_k` 均走 `self.config`（RAGConfig），且 RAGConfig 缺 `rrf_k` 字段，参数无法按实例覆盖 | [main.py](main.py) | `RAGConfig` 新增 `rrf_k: int = config.RRF_K` 字段；两处 RRF 融合统一改走 `self.config.rrf_k` |

### 验证情况

- ✅ 三个改动文件 `ast` 语法检查通过
- ✅ 冒烟测试（venv + 20 篇模拟语料）：法条编号归一化双端生效（`第6条`→`第六条` 命中原文）；invalidate 后旧快照检索仍返回对齐文档；失效后新检索自动重建；RRF 双路命中的文档排名第一
- ⚠️ 小语料（N≤2）时 BM25Okapi 的 IDF 恒为 0（ln(1.5/1.5)）、所有得分被 >0 过滤属 rank-bm25 数学特性，非本次缺陷；真实库 150+ chunks 不受影响
- ✅ 临时测试脚本已删除，向量库零测试残留

### 建议提交信息

```
fix: hardening from BM25 hybrid search code review

- snapshot (bm25, docs) under lock in BM25Helper to avoid index/doc skew
- double-checked locking for local CrossEncoder singleton cache
- add rrf_k field to RAGConfig; Retriever reads via self.config
```

---

## 2026-09-12 — BM25 混合检索（BM25 稀疏 + 向量稠密 + RRF 融合）

背景：纯向量检索对"第6条"这类编号/关键词查询召回不准——语义相似度对数字编号无区分度。引入 BM25 稀疏检索补足字面精确匹配，两路结果经 RRF 融合，兼顾语义相关性与关键词命中。

### 改动内容

| # | 改动 | 文件 | 说明 |
|---|------|------|------|
| 1 | 新建 BM25 稀疏检索辅助模块 | [utils/bm25_helper.py](utils/bm25_helper.py) | `BM25Helper` 从 Chroma 语料懒构建索引（rank-bm25 + jieba 分词，与向量检索同一份数据）；`rrf_fuse` 按名次贡献 1/(rrf_k+rank) 融合多路排序；依赖缺失/索引失败/检索异常均返回空列表，主链路自动回退纯向量 |
| 2 | 法条编号归一化 | [utils/bm25_helper.py](utils/bm25_helper.py) | 检索前把 `第6条` → `第六条`（覆盖条/款/章/项，1~999），消除用户查询（阿拉伯数字）与法条原文（汉字数字）风格不一致导致的 BM25 分词完全错开 |
| 3 | Retriever 支持混合检索 | [main.py](main.py) | `retrieve`/`retrieve_multi` 均改为 向量+BM25 双路召回 + RRF 统一融合（多子查询跨路共识排名）；`ENABLE_HYBRID_SEARCH=0` 时保留原纯向量逻辑 |
| 4 | BM25 索引生命周期管理 | [main.py](main.py) / [api.py](api.py) | `BM25Helper` 归 `DocumentProcessor` 持有（语料写入方负责失效）；`process`/`delete_documents_by_doc_id` 后自动 `invalidate()`，下次检索按需重建（另有 chunk 数脏检测兜底）；api.py 上传后重建 Retriever 时传入同一实例，避免索引失联 |
| 5 | 混合检索配置项 | [config.py](config.py) | `ENABLE_HYBRID_SEARCH`（默认开）、`BM25_CANDIDATE_K=5`（BM25 每路候选数）、`RRF_K=60`（融合平滑常数） |
| 6 | 依赖清单 | [requirements.txt](requirements.txt) | 添加 `rank-bm25>=0.2.2`、`jieba>=0.42` |

### 验证情况

- ✅ 全部改动文件 `py_compile` 语法检查通过
- ✅ 归一化：`tokenize('第6条')` 与 `tokenize('第六条')` 分词一致（共享 token `第六条`）
- ✅ 编号类查询"刑法第6条是什么"：BM25 top1 精准命中第六条；混合融合 top1 命中，候选扩至 7 个供重排筛选
- ✅ 语义类查询（"在中国船只飞机里犯罪适用哪国法律"）：混合 top1 命中含"船舶"的第六条，无劣化
- ✅ 增删文档后索引自动同步：删除测试文档后 BM25 重建（157→152 chunks）
- ✅ 测试脚本已删除，向量库零测试残留

### 建议提交信息

```
feat: hybrid retrieval with BM25 sparse search + vector dense search fused by RRF

- add utils/bm25_helper.py: lazy BM25 index over Chroma corpus (rank-bm25 + jieba)
- normalize legal article numbers (第6条 -> 第六条) to fix BM25 token mismatch
- fuse vector + BM25 result lists via Reciprocal Rank Fusion (k=60)
- invalidate BM25 index on document add/delete; share instance across Retriever rebuilds
- add ENABLE_HYBRID_SEARCH / BM25_CANDIDATE_K / RRF_K config switches
```

---

## 2026-09-01 — 代码审查修复（6 处，6 个文件）

源自一轮 code-review 发现的真实问题，全部已修复。

### 修复内容

| # | 问题 | 文件 | 修复 |
|---|------|------|------|
| 1 | 运行时产物被提交进 git：`logs/rag.log` 已捕获用户查询与错误堆栈；`db/rag.db-wal`/`db/rag.db-shm` 是 SQLite WAL 瞬时文件，`.gitignore` 未覆盖，全新 clone 时主库缺失而 WAL 存在会引发 DB 损坏 | [.gitignore](.gitignore) | 补 `logs/`、`db/rag.db-wal`、`db/rag.db-shm`；已提交的 3 个文件待 `git rm --cached` 从跟踪移除（保留磁盘文件） |
| 2 | 维度自愈逻辑失效：`_check_embedding_compatibility` 只 `store.delete(ids)` 清空向量，collection 的 HNSW 维度元数据仍停留在旧维度，检索依旧报 `expecting embedding with dimension of X`（提交的日志已证实例） | [main.py](main.py) | 改为 `delete_collection` 后按当前 Embedding 重建 `Chroma` 实例，彻底重置维度元数据 |
| 3 | 置信度恒 0.5：DeepSeek-V4-Flash（推理模型）在数值答案里带 `…` 思考块，`float()` 每次抛异常走兜底 | [main.py](main.py) | 解析前先剥离 HTML 风格标签，再 `re` 正则提取首个浮点数并 clamp 到 [0,1]；新增 `import re` |
| 4 | `.env.example` 模板缺失：README（第 73-74 行）让用户 `cp/copy .env.example .env`，但文件从未创建 | [.env.example](.env.example) | 新建模板，覆盖 config.py 全部环境变量，含中文注释与降级演示模式说明 |
| 5 | `get_rag()` 首请求竞态：并发首请求都看到 `_rag` 为 None，各自独立构建 RAGChain（两个 DocumentProcessor/Chroma 实例、两次维度清理），后写者覆写单例 | [api.py](api.py) | 首初始化纳入 `_config_lock`，双检锁 + 锁内二次判空 |
| 6 | `utils/logger.py` 一段 4 行注释块重复两遍，描述同一 Windows stdout 重配置意图 | [utils/logger.py](utils/logger.py) | 删除重复块 |

### 验证情况

- ✅ 全部改动文件（main.py / api.py / config.py / utils/logger.py）`ast` 语法检查通过
- ⚠️ 维度自愈重建 collection 使用了 chromadb 私有属性 `store._client` / `store._name`，需在装有 langchain-chroma 的环境实测

### 建议提交信息

```
fix: code review findings (runtime artifacts, dimension heal, confidence, .env.example)

- gitignore logs/ and sqlite WAL/SHM; drop tracked runtime artifacts (pending git rm --cached)
- rebuild Chroma collection on embedding dimension mismatch instead of only deleting vectors
- strip thinking-block tags before parsing LLM confidence score (DeepSeek reasoning models)
- add missing .env.example template
- serialize get_rag() first init under config lock (double-checked locking)
- dedupe logger.py Windows stdout comment block
```

---

## 2026-08-31 — 代码审查修复（9 处，4 个文件）

源自一轮 code-review 发现的真实问题，全部已修复。

### 修复内容

| # | 问题 | 文件 | 修复 |
|---|------|------|------|
| 1 | 前端 chunk_size/chunk_overlap 参数覆盖无效（text_splitter 存的是初始值，改 config 不生效） | [main.py](main.py) | 新增 `DocumentProcessor.rebuild_text_splitter()`，按当前 config 重建 splitter |
| 2 | SimpleEmbeddings(384) ↔ OpenAIEmbeddings(1536) 切换后与持久化 Chroma 维度不匹配，检索静默返空 | [main.py](main.py) | 新增 `_check_embedding_compatibility()`：启动自检维度，不一致时告警并清空 collection 分批重删（1000 条/批） |
| 3 | 幻觉重试是空转：跳回 generate 但 context 相同，仅靠 LLM 随机采样 | [main.py](main.py) | `RAGState` 新增 `hallucination_feedback`；`Generator.generate()` 支持 feedback 参数注入 SystemMessage 纠偏 |
| 4 | SQLite 并发 `database is locked`（FastAPI 线程池多请求同时写） | [db/sqlite_db.py](db/sqlite_db.py) | `_connect()` 加 `timeout=15` + `PRAGMA foreign_keys=ON` + WAL 模式 |
| 5 | 外键未强制，孤儿消息；chat 传不存在的 session_id 打穿 FK | [db/sqlite_db.py](db/sqlite_db.py) / [api.py](api.py) | FK 真正开启；chat 前置校验返回 404 |
| 6 | 历史消息丢失 meta，刷新后来源/置信度/调试面板消失 | [db/sqlite_db.py](db/sqlite_db.py) / [api.py](api.py) | `messages` 表新增 `meta` 列（含旧库 ALTER 迁移）；`save_message` 写 JSON，`get_session_messages` 反序列化；chat 保存时附带 sources/confidence/sub_queries/rerank_scores |
| 7 | chat 覆盖全局单例 config 后从不恢复，参数跨请求/跨会话泄漏 | [api.py](api.py) | 新增 `_config_lock` 互斥锁；快照 → 覆盖 → finally 恢复（含 splitter 同步重建） |
| 8 | reranker `predict()` 对部分模型返回 (n,1) 二维或标量，zip/float 崩溃 | [utils/reranker_helper.py](utils/reranker_helper.py) | `np.asarray(scores).reshape(-1).tolist()` 拍平为一维 |
| 9 | tmp 目录按 CHROMA_DB_PATH 推断反向解析，依赖进程 CWD 而分叉；`file.filename` 可能为 None | [api.py](api.py) | tmp 锚定 `config.ROOT_DIR/tmp`；filename 空值兜底；上传的重关联/建图纳入锁 |

### 未改动

- [frontend.py](frontend.py) — sqlite 返回格式天然兼容前端渲染逻辑
- [utils/file_loader.py](utils/file_loader.py) — 异常捕获/编码回退本就完整
- `.gitignore` — 已覆盖 `tmp/`、`db/chroma_db/`、`db/rag.db`

### 验证情况

- ✅ 全部改动文件 `ast` 语法检查通过
- ✅ sqlite 层（meta 迁移、CRUD、外键、锁）本机实测通过
- ⚠️ 本机未安装 `langchain`，main.py 的检索/重排/Chroma 链路未实测，需在有依赖环境或 Docker 中验证

### 建议提交信息

```
fix: RAG system hardening from code review

- rebuild text_splitter on chunk param override (frontend sliders now effective)
- detect embedding dimension mismatch vs persisted Chroma vectors
- add SQLite busy timeout / WAL / FK enforcement + meta column migration
- persist assistant meta (sources/confidence/debug) for history reload
- serialize chat config override under lock with snapshot restore
- vary hallucination retry via LLM feedback (no futile identical retry)
- guard reranker score shape, tmp dir CWD dependency, filename None
```