"""全局配置中心：所有路径、超参数、模型名与功能开关。

- 所有配置项从项目根目录的 .env 读取，均带默认值，缺失 .env 也能正常运行；
- 硬编码常量统一收敛到本模块，其他代码一律 `import config` 使用，不各自硬编码；
- 前端可通过 API 请求体临时覆盖部分运行参数（见 api.py），不会写回本模块。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录（config.py 所在目录），路径默认值均以此为锚，与进程 CWD 无关
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# ----------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(ROOT_DIR / "db" / "chroma_db"))  # Chroma 向量库持久化目录
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(ROOT_DIR / "db" / "rag.db"))     # SQLite 元数据库文件
LOG_FILE = os.getenv("LOG_FILE", str(ROOT_DIR / "logs" / "rag.log"))              # 日志落盘路径
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))                             # 上传文件大小限制（MB）

# ----------------------------------------------------------------------
# RAG 超参数
# ----------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))        # 分块大小（字符数）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # 分块重叠（字符数）
TOP_K = int(os.getenv("TOP_K", "3"))                    # 检索返回 top-k
MAX_RETRY = int(os.getenv("MAX_RETRY", "2"))            # 幻觉检测最大重试次数（防死循环）
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))    # 生成温度
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))       # 生成最大 token 数

# ----------------------------------------------------------------------
# 模型名
# ----------------------------------------------------------------------
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash")  # 生成模型
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "BAAI/bge-m3")    # Embedding 模型
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"
)                                                                            # 重排模型
MULTI_QUERY_NUM = int(os.getenv("MULTI_QUERY_NUM", "3"))                      # Multi-Query 子查询数量

# ----------------------------------------------------------------------
# API 密钥与接入点
# ----------------------------------------------------------------------
# LLM / Embedding 统一走 OpenAI 兼容协议（默认硅基流动 SiliconFlow）；
# 换其他平台只需改 OPENAI_API_BASE 与对应模型名。
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ----------------------------------------------------------------------
# 功能开关（默认开启；可由前端通过 API 参数临时覆盖，也可改 .env）
# ----------------------------------------------------------------------
ENABLE_MULTI_QUERY = os.getenv("ENABLE_MULTI_QUERY", "1").lower() in ("1", "true", "yes")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "1").lower() in ("1", "true", "yes")

# ----------------------------------------------------------------------
# 混合检索（BM25 稀疏检索 + 向量稠密检索，RRF 融合）
# ----------------------------------------------------------------------
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "1").lower() in ("1", "true", "yes")
BM25_CANDIDATE_K = int(os.getenv("BM25_CANDIDATE_K", "5"))  # BM25 每路召回候选数
RRF_K = int(os.getenv("RRF_K", "60"))                       # RRF 融合平滑常数（业界默认 60）
