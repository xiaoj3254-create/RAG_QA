"""全局配置：集中管理所有路径、超参数与模型名，从 .env 读取。

所有硬编码常量迁移到此模块；代码各处 `import config` 使用。
字段全部带默认值，缺失 .env 时也能正常运行。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录（config.py 所在目录）
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# ----------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(ROOT_DIR / "db" / "chroma_db"))
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(ROOT_DIR / "db" / "rag.db"))
LOG_FILE = os.getenv("LOG_FILE", str(ROOT_DIR / "logs" / "rag.log"))  # 日志落盘路径
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))  # 上传文件大小限制(MB)

# ----------------------------------------------------------------------
# RAG 超参数
# ----------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))       # 分块大小
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100")) # 分块重叠
TOP_K = int(os.getenv("TOP_K", "3"))                   # 检索返回 top-k
MAX_RETRY = int(os.getenv("MAX_RETRY", "2"))           # 幻觉最大重试次数(防死循环)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# ----------------------------------------------------------------------
# 模型名
# ----------------------------------------------------------------------
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"
)
MULTI_QUERY_NUM = int(os.getenv("MULTI_QUERY_NUM", "3"))  # 多查询生成的子查询数量

# ----------------------------------------------------------------------
# API 密钥
# ----------------------------------------------------------------------
# LLM / Embedding 统一走硅基流动（SiliconFlow）；其他平台改此值即可
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ----------------------------------------------------------------------
# 功能开关（默认开启，可由前端通过 API 参数临时覆盖，也可改 .env）
# ----------------------------------------------------------------------
ENABLE_MULTI_QUERY = os.getenv("ENABLE_MULTI_QUERY", "1").lower() in ("1", "true", "yes")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "1").lower() in ("1", "true", "yes")