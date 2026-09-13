"""统一日志工具：控制台输出 + 可选日志文件。

- 各模块通过 setup_logger(name) 获取独立 logger，统一格式；
- 重复调用返回已有实例，不会重复添加 handler 导致日志刷屏；
- Windows 下自动将控制台重配置为 UTF-8，避免中文/emoji 日志编码炸栈；
- 日志不向上传播到 root logger，避免与 uvicorn 等框架的日志重复输出。
"""

import logging
import os
import sys


def setup_logger(name: str = "rag", log_file: str = None, level: int = logging.INFO):
    """获取统一的 logger 实例。

    Args:
        name: logger 名称（自动补全为 rag_xxx 风格）。
        log_file: 可选日志文件路径；为 None 时只输出控制台。
        level: 日志级别。
    """
    logger = logging.getLogger(name)
    # 防止重复添加 handler（多次调用时日志刷屏）
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Windows 控制台默认 GBK，中文/emoji 无法编码会抛 UnicodeEncodeError，
    # 导致 uvicorn 等服务的中文日志炸栈。强制把 stdout/stderr 重配置为 UTF-8 兜底。
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except (OSError, ValueError):
                    pass  # 非文本流（如已替换为管道且不可重配置）时忽略

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # 防止日志向上传播到 root logger，避免重复打印
    logger.propagate = False
    return logger