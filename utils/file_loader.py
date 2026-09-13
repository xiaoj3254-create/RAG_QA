"""文件解析工具：PDF / TXT / MD / JSON / CSV → LangChain Document 列表。

- 单个文件解析失败返回 None，批量加载时自动跳过，不中断整体流程；
- TXT/MD 按 utf-8 → gbk → latin-1 顺序回退解码；PDF 一页一个 Document；
- 所有 Document 携带 source 元数据（由底层 Loader 生成）。
"""
import os
from pathlib import Path
from typing import List, Optional, Union
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader, JSONLoader
import config
from langchain_core.documents import Document
from utils.logger import setup_logger

logger = setup_logger("file_loader", config.LOG_FILE)

# 全项目统一支持的文件后缀（与 api.py 上传白名单保持一致）
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".json", ".csv"}
_ENCODINGS = ("utf-8", "gbk", "latin-1")  # 中文优先 utf-8，失败回退 gbk


def load_file(path: Union[str, Path]) -> Optional[List[Document]]:
    """
    解析单个文件，失败返回 None（调用方批量自动跳过）
    返回：该文件解析出的Document列表；PDF一页一个Document
    """
    path = Path(path)

    if path.suffix.lower() not in SUPPORTED_EXTS:
        logger.warning("不支持的文件类型: %s", path.suffix)
        return None

    try:
        docs = _read_file(path)
        # 空文档判断：列表为空，或者所有doc的page_content都是空白
        valid_docs = [d for d in docs if d.page_content and d.page_content.strip()]
        if not valid_docs:
            logger.warning("文件无有效文本，跳过: %s", path.name)
            return None
        return valid_docs

    except Exception as e:
        logger.error("解析失败 %s: %s", path.name, str(e))
        return None


def load_files(paths: List[Union[str, Path]]) -> List[Document]:
    """批量加载文件，返回全部成功解析的文档平铺列表"""
    total_docs = []
    for p in paths:
        doc_list = load_file(p)
        if doc_list is not None:
            total_docs.extend(doc_list)
    return total_docs


def _read_file(path: Path) -> List[Document]:
    """分发加载器，读取文件返回原始Document列表"""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pdf_loader = PyPDFLoader(file_path=str(path))
        doc_list = pdf_loader.load()
        # 扫描版PDF告警：所有页面文本为空
        all_empty = all(not d.page_content.strip() for d in doc_list)
        if all_empty:
            logger.warning("PDF可能是扫描图片版，提取不到文本: %s", path.name)
        return doc_list

    elif suffix in (".txt", ".md"):
        last_err: Optional[UnicodeDecodeError] = None
        for enc in _ENCODINGS:
            try:
                text_loader = TextLoader(file_path=str(path), encoding=enc)
                return text_loader.load()
            except UnicodeDecodeError as e:
                last_err = e
                continue
        # 全部编码失败兜底（latin-1 单字节映射永不抛解码异常）
        if last_err:
            logger.warning("编码解析异常 %s，使用 latin-1 兜底: %s", path.name, str(last_err))
            return [Document(page_content=path.read_text(encoding="latin-1"), metadata={"source": str(path)})]

    elif suffix == ".json":
        json_loader = JSONLoader(
            file_path=str(path),
            jq_schema=".",
            text_content=False
        )
        return json_loader.load()

    elif suffix == ".csv":
        csv_loader = CSVLoader(file_path=str(path))
        return csv_loader.load()

    else:
        # load_file上层已经拦截后缀，理论不会走到这里
        raise ValueError(f"未处理的后缀 {path.suffix}")
