'''
Description: 文档加载器: 使用 MarkItDown 统一转为 markdown
Author: zyq
Date: 2026-01-05 14:43:09
LastEditors: zyq
LastEditTime: 2026-01-06 09:06:40
'''

from __future__ import annotations

from typing import Tuple, Dict
from pathlib import Path
from loguru import logger
from markitdown import MarkItDown


class DocumentLoader:
    def __init__(self):
        self._md = MarkItDown()

    def load(self, path: str) -> Tuple[str, Dict[str, str]]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        logger.info(f"加载文档: {path}")
        res = self._md.convert(str(file_path))
        text = res.text_content or ""
        meta = {"source_path": str(file_path), "title": file_path.name}
        return text, meta
