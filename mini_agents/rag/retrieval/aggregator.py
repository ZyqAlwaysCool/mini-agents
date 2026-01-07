'''
Description: 去重与裁剪文本块召回结果 
Author: zyq
Date: 2026-01-05 14:49:37
LastEditors: zyq
LastEditTime: 2026-01-06 11:40:19
'''

from __future__ import annotations

from typing import List
from ..models import RetrievalHit


class Aggregator:
    def aggregate(self, hits: List[RetrievalHit], top_k: int) -> List[RetrievalHit]:
        if not hits:
            return []
        hits.sort(key=lambda x: x.score, reverse=True)
        deduped: List[RetrievalHit] = []
        seen = set()
        for h in hits:
            key = h.chunk.md5 or h.chunk.id
            if key in seen:
                continue
            seen.add(key)
            deduped.append(h)
            if len(deduped) >= top_k:
                break
        return deduped
