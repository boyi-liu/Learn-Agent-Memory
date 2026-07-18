"""
策略三：向量检索 / RAG（玩具版）。

只取和当前问题“相关”的几条。可扩展到海量历史，但最复杂。

这里 embedding 用“词袋”、相似度用“词重叠”，纯离线、无依赖，用来讲清思想。
想要真实版（真 embedding 模型 + 真向量库），见 memory/rag_memory.py。
"""

import re
from typing import List, Dict

from .base import BaseMemory


def naive_embed(text: str) -> dict:
    """
    最朴素的“embedding”：把文本变成词袋 {词: 次数}。
    真实系统这里换成模型 embedding（一串浮点向量）。
    英文按词切；中文没空格，按单字切（否则整句变一个 token，永远匹配不上）。
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower()) + re.findall(r"[一-鿿]", text)
    bag = {}
    for w in tokens:
        bag[w] = bag.get(w, 0) + 1
    return bag


def similarity(a: dict, b: dict) -> float:
    """两个词袋的重叠度，代替真实的余弦相似度。"""
    common = set(a) & set(b)
    return sum(a[w] + b[w] for w in common)


class VectorMemory(BaseMemory):
    def __init__(self, path: str, top_k: int = 2):
        super().__init__(path)
        self.top_k = top_k  # 每次只取最相关的 k 条

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        q = naive_embed(query)
        scored = [(similarity(q, naive_embed(m["content"])), m) for m in self.messages]
        scored = [s for s in scored if s[0] > 0]        # 丢掉完全不相关的
        scored.sort(key=lambda x: x[0], reverse=True)   # 相似度从高到低
        return [m for _, m in scored[:self.top_k]]

    def get_context(self, query: str = "") -> List[Dict[str, str]]:
        # RAG 关键：得知道“当前问的是什么”才能检索 —— 检索是问题驱动的
        if not query:
            return self.messages[-self.top_k:]
        return self.retrieve(query)
