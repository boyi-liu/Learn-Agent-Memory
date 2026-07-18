"""
手搓的最小向量数据库（替代 ChromaDB）。

一个向量库其实只做三件事：
    1. 存：把 (向量, 原文, 元数据) 存起来
    2. 查：给一个查询向量，算它和每条的相似度，排序取最相似的 top_k
    3. 持久化：存盘 / 读盘

这里：
    - 相似度用【余弦相似度】(值域约 -1~1，越大越相似)，numpy 一行算完
    - 检索用【暴力遍历】(和所有向量都算一遍)。真实的大规模向量库(FAISS/
      Chroma/Milvus)会用近似最近邻(ANN)索引加速，但思想就是这个，只是更快。
    - 持久化故意用 JSON，方便你直接打开文件看：一条记忆 = 一段文字 + 一串数字。

纯 numpy，不依赖任何向量库。
"""

import json
import os
from typing import List, Dict, Optional

import numpy as np


class VectorStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.ids: List[str] = []
        self.vectors: List[List[float]] = []   # 每条是一串数字（向量）
        self.documents: List[str] = []         # 每条对应的原文
        self.metadatas: List[dict] = []        # 每条的附加信息，如 {"role": ...}
        if path:
            self._load()

    # ---------- 存 ----------
    def add(self, id: str, vector: List[float], document: str, metadata: dict = None) -> None:
        self.ids.append(id)
        self.vectors.append(list(map(float, vector)))
        self.documents.append(document)
        self.metadatas.append(metadata or {})
        self._save()

    # ---------- 查（向量库的灵魂）----------
    def query(self, vector: List[float], top_k: int = 3) -> List[Dict]:
        if not self.vectors:
            return []

        matrix = np.array(self.vectors, dtype=np.float32)   # (N, dim) 所有存过的向量
        q = np.array(vector, dtype=np.float32)              # (dim,)   查询向量

        # 余弦相似度 = 点积 / (各自模长)。先各自归一化，再点积就是余弦值。
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        scores = matrix_norm @ q_norm                       # (N,) 每条与查询的相似度

        # 取分数最高的 top_k（argsort 默认升序，取末尾再反转）
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.ids[i],
                "document": self.documents[i],
                "metadata": self.metadatas[i],
                "score": round(float(scores[i]), 4),  # 越接近 1 越相关
            }
            for i in top_idx
        ]

    def count(self) -> int:
        return len(self.ids)

    def clear(self) -> None:
        self.ids, self.vectors, self.documents, self.metadatas = [], [], [], []
        self._save()

    # ---------- 持久化（就是把上面几个 list 存成一个 JSON）----------
    def _save(self) -> None:
        if not self.path:
            return
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = {
            "ids": self.ids,
            "vectors": self.vectors,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _load(self) -> None:
        if self.path and os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.ids = data.get("ids", [])
            self.vectors = data.get("vectors", [])
            self.documents = data.get("documents", [])
            self.metadatas = data.get("metadatas", [])
