"""
真实的 RAG 记忆：真 embedding 模型 + 手搓向量库(VectorStore)。

和 vector.py 里的玩具版 VectorMemory 是同一个思想，把两处换成“真的”：
    假 embedding（词袋）      -> sentence-transformers 真模型，文字变成向量
    假相似度（词重叠）+ list  -> 手搓 VectorStore，余弦相似度检索 + 落盘

不依赖 ChromaDB —— 存储和检索都在 memory/vector_store.py 里手写实现，
方便你看清向量库内部到底在做什么。对外接口仍是 add() / get_context(query)。

⚠️ embedding 模型依赖较重（torch / sentence-transformers），用虚拟环境运行：
       .venv/bin/python demos/rag_demo.py
"""

import os
import time
from pathlib import Path
from typing import List, Dict

from .vector_store import VectorStore


def _model_is_cached(model_name: str) -> bool:
    """模型是否已下载到本地 HuggingFace 缓存。"""
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    folder = "models--" + model_name.replace("/", "--")
    # 官方模型名不含组织前缀时，实际缓存在 sentence-transformers 命名空间下
    return (cache / folder).exists() or (
        cache / f"models--sentence-transformers--{model_name}"
    ).exists()


class RagMemory:
    def __init__(
        self,
        persist_path: str,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 3,
    ):
        # 启动优化：模型已缓存时开启离线模式，跳过每次启动都联网检查更新的 ~6 秒。
        # 必须在 import sentence_transformers 之前设置，否则 huggingface_hub 读不到。
        if _model_is_cached(model_name):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # embedding 模型较重，延迟到这里再导入
        from sentence_transformers import SentenceTransformer

        # 1) embedding 模型：第一次会自动下载（多语言，支持中文）
        self.model = SentenceTransformer(model_name)

        # 2) 手搓的向量库：数据落到 persist_path（一个 JSON 文件），重启还在
        self.store = VectorStore(path=persist_path)

        self.top_k = top_k

    # ---------- 把文字变成向量 ----------
    def _embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    # ---------- 写入：内容连同它的向量一起存进向量库 ----------
    def add(self, role: str, content: str, importance: float = None,
            timestamp: float = None) -> str:
        # 每条自动打时间戳（供“新近度”打分用）；importance 可选（供“重要性”打分用）
        meta = {
            "role": role,
            "timestamp": timestamp if timestamp is not None else time.time(),
        }
        if importance is not None:
            meta["importance"] = float(importance)
        # id 交给 store 自动生成（永不重复），返回该 id 方便上层管理
        return self.store.add(vector=self._embed(content), document=content, metadata=meta)

    # ---------- 读取：按与 query 的向量相似度，取最相关的几条 ----------
    def get_context(self, query: str = "", top_k: int = None) -> List[Dict[str, str]]:
        k = top_k or self.top_k
        if self.store.count() == 0 or not query:
            return []
        hits = self.store.query(self._embed(query), top_k=k)
        return [
            {
                "id": h["id"],                             # 带上 id，便于更新/删除
                "role": h["metadata"].get("role", "user"),
                "content": h["document"],
                "score": h["score"],                       # 越接近 1 越相关
            }
            for h in hits
        ]

    def clear(self) -> None:
        self.store.clear()
