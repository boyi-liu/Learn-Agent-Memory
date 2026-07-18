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

from typing import List, Dict

from .vector_store import VectorStore


class RagMemory:
    def __init__(
        self,
        persist_path: str,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 3,
    ):
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
    def add(self, role: str, content: str) -> None:
        n = self.store.count()  # 用当前条数当自增 id
        self.store.add(
            id=f"msg-{n}",
            vector=self._embed(content),
            document=content,
            metadata={"role": role},
        )

    # ---------- 读取：按与 query 的向量相似度，取最相关的几条 ----------
    def get_context(self, query: str = "", top_k: int = None) -> List[Dict[str, str]]:
        k = top_k or self.top_k
        if self.store.count() == 0 or not query:
            return []
        hits = self.store.query(self._embed(query), top_k=k)
        return [
            {
                "role": h["metadata"].get("role", "user"),
                "content": h["document"],
                "score": h["score"],  # 越接近 1 越相关
            }
            for h in hits
        ]

    def clear(self) -> None:
        self.store.clear()
