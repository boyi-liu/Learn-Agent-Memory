"""
mem0g.py —— 忠实实现 Mem0g（图版）的核心算法。

和向量版 Mem0 不同，Mem0g 不把记忆存成一条条句子，而是抽成【知识三元组】：
    (主体 source, 关系 relation, 客体 target)
例如「小明住在杭州」→ (小明, 住在, 杭州)。许多三元组连起来就是一张【知识图谱】。

好处：关系是显式的、可组合的。想知道“小明住哪”，直接沿 (小明, 住在, ?) 这条边查，
还能多跳推理（小明→在杭州公司→公司在西湖区…）。向量版只能靠语义相似度模糊召回。

写入两阶段（和 Mem0 同构，只是对象从“句子”变“边”）：
    1. Extraction：LLM 从对话里抽出若干三元组
    2. Update    ：对每个新三元组，看图里主体已有哪些边，让 LLM 决定
                   ADD / UPDATE（替换旧边，如搬家）/ DELETE（否定）/ NOOP
检索：先用 LLM 找出问题里的实体，再返回图中与这些实体相连的子图（相关三元组）。

存储用手搓的 GraphStore（一堆边 + JSON 落盘），不依赖 Neo4j 之类。
只需要注入 llm_fn，不需要 embedding 模型 —— 所以本文件普通 python3 就能跑。
"""

import json
import os
import re
from typing import Callable, List, Dict, Optional

from .memory_manager import _parse_json


# ============================================================
# 手搓知识图谱：本质就是一张“边”的列表 + 增删查 + 落盘
# ============================================================
class GraphStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.edges: List[Dict] = []   # 每条边 {id, source, relation, target}
        self.seq = 0
        if path:
            self._load()

    def add_edge(self, source: str, relation: str, target: str) -> str:
        edge_id = f"e-{self.seq}"
        self.seq += 1
        self.edges.append({"id": edge_id, "source": source,
                           "relation": relation, "target": target})
        self._save()
        return edge_id

    def delete_edge(self, edge_id: str) -> None:
        self.edges = [e for e in self.edges if e["id"] != edge_id]
        self._save()

    def entities(self) -> set:
        out = set()
        for e in self.edges:
            out.add(e["source"])
            out.add(e["target"])
        return out

    def edges_touching(self, names: List[str]) -> List[Dict]:
        """返回 source 或 target 命中任一 name（按子串、大小写不敏感）的边。"""
        keys = [n.lower() for n in names if n]
        hit = []
        for e in self.edges:
            s, t = e["source"].lower(), e["target"].lower()
            if any(k in s or s in k or k in t or t in k for k in keys):
                hit.append(e)
        return hit

    def edges_of_source(self, source: str) -> List[Dict]:
        s = source.lower()
        return [e for e in self.edges if e["source"].lower() == s]

    def count(self) -> int:
        return len(self.edges)

    def clear(self) -> None:
        self.edges = []
        self._save()

    def _save(self) -> None:
        if not self.path:
            return
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"seq": self.seq, "edges": self.edges}, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        if self.path and os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.edges = data.get("edges", [])
            self.seq = data.get("seq", len(self.edges))


# ============================================================
# Mem0g：图谱记忆
# ============================================================
class Mem0g:
    def __init__(self, llm_fn: Callable[[list], str], persist_path: str = None,
                 context_window: int = 6):
        self.llm = llm_fn
        self.graph = GraphStore(persist_path)
        self.context_window = context_window
        self.recent: List[Dict[str, str]] = []

    # ========== 写入 ==========
    def add(self, user_text: str, assistant_text: str = "") -> List[Dict]:
        triples = self._extract_triples(user_text, assistant_text)
        results = [self._update(t) for t in triples]
        self.recent.append({"role": "user", "content": user_text})
        if assistant_text:
            self.recent.append({"role": "assistant", "content": assistant_text})
        self.recent[:] = self.recent[-self.context_window:]
        return results

    # ---------- 阶段一：抽三元组 ----------
    def _extract_triples(self, user_text: str, assistant_text: str) -> List[Dict]:
        context = "\n".join(f"{m['role']}: {m['content']}" for m in self.recent) or "（无）"
        messages = [
            {"role": "system", "content":
                "你是知识图谱抽取器。结合上下文，从当前对话中抽取关于用户的【知识三元组】"
                "(主体, 关系, 客体)，例如「小明住在杭州」→ 主体=小明, 关系=住在, 客体=杭州。"
                "主体尽量用具体实体名（已知用户名就用用户名，否则用“用户”）。"
                "忽略寒暄和一次性信息。"
                '以 JSON 数组返回，每项 {"source":"主体","relation":"关系","target":"客体"}；'
                "没有就返回 []。只输出 JSON。"},
            {"role": "user", "content":
                f"最近对话上下文：\n{context}\n\n当前这轮：\n用户：{user_text}\n助手：{assistant_text}"},
        ]
        raw = _parse_json(self.llm(messages), default=[])
        triples = []
        for item in raw:
            if (isinstance(item, dict) and item.get("source")
                    and item.get("relation") and item.get("target")):
                triples.append({
                    "source": str(item["source"]).strip(),
                    "relation": str(item["relation"]).strip(),
                    "target": str(item["target"]).strip(),
                })
        return triples

    # ---------- 阶段二：四操作更新一条边 ----------
    def _update(self, triple: Dict) -> Dict:
        s, r, t = triple["source"], triple["relation"], triple["target"]
        existing = self.graph.edges_of_source(s)   # 该主体已有的边，作为冲突候选
        if not existing:
            edge_id = self.graph.add_edge(s, r, t)
            return {"action": "ADD", "triple": triple, "id": edge_id}

        listing = "\n".join(
            f"[{i}] ({e['source']}, {e['relation']}, {e['target']})"
            for i, e in enumerate(existing))
        messages = [
            {"role": "system", "content":
                "你在维护一张知识图谱。给你【主体已有的边】和一条【新三元组】，判断操作：\n"
                "- ADD：全新的关系/属性，和已有边讲的不是一回事\n"
                "- UPDATE：新三元组是【同一属性的最新值】，取代某条旧边。"
                "【即使关系用词不同】，只要语义上在讲同一件事就算 UPDATE，"
                "例如旧边(小明,居住于,杭州) 遇到新三元组(小明,搬到,北京)，都是“住在哪”，选 UPDATE\n"
                "- DELETE：新信息否定了某条旧边（如“不再在A公司了”）\n"
                "- NOOP：某条旧边已表达该信息\n"
                '只输出 JSON：{"action":"ADD|UPDATE|DELETE|NOOP","target":<旧边编号或null>}'},
            {"role": "user", "content":
                f"主体已有的边：\n{listing}\n\n新三元组：({s}, {r}, {t})"},
        ]
        decision = _parse_json(self.llm(messages), default={"action": "ADD"})
        action = str(decision.get("action", "ADD")).upper()
        target = decision.get("target")
        old_id, old = None, None
        if isinstance(target, int) and 0 <= target < len(existing):
            old_id = existing[target]["id"]
            old = existing[target]

        if action == "NOOP":
            return {"action": "NOOP", "triple": triple, "match": old}

        if action == "DELETE" and old_id is not None:
            self.graph.delete_edge(old_id)
            return {"action": "DELETE", "triple": triple, "removed": old}

        if action == "UPDATE" and old_id is not None:
            self.graph.delete_edge(old_id)
            new_id = self.graph.add_edge(s, r, t)
            return {"action": "UPDATE", "triple": triple, "old": old, "id": new_id}

        new_id = self.graph.add_edge(s, r, t)
        return {"action": "ADD", "triple": triple, "id": new_id}

    # ========== 检索：找出问题里的实体 → 返回相连子图 ==========
    def search(self, query: str) -> List[Dict]:
        entities = self._query_entities(query)
        edges = self.graph.edges_touching(entities) if entities else []
        return edges

    def _query_entities(self, query: str) -> List[str]:
        messages = [
            {"role": "system", "content":
                "从下面的问题里找出涉及的【实体/关键词】（人名、地名、机构、事物等），"
                '用 JSON 字符串数组返回。只输出 JSON。'},
            {"role": "user", "content": query},
        ]
        raw = _parse_json(self.llm(messages), default=[])
        return [str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()]

    def clear(self) -> None:
        self.graph.clear()
        self.recent.clear()
