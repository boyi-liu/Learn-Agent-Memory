"""
mem0.py —— 忠实实现 Mem0（向量版）的核心算法。

Mem0 的写入是两阶段（和 memory_manager 很像，但更完整）：

    1. Extraction 抽取：结合【最近几轮对话上下文】，用 LLM 从当前这轮抽取候选事实。
       —— 比 memory_manager 只看单轮更强：跨轮的指代（“它”“那家公司”）也能抽对。
    2. Update 更新：对每条候选事实，先向量检索出相关旧记忆，再让 LLM 决定四种操作之一：
           ADD    全新信息，直接加
           UPDATE 补充/修正某条旧记忆，合并成更完整的一条（删旧存新）
           DELETE 新事实【否定/推翻】了某条旧记忆 → 删掉那条旧的（负向信息本身不入库）
           NOOP   已有记忆已包含，什么都不做

检索（search）就是纯向量相似度 —— 这也是 Mem0 生产版的做法。

和 memory_manager 的关键区别：多了 DELETE，且提炼带多轮上下文，所以这是更完整的 Mem0。
依赖注入 llm_fn，存储复用 RagMemory（embedding + 手搓向量库）。
"""

from typing import Callable, List, Dict

from .memory_manager import _parse_json


class Mem0:
    def __init__(self, memory, llm_fn: Callable[[list], str],
                 context_window: int = 6, update_top_k: int = 5, search_top_k: int = 5):
        self.memory = memory                 # RagMemory：向量存储 + 相似度检索
        self.llm = llm_fn
        self.context_window = context_window  # 提炼时带上最近多少条消息作上下文
        self.update_top_k = update_top_k      # Update 阶段检索几条相关旧记忆
        self.search_top_k = search_top_k
        self.recent: List[Dict[str, str]] = []  # 滚动对话上下文

    # ========== 写入主入口 ==========
    def add(self, user_text: str, assistant_text: str = "") -> List[Dict]:
        """处理一轮对话：抽取候选事实 → 逐条 ADD/UPDATE/DELETE/NOOP。返回操作日志。"""
        facts = self._extract(user_text, assistant_text)
        results = [self._update(f) for f in facts]
        # 抽取用的是“旧上下文”，处理完再把这轮追加进上下文
        self.recent.append({"role": "user", "content": user_text})
        if assistant_text:
            self.recent.append({"role": "assistant", "content": assistant_text})
        self.recent[:] = self.recent[-self.context_window:]
        return results

    # ========== 阶段一：带上下文的抽取 ==========
    def _extract(self, user_text: str, assistant_text: str) -> List[str]:
        context = "\n".join(f"{m['role']}: {m['content']}" for m in self.recent) or "（无）"
        messages = [
            {"role": "system", "content":
                "你是 Mem0 的记忆抽取器。结合【最近对话上下文】理解【当前这轮】，"
                "抽取值得长期记住的、关于用户的稳定事实（身份/偏好/关系/约束等）。"
                "利用上下文解决指代（如“它/那家公司”指的是谁）。"
                "忽略寒暄和一次性信息。以 JSON 字符串数组返回，没有就返回 []。只输出 JSON。"},
            {"role": "user", "content":
                f"最近对话上下文：\n{context}\n\n当前这轮：\n用户：{user_text}\n助手：{assistant_text}"},
        ]
        raw = _parse_json(self.llm(messages), default=[])
        facts = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                facts.append(item.strip())
            elif isinstance(item, dict):   # 容错：模型返回了 {"fact": ...} 这类对象
                val = item.get("fact") or item.get("text") or item.get("content")
                if val and str(val).strip():
                    facts.append(str(val).strip())
        return facts

    # ========== 阶段二：四操作更新 ==========
    def _update(self, fact: str) -> Dict:
        if self.memory.store.count() == 0:
            self.memory.add("memory", fact)
            return {"action": "ADD", "fact": fact}

        candidates = self.memory.get_context(fact, top_k=self.update_top_k)
        listing = "\n".join(f"[{i}] {c['content']}" for i, c in enumerate(candidates))
        messages = [
            {"role": "system", "content":
                "你在维护 Mem0 长期记忆库。给你若干【已有记忆】和一条【新事实】，"
                "判断对新事实执行哪种操作：\n"
                "- ADD：全新信息，和已有都不冲突不重复\n"
                "- UPDATE：是对某条已有记忆的补充/修正，应合并成更完整的一条\n"
                "- DELETE：新事实【否定或推翻】了某条已有记忆（如“不再…了”），应删掉那条旧的\n"
                "- NOOP：某条已有记忆已包含该信息，无需变动\n"
                '只输出 JSON：{"action":"ADD|UPDATE|DELETE|NOOP","target":<相关旧记忆编号或null>,'
                '"text":<UPDATE 时给出合并后的最终文本，否则 null>}'},
            {"role": "user", "content": f"已有记忆：\n{listing}\n\n新事实：{fact}"},
        ]
        decision = _parse_json(self.llm(messages), default={"action": "ADD"})
        action = str(decision.get("action", "ADD")).upper()
        target = decision.get("target")

        target_id, old_text = None, None
        if isinstance(target, int) and 0 <= target < len(candidates):
            target_id = candidates[target]["id"]
            old_text = candidates[target]["content"]

        if action == "NOOP":
            return {"action": "NOOP", "fact": fact, "match": old_text}

        if action == "DELETE" and target_id is not None:
            self.memory.store.delete(target_id)     # 删掉被推翻的旧记忆，负向信息本身不入库
            return {"action": "DELETE", "fact": fact, "removed": old_text, "removed_id": target_id}

        if action == "UPDATE" and target_id is not None:
            merged = (decision.get("text") or fact).strip()
            self.memory.store.delete(target_id)
            self.memory.add("memory", merged)
            return {"action": "UPDATE", "fact": merged, "old": old_text, "old_id": target_id}

        self.memory.add("memory", fact)             # 兜底 ADD
        return {"action": "ADD", "fact": fact}

    # ========== 检索：纯向量相似度 ==========
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        return self.memory.get_context(query, top_k=top_k or self.search_top_k)
