"""
MemoryManager —— 给长期记忆加一层“主动管理”。

RagMemory 本身很纯粹：只负责“存 + 纯相似度召回”。MemoryManager 包在它外面，
承担所有“聪明”的记忆策略，做三件事：

    1. 提炼(Write)    ：每轮对话后让 LLM 判断有没有值得长期记的【事实】，
                        并给每条事实评一个【重要性 1-10】，只存提炼出的事实。
    2. 消解(Reconcile)：事实入库前先按相似度检索旧记忆，交给 LLM 判断
                        ADD（新知识）/ SKIP（重复）/ UPDATE（取代旧记忆，删旧存新）。
    3. 召回(Recall)   ：进阶检索，不只看相似度，而是
                        相似度 + 新近度 + 重要性 的混合打分（经典 Generative Agents）。

分层原因：RAG 的检索保持“老 metric”（纯相似度，见 RagMemory.get_context，也供上面
消解去重用）；而重要性是 manager 打的分、新近度是管理策略，混合召回自然属于这一层。

设计上刻意【不 import llm 包】，而是把 llm 函数作为参数传进来（依赖注入），
这样 memory 包保持独立、可用假 LLM 测试；接进 main.py 时把 deepseek_llm 传进来即可。

用法：
    from memory import RagMemory, MemoryManager
    from llm import deepseek_llm
    mem = RagMemory(persist_path=...)
    manager = MemoryManager(mem, llm_fn=deepseek_llm)
    manager.observe(user_text, assistant_text)   # 每轮对话后调一次（写入）
    manager.recall("用户爱喝什么")               # 需要回忆时调（混合召回）
"""

import json
import re
import time
from typing import Callable, List, Dict


def _parse_json(text: str, default):
    """从模型输出里尽量抠出 JSON（容忍 ```json 代码块和前后废话）。"""
    if not text:
        return default
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return default


def _minmax(xs: List[float]) -> List[float]:
    """把一组分数归一化到 [0,1]；全相等时统一给 0.5（该信号不影响排序）。"""
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


class MemoryManager:
    def __init__(self, memory, llm_fn: Callable[[list], str], reconcile_top_k: int = 4):
        self.memory = memory              # 一个 RagMemory（需有 add/get_context/_embed/store）
        self.llm = llm_fn                 # 注入的 LLM 调用：messages -> str
        self.reconcile_top_k = reconcile_top_k

    # ========== 写入主入口：看一轮对话，完成提炼 + 消解 ==========
    def observe(self, user_text: str, assistant_text: str = "") -> List[Dict]:
        """
        处理一轮对话，返回本轮对长期记忆做的变更日志，形如：
            [{"action": "ADD",    "fact": "...", "importance": 7},
             {"action": "UPDATE", "fact": "...", "old": "...", "old_id": "mem-3"},
             {"action": "SKIP",   "fact": "...", "dup": "..."}]
        """
        facts = self._extract(user_text, assistant_text)
        return [self._reconcile(f["fact"], f["importance"]) for f in facts]

    # ========== 第一步：提炼事实（并评重要性） ==========
    def _extract(self, user_text: str, assistant_text: str) -> List[Dict]:
        messages = [
            {"role": "system", "content":
                "你是记忆提炼器。从下面一轮对话中，抽取【值得长期记住的、关于用户的稳定事实】"
                "（身份、职业、偏好、人际关系、健康约束、长期目标等）。"
                "忽略寒暄、天气、一次性的临时信息、以及助手说的话本身。"
                "同时给每条事实评一个【重要性】1-10（越关乎安全/身份/长期越高，"
                "如过敏=9，随口偏好=4）。"
                '以 JSON 数组返回，每个元素形如 {"fact":"简洁陈述句","importance":1-10}；'
                "没有值得记的就返回 []。只输出 JSON。"},
            {"role": "user", "content":
                f"用户说：{user_text}\n助手说：{assistant_text}"},
        ]
        raw = _parse_json(self.llm(messages), default=[])
        facts = []
        for item in raw:
            if isinstance(item, dict) and item.get("fact"):
                imp = item.get("importance", 5)
                try:
                    imp = max(1.0, min(10.0, float(imp)))
                except (TypeError, ValueError):
                    imp = 5.0
                facts.append({"fact": str(item["fact"]).strip(), "importance": imp})
            elif isinstance(item, str) and item.strip():   # 容错：只给了字符串
                facts.append({"fact": item.strip(), "importance": 5.0})
        return facts

    # ========== 第二步：与已有记忆消解（ADD / SKIP / UPDATE） ==========
    def _reconcile(self, fact: str, importance: float = 5.0) -> Dict:
        # 库为空：无可比较，直接存
        if self.memory.store.count() == 0:
            self.memory.add("memory", fact, importance=importance)
            return {"action": "ADD", "fact": fact, "importance": importance}

        # 用【纯相似度】检索候选（RAG 的老 metric），交给 LLM 判断
        candidates = self.memory.get_context(fact, top_k=self.reconcile_top_k)
        listing = "\n".join(f"[{i}] {c['content']}" for i, c in enumerate(candidates))

        messages = [
            {"role": "system", "content":
                "你在维护一个长期记忆库。下面有若干【已有记忆】和一条【新事实】。判断它们的关系并决定操作：\n"
                "- ADD：新事实是全新信息，和已有记忆都不冲突也不重复\n"
                "- SKIP：新事实与某条已有记忆基本重复，无需再存\n"
                "- UPDATE：新事实更新/取代了某条已有记忆（如地点、状态、偏好发生了变化）\n"
                '只输出 JSON：{"action":"ADD|SKIP|UPDATE","target":<相关已有记忆的编号或null>,'
                '"merged":<UPDATE 时给出合并后的最终事实文本，否则为 null>}'},
            {"role": "user", "content":
                f"已有记忆：\n{listing}\n\n新事实：{fact}"},
        ]
        decision = _parse_json(self.llm(messages), default={"action": "ADD"})
        action = decision.get("action", "ADD")
        target = decision.get("target")

        target_id, old_text = None, None
        if isinstance(target, int) and 0 <= target < len(candidates):
            target_id = candidates[target]["id"]
            old_text = candidates[target]["content"]

        if action == "SKIP":
            return {"action": "SKIP", "fact": fact, "dup": old_text}

        if action == "UPDATE" and target_id is not None:
            merged = (decision.get("merged") or fact).strip()
            self.memory.store.delete(target_id)                     # 删掉过期旧记忆
            self.memory.add("memory", merged, importance=importance)  # 存入合并后新记忆
            return {"action": "UPDATE", "fact": merged, "old": old_text, "old_id": target_id}

        self.memory.add("memory", fact, importance=importance)      # 兜底都当 ADD
        return {"action": "ADD", "fact": fact, "importance": importance}

    # ========== 召回：相似度 + 新近度 + 重要性 的混合检索 ==========
    def recall(
        self,
        query: str,
        top_k: int = 5,
        pool: int = 20,
        now: float = None,
        w_similarity: float = 1.0,
        w_recency: float = 1.0,
        w_importance: float = 0.5,
        half_life_hours: float = 72.0,
    ) -> List[Dict]:
        """
        经典 Generative Agents 混合检索：三种信号各自归一化后加权求和再排序。
            相似度 similarity ：和 query 的语义接近程度（余弦）
            新近度 recency    ：越新越高，按 0.5^(age/半衰期) 指数衰减
            重要性 importance ：写入时 LLM 评的分（1-10）
        两阶段：先用相似度粗筛出候选池 pool，再在池内用混合分重排序取 top_k。
        返回每条带上「归一化×权重」后的各贡献量（三者相加 = score），便于看清排序原因。
        """
        store = self.memory.store
        n = store.count()
        if n == 0 or not query:
            return []
        now = now if now is not None else time.time()

        # 阶段一：相似度粗筛候选池（store.query 会带回 metadata）
        hits = store.query(self.memory._embed(query), top_k=min(pool, n))

        sims = [h["score"] for h in hits]
        recs = []
        for h in hits:
            t = h["metadata"].get("timestamp")
            if t is None:
                recs.append(0.0)  # 没时间戳的老数据视为最久远
            else:
                age_hours = max(0.0, (now - t) / 3600.0)
                recs.append(0.5 ** (age_hours / half_life_hours))
        imps = [float(h["metadata"].get("importance", 5.0)) for h in hits]

        # 阶段二：各信号归一化后加权
        sim_n, rec_n, imp_n = _minmax(sims), _minmax(recs), _minmax(imps)
        results = []
        for i, h in enumerate(hits):
            c_sim = w_similarity * sim_n[i]
            c_rec = w_recency * rec_n[i]
            c_imp = w_importance * imp_n[i]
            results.append({
                "id": h["id"],
                "content": h["document"],
                "similarity": round(sims[i], 4),
                "recency": round(recs[i], 4),
                "importance": imps[i],
                "c_similarity": round(c_sim, 4),
                "c_recency": round(c_rec, 4),
                "c_importance": round(c_imp, 4),
                "score": round(c_sim + c_rec + c_imp, 4),
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]
