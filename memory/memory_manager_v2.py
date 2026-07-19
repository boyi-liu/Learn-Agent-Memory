"""
MemoryManagerV2 —— 在 MemoryManager 之上加「归纳 / 反思」(reflection / consolidation)。

v1 解决的是【单条】记忆的写入质量：提炼、去重、更新、混合召回。
但它永远只有一层扁平的底层事实，不会“想通”什么。真实的类人记忆还会做一件事：

    把一批相关的底层记忆，归纳成更高层的洞见。
    例如从「周一加班」「周三加班」「周末还在回邮件」→ 归纳出「用户工作压力大」。

这就是 Generative Agents 里的 reflection：记忆从一层"事实"，长出第二层"洞见"，
形成一棵有层次的记忆树。高层洞见信息密度更高，召回时往往比零散事实更有用。

MemoryManagerV2 继承 v1（白拿 observe / reconcile / recall），只新增：
    reflect()      ：把底层记忆归纳成若干高层洞见，作为新记忆存入（标记 type=reflection）
    自动触发       ：每写入 reflect_every 条新记忆，自动反思一次

设计上仍不 import llm 包（llm 函数是注入的）。归纳出的洞见会：
    - 标 role="reflection"，并记录它由哪些底层记忆(sources)归纳而来（provenance 溯源）
    - 给较高的 importance，从而在混合召回里更容易被选中
去重：把"已有洞见"一并给 LLM，并要求"只输出新的、别重复"，避免反复反思堆出重复洞见。
"""

from typing import List, Dict

from .memory_manager import MemoryManager, _parse_json

REFLECTION_ROLE = "reflection"


class MemoryManagerV2(MemoryManager):
    def __init__(self, memory, llm_fn, reconcile_top_k: int = 4,
                 reflect_every: int = 6, max_insights: int = 3):
        super().__init__(memory, llm_fn, reconcile_top_k)
        self.reflect_every = reflect_every   # 每写入这么多条新记忆自动反思一次；<=0 关闭自动
        self.max_insights = max_insights     # 每次反思最多产出几条洞见
        self._since_reflect = 0              # 距上次反思，又写入了多少条

    # ---------- 覆盖 observe：写入之余，累计计数并按需自动反思 ----------
    def observe(self, user_text: str, assistant_text: str = "") -> List[Dict]:
        changes = super().observe(user_text, assistant_text)
        # 只有真正落库的(ADD/UPDATE)才算“新写入”，SKIP 不算
        written = sum(1 for c in changes if c["action"] in ("ADD", "UPDATE"))
        self._since_reflect += written
        if self.reflect_every > 0 and self._since_reflect >= self.reflect_every:
            self.reflect()
            self._since_reflect = 0
        return changes

    # ---------- 反思：把底层记忆归纳成高层洞见 ----------
    def reflect(self) -> List[Dict]:
        """
        读取现有底层记忆，归纳出若干更高层的洞见并存入。返回本次新增的洞见列表：
            [{"insight": "...", "importance": 8, "sources": ["mem-1","mem-3"]}]
        """
        base = self._memories_of(exclude_role=REFLECTION_ROLE)   # 只对底层事实反思
        if len(base) < 3:
            return []   # 事实太少，不值得归纳

        existing = [m["content"] for m in self._memories_of(only_role=REFLECTION_ROLE)]
        insights = self._synthesize(base, existing)

        results = []
        for ins in insights:
            # 用父类的写入通道（自动 embed + 时间戳），标成 reflection 并给高重要性
            new_id = self.memory.add(REFLECTION_ROLE, ins["insight"], importance=ins["importance"])
            self._tag(new_id, type="reflection", sources=ins["sources"])   # 溯源
            results.append(ins)
        return results

    # ---------- LLM 归纳 ----------
    def _synthesize(self, base: List[Dict], existing_insights: List[str]) -> List[Dict]:
        listing = "\n".join(f"[{i}] {m['content']}" for i, m in enumerate(base))
        existing_text = ("\n".join(f"- {t}" for t in existing_insights)
                         if existing_insights else "（暂无）")
        messages = [
            {"role": "system", "content":
                "你是记忆归纳器。下面是一批零散的【底层记忆】和一批【已有的高层洞见】。"
                "请把相关的底层记忆归纳成【更高层的洞见】——即把多条事实抽象成一个结论"
                "（例：多次提到加班 → “用户工作压力较大”）。要求：\n"
                "- 只在确有可归纳的规律时输出；单条事实、泛泛之谈都不要输出\n"
                "- 不要重复或改写【已有的高层洞见】，只输出新的\n"
                "- 每条给出重要性 importance(1-10) 和它归纳自哪些底层记忆的编号 sources\n"
                '返回 JSON 数组：[{"insight":"高层结论","importance":1-10,"sources":[编号...]}]；'
                "没有可归纳的就返回 []。只输出 JSON。"},
            {"role": "user", "content":
                f"底层记忆：\n{listing}\n\n已有的高层洞见：\n{existing_text}\n\n"
                f"最多归纳 {self.max_insights} 条。"},
        ]
        raw = _parse_json(self.llm(messages), default=[])
        insights = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("insight", "")).strip()
            if not text:
                continue
            try:
                imp = max(1.0, min(10.0, float(item.get("importance", 6))))
            except (TypeError, ValueError):
                imp = 6.0
            nums = item.get("sources") or []
            src_ids = [base[n]["id"] for n in nums if isinstance(n, int) and 0 <= n < len(base)]
            insights.append({"insight": text, "importance": imp, "sources": src_ids})
        return insights[:self.max_insights]

    # ---------- 小工具：按 role 过滤记忆 / 给某条记忆打标签 ----------
    def _memories_of(self, exclude_role: str = None, only_role: str = None) -> List[Dict]:
        st = self.memory.store
        out = []
        for i, doc in enumerate(st.documents):
            role = st.metadatas[i].get("role")
            if exclude_role and role == exclude_role:
                continue
            if only_role and role != only_role:
                continue
            out.append({"id": st.ids[i], "content": doc})
        return out

    def _tag(self, mem_id: str, **fields) -> None:
        st = self.memory.store
        if mem_id in st.ids:
            st.metadatas[st.ids.index(mem_id)].update(fields)
            st._save()
