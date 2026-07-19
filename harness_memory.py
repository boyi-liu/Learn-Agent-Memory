"""
harness_memory —— 把各种记忆统一成 harness 主循环能用的一个接口。

问题：不同记忆的用法不一样。
    naive/window/summary：get_context() 直接返回要塞回的历史（不需要检索）
    rag                 ：get_context(query) 按相似度召回长期记忆 + 自己维护短期窗口
    managed             ：写入走 manager.observe（提炼/去重/更新），读取走 manager.recall（混合召回）

解决：给每种记忆包一个适配器，都只暴露主循环需要的两个动作：
    context(user_input) -> 要插进 prompt 的一段消息（历史或召回到的记忆）
    record(user_input, reply) -> 把这一轮存下来

这样 main.py 的循环完全不用关心底层是哪种记忆，换记忆只改 config.yaml。
build_memory() 是工厂：读配置，构造对应后端。重依赖（RagMemory→torch）按需 import，
选 naive/window/summary 时根本不会加载模型。
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / ".data"
HISTORY_PATH = str(DATA_DIR / "main_history.json")   # naive/window/summary 存这里（消息列表）
VECTOR_PATH = str(DATA_DIR / "main_memory.json")     # rag/managed 存这里（向量库）

MEMORY_BLOCK_HEADER = "【相关记忆】（从长期记忆检索到，可能有用）：\n"


def _memory_block(contents):
    """把召回到的若干条记忆拼成一条 system 消息。"""
    body = "\n".join(f"- {c}" for c in contents)
    return {"role": "system", "content": MEMORY_BLOCK_HEADER + body}


# ============================================================
# 后端一：全量历史型（naive / window / summary）
#   这类记忆本身就决定“塞回什么”，直接用它的 get_context()。
# ============================================================
class FullHistoryMemory:
    def __init__(self, mem, name):
        self.mem = mem
        self.name = name

    def context(self, user_input):
        return list(self.mem.get_context())          # 忽略 user_input，返回历史

    def record(self, user_input, reply):
        self.mem.add("user", user_input)
        self.mem.add("assistant", reply)

    def count(self):
        return len(self.mem.messages)

    def clear(self):
        self.mem.clear()


# ============================================================
# 检索型公共部分：长期检索 + 短期最近窗口
# ============================================================
class _RetrievalMemory:
    def __init__(self, recent_window):
        self.window = recent_window
        self.recent = []          # 短期记忆：本进程最近几条原文

    def _retrieve(self, user_input):
        """子类实现：返回召回到的记忆内容列表（字符串）。"""
        raise NotImplementedError

    def context(self, user_input):
        hits = self._retrieve(user_input)
        msgs = []
        if hits:
            msgs.append(_memory_block(hits))
        msgs.extend(self.recent)                      # 再拼上最近几句，保证连贯
        return msgs

    def _push_recent(self, user_input, reply):
        self.recent.append({"role": "user", "content": user_input})
        self.recent.append({"role": "assistant", "content": reply})
        self.recent[:] = self.recent[-self.window:]

    def count(self):
        return self.mem.store.count()

    def clear(self):
        self.mem.clear()
        self.recent.clear()


# 后端二：RAG（每句都存，纯相似度召回）
class RagBackend(_RetrievalMemory):
    name = "rag（向量检索）"

    def __init__(self, mem, recent_window, top_k):
        super().__init__(recent_window)
        self.mem = mem
        self.top_k = top_k

    def _retrieve(self, user_input):
        return [h["content"] for h in self.mem.get_context(user_input, top_k=self.top_k)]

    def record(self, user_input, reply):
        self._push_recent(user_input, reply)
        self.mem.add("user", user_input)
        self.mem.add("assistant", reply)


# 后端三：Managed（提炼/去重/更新 + 混合召回）
class ManagedBackend(_RetrievalMemory):
    name = "managed（主动管理 + 混合召回）"

    def __init__(self, manager, recent_window, top_k):
        super().__init__(recent_window)
        self.manager = manager
        self.mem = manager.memory
        self.top_k = top_k

    def _retrieve(self, user_input):
        return [h["content"] for h in self.manager.recall(user_input, top_k=self.top_k)]

    def record(self, user_input, reply):
        self._push_recent(user_input, reply)
        self.manager.observe(user_input, reply)      # 提炼事实、去重/更新后写入长期


# ============================================================
# 工厂：读配置，构造对应后端
# ============================================================
def build_memory(mem_cfg: dict, llm_fn):
    backend = (mem_cfg.get("backend") or "managed").lower()
    recent_window = int(mem_cfg.get("recent_window", 6))
    top_k = int(mem_cfg.get("retrieve_top_k", 3))

    if backend == "naive":
        from memory import NaiveMemory
        return FullHistoryMemory(NaiveMemory(HISTORY_PATH), name="naive（全量历史）")

    if backend == "window":
        from memory import SlidingWindowMemory
        return FullHistoryMemory(
            SlidingWindowMemory(HISTORY_PATH, window=recent_window),
            name=f"window（最近{recent_window}条）")

    if backend == "summary":
        from memory import SummaryMemory
        keep = int(mem_cfg.get("summary_keep_recent", 2))
        return FullHistoryMemory(
            SummaryMemory(HISTORY_PATH, keep_recent=keep),
            name="summary（摘要 + 近期）")

    if backend == "rag":
        from memory import RagMemory
        return RagBackend(RagMemory(VECTOR_PATH, top_k=top_k), recent_window, top_k)

    if backend in ("managed", "managed_v2"):
        from memory import RagMemory
        mem = RagMemory(VECTOR_PATH, top_k=top_k)
        if backend == "managed_v2":
            from memory import MemoryManagerV2
            manager = MemoryManagerV2(
                mem, llm_fn=llm_fn, reflect_every=int(mem_cfg.get("reflect_every", 6)))
        else:
            from memory import MemoryManager
            manager = MemoryManager(mem, llm_fn=llm_fn)
        b = ManagedBackend(manager, recent_window, top_k)
        if backend == "managed_v2":
            b.name = "managed_v2（主动管理 + 反思归纳 + 混合召回）"
        return b

    raise ValueError(
        f"未知的 memory backend：{backend!r}。"
        "可选：naive / window / summary / rag / managed / managed_v2")
