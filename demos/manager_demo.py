"""
MemoryManager 演示：主动提炼 + 去重/更新，对比“无脑全存”的区别。

不改 main.py —— 这里独立地把 MemoryManager 包在 RagMemory 外面跑一遍。

四轮对话覆盖四种情况：
    1. 交代多个事实      -> 提炼出多条并 ADD
    2. 纯闲聊（天气）    -> 提炼出 []，长期记忆不增长（对比：naive 会把废话也存进去）
    3. 状态变化（搬家）  -> UPDATE：删掉旧的“住杭州”，存入“住北京”
    4. 重复已知（过敏）  -> SKIP：不重复存

⚠️ 用虚拟环境运行：
       .venv/bin/python demos/manager_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import RagMemory, MemoryManager
from llm import has_key, deepseek_llm

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

# 每轮 = (用户说的, 助手大致回了什么)。助手内容只是陪衬，提炼只关心用户事实。
TURNS = [
    ("我叫小明，住在杭州，对花生过敏。", "你好小明！记住啦。"),
    ("今天天气真不错，适合出去走走。", "是呀，好天气。"),
    ("我上周从杭州搬到北京了，以后不住杭州了。", "了解，祝你在北京一切顺利。"),
    ("提醒你一下，我对花生过敏，帮我点餐时注意。", "好的，一定避开花生。"),
]


def main():
    if not has_key():
        print("未配置 DeepSeek key，无法运行（提炼/消解都要调用模型）。")
        return

    print("加载长期记忆与模型……")
    mem = RagMemory(persist_path=str(DATA_DIR / "manager_memory.json"), top_k=4)
    mem.clear()
    manager = MemoryManager(mem, llm_fn=deepseek_llm)

    for i, (user, assistant) in enumerate(TURNS, 1):
        print(f"\n第 {i} 轮  用户：{user}")
        changes = manager.observe(user, assistant)
        if not changes:
            print("  （没有值得长期记住的事实，长期记忆不变）")
        for c in changes:
            if c["action"] == "ADD":
                print(f"  + ADD    {c['fact']}")
            elif c["action"] == "SKIP":
                print(f"  = SKIP   {c['fact']}   （与已有『{c['dup']}』重复）")
            elif c["action"] == "UPDATE":
                print(f"  ~ UPDATE {c['fact']}   （取代旧记忆『{c['old']}』）")

    print("\n===== 最终长期记忆库 =====")
    for doc in mem.store.documents:
        print(f"  · {doc}")
    print(f"\n共 {mem.store.count()} 条。注意：闲聊没进库，搬家把杭州更新成了北京，过敏没被重复存。")


if __name__ == "__main__":
    main()
