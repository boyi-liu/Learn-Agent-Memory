"""
归纳/反思演示：把一堆零散的底层记忆，归纳成更高层的洞见。

先喂几轮都在暗示同一主题的对话（反复加班 / 开始跑步），让 MemoryManagerV2
提炼出零散事实；再调用 reflect()，看它归纳出「用户工作压力大」「用户开始规律锻炼」
这类高层洞见，并作为新记忆（标 reflection、带来源）存入。

⚠️ 用虚拟环境运行：
       .venv/bin/python demos/reflection_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import RagMemory, MemoryManagerV2
from memory.memory_manager_v2 import REFLECTION_ROLE
from llm import has_key, deepseek_llm

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

TURNS = [
    ("我这周又加班到十点了，累死了。", "辛苦了，注意休息。"),
    ("老板今天又甩给我三个新需求。", "任务真不少。"),
    ("周末我也在处理工作邮件，没怎么歇。", "该给自己放个假了。"),
    ("我最近开始每天早上跑步五公里。", "坚持锻炼很棒！"),
    ("今天跑完步整个人都清爽了，打算长期坚持。", "好习惯，加油。"),
]


def dump(mem):
    print("  底层事实：")
    for i, doc in enumerate(mem.store.documents):
        if mem.store.metadatas[i].get("role") != REFLECTION_ROLE:
            print(f"    · {doc}")
    print("  高层洞见（reflection）：")
    any_r = False
    for i, doc in enumerate(mem.store.documents):
        if mem.store.metadatas[i].get("role") == REFLECTION_ROLE:
            any_r = True
            src = mem.store.metadatas[i].get("sources", [])
            print(f"    ★ {doc}   (归纳自 {len(src)} 条: {src})")
    if not any_r:
        print("    （暂无）")


def main():
    if not has_key():
        print("未配置 DeepSeek key，无法运行（提炼与归纳都要调用模型）。")
        return

    print("加载模型……")
    mem = RagMemory(persist_path=str(DATA_DIR / "reflection_memory.json"), top_k=4)
    mem.clear()
    # 关掉自动反思，手动触发以便观察（reflect_every=0）
    manager = MemoryManagerV2(mem, llm_fn=deepseek_llm, reflect_every=0)

    print("\n【第一步】逐轮对话，提炼零散事实……")
    for user, assistant in TURNS:
        manager.observe(user, assistant)
    dump(mem)

    print("\n【第二步】调用 reflect()，把零散事实归纳成高层洞见……")
    insights = manager.reflect()
    for ins in insights:
        print(f"  + 新洞见（importance={ins['importance']}）：{ins['insight']}")

    print("\n===== 归纳后的记忆全貌 =====")
    dump(mem)

    print("\n【第三步】混合召回验证：问“最近状态如何”，看高层洞见会不会被召回")
    for r in manager.recall("最近我的工作和生活状态怎么样？", top_k=4):
        tag = "★洞见" if r["content"] in [i["insight"] for i in insights] else "事实"
        print(f"  [{tag}] score={r['score']:.3f}  {r['content']}")


if __name__ == "__main__":
    main()
