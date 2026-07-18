"""
把同一段对话喂给四种记忆，打印它们各自【会塞回给模型的内容】，
一眼看懂四者区别。（本 demo 不调模型，纯看记忆行为，无需 key。）

运行：
    python3 demos/compare_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import NaiveMemory, SlidingWindowMemory, SummaryMemory, VectorMemory

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

# 开头交代关键事实（名字、宠物），中间是闲聊
CONVERSATION = [
    ("user", "我叫小明"),
    ("assistant", "你好小明！"),
    ("user", "我养了一只猫叫咪咪"),
    ("assistant", "咪咪真可爱~"),
    ("user", "今天天气不错"),
    ("assistant", "是呀，适合出门"),
    ("user", "我中午吃了拉面"),
    ("assistant", "听起来很香"),
]
QUERY = "我的猫叫什么名字？"  # RAG 靠它决定取哪几条


def show(title, messages):
    print(f"\n=== {title} ===  （塞回给模型 {len(messages)} 条）")
    for m in messages:
        print(f"  [{m['role']}] {m['content']}")


def load(mem):
    mem.clear()
    for role, content in CONVERSATION:
        mem.add(role, content)
    return mem


show("Naive 全量", load(NaiveMemory(str(DATA_DIR / "mem_naive.json"))).get_context())
show("滑动窗口 window=4",
     load(SlidingWindowMemory(str(DATA_DIR / "mem_window.json"), window=4)).get_context())
show("摘要压缩 keep_recent=2",
     load(SummaryMemory(str(DATA_DIR / "mem_summary.json"), keep_recent=2)).get_context())
vmem = load(VectorMemory(str(DATA_DIR / "mem_vector.json"), top_k=2))
show(f"向量检索 RAG（问题：{QUERY}）", vmem.get_context(QUERY))

print("\n结论：")
print("  Naive   —— 最全，但历史一长就爆炸")
print("  滑动窗口 —— 最省事，但会忘掉久远但重要的事（这里忘了猫名）")
print("  摘要    —— 兼顾长期，但细节被压缩会失真")
print("  RAG     —— 海量历史里精准捞相关的几条，最强但最复杂")
