"""
混合检索演示：纯相似度（RAG 的老 metric）vs 混合（相似度 + 新近度 + 重要性）。

混合检索是 MemoryManager.recall() 提供的进阶能力；RagMemory 本身只做纯相似度。
本 demo 只做检索、不写入，所以给 MemoryManager 传一个假 llm 即可（recall 不用 LLM）。

场景：库里有两条都和“喝什么”相关的记忆，一条很旧（30天前爱美式咖啡），
一条很新（1小时前迷上抹茶拿铁）。纯相似度分不清新旧；混合检索会把“最近”的
那条顶上来 —— 这正是加入时间维度的价值。另有一条高重要性的记忆（过敏）做对照。

⚠️ 用虚拟环境运行：
       .venv/bin/python demos/hybrid_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import RagMemory, MemoryManager

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

HOUR = 3600
DAY = 86400


def main():
    print("加载模型……")
    mem = RagMemory(persist_path=str(DATA_DIR / "hybrid_memory.json"), top_k=3)
    mem.clear()
    # recall 不需要 LLM，这里传个占位函数即可
    manager = MemoryManager(mem, llm_fn=lambda messages: "")

    now = time.time()
    # (内容, 多久以前, 重要性1-10)
    seeds = [
        ("用户喜欢喝美式咖啡",       30 * DAY, 5),   # 旧偏好
        ("用户最近迷上了抹茶拿铁",     1 * HOUR, 5),   # 新偏好
        ("用户是一名后端工程师",      10 * DAY, 4),
        ("用户对花生过敏，很严重",    20 * DAY, 9),   # 老，但很重要
    ]
    for content, ago, imp in seeds:
        mem.add("memory", content, importance=imp, timestamp=now - ago)

    query = "用户平时爱喝什么饮料？"
    print(f"\n查询：{query}\n")

    # A) 纯相似度（原来的 get_context）
    print("=== 纯相似度检索 ===")
    for r in mem.get_context(query, top_k=3):
        print(f"  score={r['score']:.4f}  {r['content']}")

    # B) 混合检索（相似度 + 新近度 + 重要性）—— 由 MemoryManager.recall 提供
    #    下面各列是「归一化 × 权重」后的贡献量，三者相加正好等于 combined。
    print("\n=== 混合检索（贡献量：combined = 相似度 + 新近度 + 重要性）===")
    print(f"  {'combined':>8} = {'sim':>6} + {'recency':>7} + {'import':>6}   内容")
    for r in manager.recall(query, top_k=3, now=now):
        print(f"  {r['score']:>8.3f}   "
              f"{r['c_similarity']:>6.3f}   {r['c_recency']:>7.3f}   {r['c_importance']:>6.3f}   {r['content']}")

    print("\n观察：纯相似度里“美式咖啡”还排在“抹茶拿铁”前面（会返回过时偏好）；")
    print("      混合检索靠【新近度】把刚发生的“抹茶拿铁”顶到最前，更贴合当下。")


if __name__ == "__main__":
    main()
