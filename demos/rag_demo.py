"""
真实 RAG 记忆演示：真 embedding 模型 + 真向量库(ChromaDB)。

场景：agent 先“记住”了一堆关于用户的事实（长期记忆），
之后用户问一个问题，agent 不把全部事实塞回去，而是：
    1. 把问题 embed 成向量
    2. 去向量库里检索最相关的几条事实
    3. 只把这几条 + 问题交给 DeepSeek 作答
这就是真实产品里“长期记忆 / 知识库问答”的最小骨架。

⚠️ 必须用虚拟环境运行（RAG 依赖装在 .venv 里）：
       .venv/bin/python demos/rag_demo.py
   首次运行会自动下载 embedding 模型（约几百 MB），请耐心等。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import RagMemory
from llm import has_key, deepseek_llm

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

# agent 的长期记忆：一堆零散事实（可以想象成几百上千条，这里放几条演示）
FACTS = [
    "用户叫小明，是一名后端工程师。",
    "小明养了一只叫咪咪的橘猫。",
    "小明对花生过敏。",
    "小明住在杭州，喜欢周末去西湖骑车。",
    "小明最喜欢的编程语言是 Go。",
    "小明的生日是 3 月 14 日。",
]

QUESTION = "帮我推荐一家餐厅，有什么忌口要注意的吗？"


def main():
    print("加载 embedding 模型中（首次会下载）……")
    mem = RagMemory(persist_path=str(DATA_DIR / "vector_store.json"), top_k=3)
    mem.clear()

    # 1) 把事实逐条写进向量库
    for f in FACTS:
        mem.add("user", f)
    print(f"已把 {len(FACTS)} 条事实存进向量库。\n")

    # 2) 针对问题，检索最相关的几条（注意：不是全部）
    retrieved = mem.get_context(QUESTION)
    print(f"问题：{QUESTION}")
    print(f"检索到最相关的 {len(retrieved)} 条记忆（分数越接近 1 越相关）：")
    for r in retrieved:
        print(f"  (score={r['score']}) {r['content']}")
    print()

    # 关键观察：忌口相关的“对花生过敏”应该被排在最前，
    # 而“最喜欢 Go 语言”这种无关事实不会被取出来 —— 这就是检索的价值。

    # 3) 只把检索到的记忆 + 问题交给模型作答
    if not has_key():
        print("（未配置 DeepSeek key，跳过生成回答。检索部分已完成演示。）")
        return

    context_text = "\n".join(f"- {r['content']}" for r in retrieved)
    messages = [
        {"role": "system", "content":
            "你是用户的私人助理。下面是关于用户的已知信息，请据此回答，"
            "尤其注意健康和安全相关的点。\n\n已知信息：\n" + context_text},
        {"role": "user", "content": QUESTION},
    ]
    print("DeepSeek 回答：")
    print(deepseek_llm(messages))


if __name__ == "__main__":
    main()
