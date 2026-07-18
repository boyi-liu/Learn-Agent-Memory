"""
把记忆接到真模型（DeepSeek）上，看记忆怎么起作用。

有 key（config.yaml 或环境变量）就用真 DeepSeek，没有就退回 fake_llm，
所以没配 key 也能跑通流程。不管真假模型，memory 代码都不用改。

运行：
    python3 demos/agent_demo.py
"""

import sys
from pathlib import Path

# 让本文件无论从哪运行，都能 import 到项目里的 memory / llm 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import NaiveMemory
from llm import has_key, deepseek_llm

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"




def llm(messages):
    if has_key():
        return deepseek_llm(messages)
    return "（未配置 DeepSeek API key）"  # 兜底，没配 key 就不调模型


def chat(memory: NaiveMemory, user_input: str) -> str:
    memory.add("user", user_input)               # 1. 用户这句写进记忆
    reply = llm(memory.get_context())            # 2. 取出全部记忆喂给模型
    memory.add("assistant", reply)               # 3. 模型回复也写进记忆
    return reply


if __name__ == "__main__":
    print("模式：", "真 DeepSeek" if has_key() else "兜底 fake_llm（未配 key）", "\n")

    mem = NaiveMemory(str(DATA_DIR / "agent_memory.json"))
    mem.clear()

    for line in ["我叫小明", "今天天气不错", "还记得我叫什么吗？"]:
        print(f"用户: {line}")
        print(f"Agent: {chat(mem, line)}\n")

    print(f"共 {len(mem.get_context())} 条消息，已存进 .data/agent_memory.json")
