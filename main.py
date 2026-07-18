"""
main.py —— 一个真正支持记忆的对话 harness。

它把整个项目的零件组装成一个可交互的 agent，用一个 while 循环不停对话。
记忆采用真实 agent 常见的【双层结构】：

    短期记忆（recent）：本轮会话最近几句原文，保证多轮对话连贯。
    长期记忆（RagMemory）：每句都 embed 存进向量库、跨会话持久化到磁盘；
                           每轮回答前，用当前问题去检索最相关的几条旧记忆。

每一轮的处理流程（就是记忆系统的核心）：
    1. 读用户输入
    2. 用输入去长期记忆里【检索】相关记忆
    3. 组装 prompt = 系统设定 + 检索到的长期记忆 + 最近几句 + 本次输入
    4. 调 DeepSeek 生成回复
    5. 把这轮的 user / assistant 都【写回】短期 + 长期记忆

斜杠命令：/mem 查看检索、/clear 清空长期记忆、/help 帮助、/exit 退出。

⚠️ 用虚拟环境运行（长期记忆依赖 embedding 模型）：
       .venv/bin/python main.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from llm import has_key, deepseek_llm
from memory import RagMemory

DATA_DIR = ROOT / ".data"
RECENT_WINDOW = 6          # 短期记忆保留最近多少条消息（约 3 轮问答）
RETRIEVE_TOP_K = 3         # 每轮从长期记忆检索几条

SYSTEM_PROMPT = (
    "你是一个拥有长期记忆的私人助理。回答时请参考下面提供的“相关记忆”"
    "（那是你过去和用户交流时记住的事），让回答更贴合用户。"
    "如果记忆里没有相关信息，就正常回答，不要编造。"
)


def build_messages(system_prompt, retrieved, recent, user_input):
    """把系统设定 + 检索到的长期记忆 + 最近对话 + 本次输入拼成 messages。"""
    messages = [{"role": "system", "content": system_prompt}]

    if retrieved:
        memory_text = "\n".join(f"- {r['content']}" for r in retrieved)
        messages.append({
            "role": "system",
            "content": "【相关记忆】（从长期记忆检索到，可能有用）：\n" + memory_text,
        })

    messages.extend(recent)                               # 最近几句原文
    messages.append({"role": "user", "content": user_input})
    return messages


def main():
    if not has_key():
        print("未配置 DeepSeek key。请在 config.yaml 填 deepseek_api_key 后再运行。")
        return

    print("加载长期记忆（首次会下载 embedding 模型）……")
    long_term = RagMemory(persist_path=str(DATA_DIR / "main_memory.json"),
                          top_k=RETRIEVE_TOP_K)
    recent = []   # 短期记忆：本进程内的最近对话

    print("\n记忆 agent 已就绪。直接输入对话；/help 看命令，/exit 退出。")
    print(f"（长期记忆已有 {long_term.store.count()} 条，跨会话保留）\n")

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # ---- 斜杠命令 ----
        if user_input in ("/exit", "/quit"):
            print("再见！长期记忆已保存在 .data/main_memory.json。")
            break
        if user_input == "/help":
            print("命令：/mem 查看本轮检索到的记忆 | /clear 清空长期记忆 | /exit 退出")
            continue
        if user_input == "/clear":
            long_term.clear()
            recent.clear()
            print("已清空长期记忆和当前会话。")
            continue
        if user_input == "/mem":
            print(f"长期记忆共 {long_term.store.count()} 条；短期记忆 {len(recent)} 条。")
            continue

        # ---- 1) 检索长期记忆 ----
        retrieved = long_term.get_context(user_input)

        # ---- 2) 组装 prompt 并调用模型 ----
        messages = build_messages(SYSTEM_PROMPT, retrieved, recent, user_input)
        reply = deepseek_llm(messages)
        print(f"助手 > {reply}\n")

        # ---- 3) 写回记忆（短期 + 长期）----
        recent.append({"role": "user", "content": user_input})
        recent.append({"role": "assistant", "content": reply})
        recent[:] = recent[-RECENT_WINDOW:]          # 短期只留最近 N 条
        long_term.add("user", user_input)            # 长期永久留存
        long_term.add("assistant", reply)


if __name__ == "__main__":
    main()
