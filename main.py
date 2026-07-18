"""
main.py —— 一个真正支持记忆的对话 harness，用一个 while 循环不停对话。

用哪种记忆由 config.yaml 的 memory.backend 决定（naive/window/summary/rag/managed），
主循环完全不关心底层是哪种 —— 全靠 harness_memory 把它们统一成 context()/record() 两个动作。

每一轮的处理流程：
    1. 读用户输入
    2. backend.context(输入)  取要塞进 prompt 的记忆（历史 或 召回到的长期记忆）
    3. 组装 prompt = 系统设定 + 记忆 + 本次输入，调 DeepSeek
    4. backend.record(输入, 回复)  把这一轮存回记忆

斜杠命令：/mem 看记忆条数、/clear 清空、/help 帮助、/exit 退出。

⚠️ 若 backend 选了 rag/managed，需要用虚拟环境运行（要加载 embedding 模型）：
       .venv/bin/python main.py
   选 naive/window/summary 则不需要模型，普通 python3 也行。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from llm import has_key, deepseek_llm, get_config
from harness_memory import build_memory

SYSTEM_PROMPT = (
    "你是一个拥有记忆的私人助理。回答时请参考下面提供的记忆"
    "（那是你过去和用户交流时记住的事），让回答更贴合用户。"
    "如果记忆里没有相关信息，就正常回答，不要编造。"
)


def main():
    if not has_key():
        print("未配置 DeepSeek key。请在 config.yaml 填 deepseek_api_key 后再运行。")
        return

    mem_cfg = get_config().get("memory", {})
    print(f"正在按 config.yaml 启动记忆后端：{mem_cfg.get('backend', 'managed')} ……")
    backend = build_memory(mem_cfg, llm_fn=deepseek_llm)

    print(f"\n记忆 agent 已就绪（后端：{backend.name}）。/help 看命令，/exit 退出。")
    print(f"（当前记忆已有 {backend.count()} 条）\n")

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
            print("再见！记忆已保存在 .data/ 下。")
            break
        if user_input == "/help":
            print("命令：/mem 查看记忆条数 | /clear 清空记忆 | /exit 退出")
            continue
        if user_input == "/clear":
            backend.clear()
            print("已清空记忆。")
            continue
        if user_input == "/mem":
            print(f"当前记忆共 {backend.count()} 条。")
            continue

        # ---- 一轮对话：取记忆 -> 组装 -> 生成 -> 存回 ----
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(backend.context(user_input))          # 记忆（历史/召回）
        messages.append({"role": "user", "content": user_input})

        reply = deepseek_llm(messages)
        print(f"助手 > {reply}\n")

        backend.record(user_input, reply)


if __name__ == "__main__":
    main()
