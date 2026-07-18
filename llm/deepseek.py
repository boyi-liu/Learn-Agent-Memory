"""
DeepSeek API 调用。DeepSeek 完全兼容 OpenAI 接口，
所以用 openai SDK，只把 base_url 指向 DeepSeek 即可。
"""

from openai import OpenAI

from .config import get_api_key, get_model, has_key

_client = OpenAI(api_key=get_api_key(), base_url="https://api.deepseek.com")


def deepseek_llm(messages, model: str = None) -> str:
    """
    输入 messages（就是 memory.get_context() 的返回值，格式天然吻合），
    调用 DeepSeek，返回回复纯文本。

    messages 里的 system / user / assistant 三种 role，各种 Memory 产出的
    正好是这个格式，无需任何转换 —— 这就是记忆与模型解耦的价值。
    """
    if not has_key():
        raise RuntimeError(
            "没配好 DeepSeek key。二选一：\n"
            '    1) export DEEPSEEK_API_KEY="sk-你的key"\n'
            "    2) 在 config.yaml 里填上 deepseek_api_key"
        )
    resp = _client.chat.completions.create(
        model=model or get_model(),
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content
