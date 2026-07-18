"""
策略二：摘要压缩。

历史一多，就把旧对话压成一段“摘要”，只保留最近几条原文。
兼顾长期记忆，但细节会在压缩中失真。

fake_summarizer 是假摘要（离线可跑）；真实项目把它换成真 LLM 摘要即可。
"""

from typing import List, Dict

from .base import BaseMemory


def fake_summarizer(messages: List[Dict[str, str]]) -> str:
    """
    假装是大模型做摘要。真实版本应调用 LLM：“把下面对话压缩成要点”。
    这里简单地把 user 说过的话拼起来，够演示用。
    """
    points = [m["content"] for m in messages if m["role"] == "user"]
    return "早前对话要点：" + "；".join(points)


class SummaryMemory(BaseMemory):
    def __init__(self, path: str, keep_recent: int = 2, summarizer=fake_summarizer):
        super().__init__(path)
        self.keep_recent = keep_recent      # 保留最近几条原文
        self.summarizer = summarizer        # 可换成真 LLM 摘要函数

    def get_context(self) -> List[Dict[str, str]]:
        if len(self.messages) <= self.keep_recent:
            return self.messages
        old = self.messages[:-self.keep_recent]
        recent = self.messages[-self.keep_recent:]
        summary_msg = {"role": "system", "content": self.summarizer(old)}
        return [summary_msg] + recent
