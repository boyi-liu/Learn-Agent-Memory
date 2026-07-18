"""
策略一：滑动窗口。

只把最近 N 条塞回给模型。最简单省钱，但会忘掉久远但重要的事。
"""

from typing import List, Dict

from .base import BaseMemory


class SlidingWindowMemory(BaseMemory):
    def __init__(self, path: str, window: int = 4):
        super().__init__(path)
        self.window = window  # 最多塞回最近多少条

    def get_context(self) -> List[Dict[str, str]]:
        # 全都记着（不丢），但只【喂回】最近 window 条
        return self.messages[-self.window:]
