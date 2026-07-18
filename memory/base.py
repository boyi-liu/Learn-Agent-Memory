"""
所有记忆的公共基类：负责 add / 存盘 / 读盘。
子类只需决定 get_context() 怎么取（该塞回什么给模型）。
"""

import json
import os
from typing import List, Dict


class BaseMemory:
    def __init__(self, path: str):
        self.path = path
        # messages 是记忆的载体：一个 list，每项形如
        #   {"role": "user"/"assistant"/"system", "content": "..."}
        self.messages: List[Dict[str, str]] = []
        self._load()

    # ---------- 写入 ----------
    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._save()

    # ---------- 读取（子类重写这里以实现不同策略） ----------
    def get_context(self) -> List[Dict[str, str]]:
        """默认：全都给（即 naive 行为）。"""
        return self.messages

    def clear(self) -> None:
        self.messages = []
        self._save()

    # ---------- 持久化 ----------
    def _save(self) -> None:
        # 目录不存在就先建，避免写盘报错
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.messages = json.load(f)
