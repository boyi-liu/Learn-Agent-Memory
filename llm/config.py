"""
读取配置。优先级：环境变量 > config.yaml。

这样两种方式都支持：
    - 临时用环境变量：export DEEPSEEK_API_KEY="sk-..."
    - 长期用配置文件：把 key 填进项目根目录的 config.yaml
"""

from __future__ import annotations  # 让 str | None 这种注解在 Python 3.9 也能用

import os
from pathlib import Path

import yaml

# 项目根目录 = 本文件（llm/config.py）的上一级
_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"

_PLACEHOLDER = "sk-你的key填这里"


def _load_yaml() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_api_key() -> str | None:
    """先环境变量，再 config.yaml。"""
    return os.environ.get("DEEPSEEK_API_KEY") or _load_yaml().get("deepseek_api_key")


def get_model() -> str:
    return _load_yaml().get("deepseek_model", "deepseek-chat")


def has_key() -> bool:
    """是否配好了可用的 key（占位符不算）。"""
    key = get_api_key()
    return bool(key) and key != _PLACEHOLDER
