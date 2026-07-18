"""llm 包：对外暴露 DeepSeek 调用和 key 检查。"""

from .config import has_key, get_model
from .deepseek import deepseek_llm

__all__ = ["deepseek_llm", "has_key", "get_model"]
