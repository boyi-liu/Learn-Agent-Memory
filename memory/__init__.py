"""memory 包：一个基类 + 四种记忆策略，对外统一 add() / get_context() 接口。"""

from .base import BaseMemory
from .naive import NaiveMemory
from .sliding_window import SlidingWindowMemory
from .summary import SummaryMemory
from .vector import VectorMemory
from .vector_store import VectorStore

__all__ = [
    "BaseMemory",
    "NaiveMemory",
    "SlidingWindowMemory",
    "SummaryMemory",
    "VectorMemory",
    "VectorStore",
    "RagMemory",
]


def __getattr__(name):
    # 延迟加载：只有真正用到 RagMemory 时才 import 它的重依赖（torch 等），
    # 平时 import memory 包保持轻量。
    if name == "RagMemory":
        from .rag_memory import RagMemory
        return RagMemory
    raise AttributeError(f"module 'memory' has no attribute {name!r}")
