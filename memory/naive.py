"""
最朴素的记忆：把每一轮都记下来，下次【全部】塞回给模型。

这就是记忆的本质。它的唯一缺点是：历史越长 -> prompt 越大 ->
越贵越慢，最终撑爆上下文窗口。sliding_window / summary / vector 三种策略就是来解决这个的。

因为“全都给”正是基类的默认行为，所以 NaiveMemory 直接继承即可，
一行逻辑都不用写 —— 这本身就说明了 naive 有多朴素。
"""

from .base import BaseMemory


class NaiveMemory(BaseMemory):
    pass
