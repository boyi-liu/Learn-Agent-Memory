# Learn Agent Memory

从零学习 agent memory 的最小可运行代码。核心就一句话：
**把每轮对话存下来，下次取出来喂回给模型。**

## 目录结构

```
├── main.py                完整对话 harness：双层记忆 + while 循环
├── config.yaml            你的 DeepSeek key（私密，已被 .gitignore 挡住）
├── config.example.yaml    配置模板（可分享）
├── memory/                记忆策略（本项目的重点）
│   ├── base.py            公共基类：add / 存盘 / 读盘
│   ├── naive.py           朴素：全部记、全部塞回
│   ├── sliding_window.py  策略一：滑动窗口
│   ├── summary.py         策略二：摘要压缩
│   ├── vector.py          策略三：向量检索(玩具版RAG)
│   ├── vector_store.py    手搓的向量库（numpy 余弦检索 + 增删 + JSON 落盘）
│   ├── rag_memory.py      真实RAG：sentence-transformers + 手搓向量库
│   └── memory_manager.py  主动记忆管理：提炼 + 去重/更新（ADD/SKIP/UPDATE）
├── llm/                   模型调用
│   ├── config.py          读 config.yaml（或环境变量）
│   └── deepseek.py        DeepSeek 调用（OpenAI 兼容）
└── demos/                 可运行示例
    ├── agent_demo.py      记忆 + 真模型跑一轮对话
    ├── compare_demo.py    四种记忆同台对比
    ├── rag_demo.py        真实 RAG 检索问答
    ├── manager_demo.py    主动记忆管理：提炼/去重/更新演示
    └── hybrid_demo.py     混合检索：相似度 + 新近度 + 重要性
```

## 快速开始

```bash
# 1. 配置 key：复制模板，填上你的 key
cp config.example.yaml config.yaml
#   然后编辑 config.yaml，填 deepseek_api_key（去 platform.deepseek.com 拿）

# 2. 看四种记忆的区别（不用 key）
python3 demos/compare_demo.py

# 3. 接真模型跑一轮对话（配了 key 就用 DeepSeek，没配就退回假模型）
python3 demos/agent_demo.py
```

## 真实 RAG（需要虚拟环境）

`vector.py` 里的 `VectorMemory` 是玩具版（词袋当 embedding、list 当数据库）。
`memory/rag_memory.py` 是**真实版**：真 embedding 模型 + 手搓的向量库。

embedding 模型依赖较重（torch / sentence-transformers），装在独立虚拟环境 `.venv` 里，
不污染系统 Python：

```bash
# 建环境 + 装依赖（只做一次）
python3 -m venv .venv
.venv/bin/python -m pip install sentence-transformers openai pyyaml numpy

# 运行真实 RAG demo（首次会下载 embedding 模型，约几百 MB）
.venv/bin/python demos/rag_demo.py
```

- **embedding 模型**：`sentence-transformers`（多语言，支持中文），把文字变成向量
- **向量数据库**：`memory/vector_store.py`，手写的 numpy 余弦相似度检索，
  数据以 JSON 落盘到 `.data/vector_store.json`（可直接打开看：文字 + 一串数字）

## 完整对话 harness（main.py）

把所有零件组装成一个能连续对话、且**跨会话记忆**的 agent，用一个 `while` 循环驱动。
采用真实 agent 常见的**双层记忆**：

- **短期记忆**：本轮会话最近几句原文，保证多轮对话连贯
- **长期记忆**：RAG 向量库，每句都存、持久化到 `.data/main_memory.json`；
  每轮回答前用当前输入检索最相关的旧记忆

```bash
.venv/bin/python main.py
```

每一轮的核心流程：`检索长期记忆 → 组装 prompt(系统设定 + 检索记忆 + 最近几句 + 本次输入) → 调模型 → 写回短期与长期记忆`。
斜杠命令：`/mem` 查看记忆条数、`/clear` 清空长期记忆、`/help` 帮助、`/exit` 退出。

> 验证长期记忆：先运行一次告诉它一个事实并 `/exit`，再重新运行、问它那个事实——
> 新进程短期记忆为空，仍能答对，说明记忆确实跨会话保存在了磁盘上。

## 主动记忆管理（MemoryManager）

前面的记忆都是**被动**的：来一句存一句、只增不改，很快堆满噪音，还会因旧事实过期而自相矛盾。
`memory/memory_manager.py` 在 `RagMemory` 外面加一层**主动加工**，让系统从
「会检索的聊天记录」跨到「会自我整理的记忆」：

- **提炼(Write)**：每轮对话后让 LLM 判断有没有值得长期记的**事实**，并给每条评**重要性 1-10**，只存事实、忽略寒暄
- **消解(Reconcile)**：入库前先按相似度检索旧记忆，让 LLM 判断
  `ADD`（新知识）/ `SKIP`（重复）/ `UPDATE`（取代旧记忆，如搬家、换宠物）
- **召回(Recall)**：进阶检索，混合打分（见下一节）

它不 import `llm` 包，而是把 llm 函数**注入**进来（`MemoryManager(mem, llm_fn=deepseek_llm)`），
所以 `memory` 包保持独立、可用假 LLM 测试。运行演示：

```bash
.venv/bin/python demos/manager_demo.py
```

> 演示效果：闲聊不进库；「搬到北京」把旧的「住杭州」**更新**掉；重复的「对花生过敏」被 **SKIP**。

## 混合检索（相似度 + 新近度 + 重要性）

分层：`RagMemory` 只负责纯相似度召回（`get_context()`，也供上面的消解去重用）；
更聪明的检索策略放在管理层 **`MemoryManager.recall()`**，实现经典 Generative Agents 的**混合打分**：

```
最终分 = w_sim·相似度 + w_recency·新近度 + w_importance·重要性
```

- **相似度**：与查询的语义接近程度（余弦）
- **新近度**：越新越高，按 `0.5^(age/半衰期)` 指数衰减（`RagMemory.add` 自动记时间戳）
- **重要性**：这条记忆本身多重要（由 `MemoryManager` 提炼时让 LLM 评的 1-10 分）

采用**两阶段**：先用相似度粗筛候选池，再在池内按混合分重排序。

采用**两阶段**：先用相似度粗筛候选池，再在池内按混合分重排序。运行演示：

```bash
.venv/bin/python demos/hybrid_demo.py
```

> 演示效果：查“爱喝什么”，纯相似度把 30 天前的「美式咖啡」排在前；
> 混合检索靠新近度把 1 小时前的「抹茶拿铁」顶到第一。

## 四种记忆一览

| 策略 | 塞回给模型的内容 | 优点 | 缺点 |
|---|---|---|---|
| Naive | 全部历史 | 最全 | 历史一长就爆上下文 |
| 滑动窗口 | 最近 N 条 | 简单省钱 | 忘掉久远但重要的事 |
| 摘要压缩 | 旧对话摘要 + 最近几条 | 兼顾长期 | 细节失真 |
| 向量检索 RAG | 与当前问题最相关的几条 | 可扩展到海量 | 最复杂 |

四者对外都是同样的 `add()` / `get_context()` 接口，所以 agent 换记忆不用改代码。

## 两处是“假”的，真实项目怎么换
- `summary.fake_summarizer` → 换成真 LLM 摘要
- `vector.naive_embed` + `similarity` → 换成模型 embedding + 向量数据库
