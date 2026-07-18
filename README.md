# Learn Agent Memory

从零学习 agent memory 的最小可运行代码。核心就一句话：
**把每轮对话存下来，下次取出来喂回给模型。**

## 目录结构

```
├── config.yaml            你的 DeepSeek key（私密，已被 .gitignore 挡住）
├── config.example.yaml    配置模板（可分享）
├── memory/                记忆策略（本项目的重点）
│   ├── base.py            公共基类：add / 存盘 / 读盘
│   ├── naive.py           朴素：全部记、全部塞回
│   ├── sliding_window.py  策略一：滑动窗口
│   ├── summary.py         策略二：摘要压缩
│   ├── vector.py          策略三：向量检索(玩具版RAG)
│   ├── vector_store.py    手搓的向量库（numpy 余弦检索 + JSON 落盘）
│   └── rag_memory.py      真实RAG：sentence-transformers + 手搓向量库
├── llm/                   模型调用
│   ├── config.py          读 config.yaml（或环境变量）
│   └── deepseek.py        DeepSeek 调用（OpenAI 兼容）
└── demos/                 可运行示例
    ├── agent_demo.py      记忆 + 真模型跑一轮对话
    └── compare_demo.py    四种记忆同台对比
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
