"""
基于 LoCoMo 的 agent memory 评估。

LoCoMo 评估三步（Mem0 / Zep 等论文都用这套）：
    1. 把整段【多会话对话】灌进记忆系统（ingest）
    2. 每个问题：从记忆里【检索】相关内容 -> 让 LLM【作答】
    3. 用 LLM-as-judge 判对错，并按【问题类别】分别统计准确率

为什么按类别看：不同记忆系统的强项不同。单跳靠直接检索就能答；多跳/时序考验
能不能把分散信息拼起来；对抗题（问的事对话里根本没提）考验会不会“编”。

数据：默认读 eval/locomo10.json（真实 LoCoMo，10 段对话、每段几百轮、共 ~2000 题），
没有则退回自带小样本 eval/locomo_sample.json。真实数据很大，所以默认只评 --samples 1 段、
每段 --limit 10 题；想全量把它们设 0（会很贵）。

可切换记忆后端做对比（评估的意义就在对比）：
    --adapter fullcontext  不做检索，把整段对话塞进 prompt（长上下文基线 / 上限参考）
    --adapter rag          RagMemory：每句进向量库，按问题相似度召回（需 .venv）
    --adapter mem0         Mem0：抽取事实 + 四操作入库，再检索（灌数据每轮都调 LLM，很贵，需 .venv）

用法：
    python3          eval/run_locomo.py --adapter fullcontext --samples 1 --limit 8
    .venv/bin/python eval/run_locomo.py --adapter rag --samples 1 --limit 10 --top-k 8
    .venv/bin/python eval/run_locomo.py --adapter rag --data eval/locomo_sample.json --samples 0
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm import has_key, deepseek_llm

DATA_DIR = ROOT / ".data"
NO_INFO = "No information available"

CATEGORY_NAMES = {
    1: "多跳推理",
    2: "时序推理",
    3: "开放域知识",
    4: "单跳检索",
    5: "对抗(不可答)",
}


# ============================================================
# 解析 LoCoMo 对话：按会话顺序，展平成 (date, speaker, dia_id, text)
# ============================================================
def iter_turns(conversation: dict):
    sessions = []
    for key in conversation:
        if key.startswith("session_") and not key.endswith("_date_time"):
            num = int(key.split("_")[1])
            date = conversation.get(f"session_{num}_date_time", "")
            sessions.append((num, date, conversation[key]))
    for _, date, turns in sorted(sessions, key=lambda x: x[0]):
        for t in turns:
            text = t.get("text", "")
            if t.get("blip_caption"):                 # 多模态轮：把图片描述并进文本，别丢信息
                text = f"{text} [分享了一张图片：{t['blip_caption']}]".strip()
            if not text:
                continue
            yield date, t["speaker"], t.get("dia_id", ""), text


def gold_answer(q: dict) -> str:
    """取标准答案：普通题用 answer；类别5(对抗)题 answer 为 None，用 adversarial_answer。"""
    ans = q.get("answer")
    if ans is None:
        ans = q.get("adversarial_answer")
    return str(ans) if ans is not None else ""


# ============================================================
# 记忆后端适配器：统一 ingest(灌数据) / retrieve(取上下文)
# ============================================================
class FullContextAdapter:
    """不做检索：把整段对话原样喂回。长上下文基线，也是“检索不丢信息”的上限参考。"""
    name = "fullcontext"

    def __init__(self):
        self.lines = []

    def ingest(self, date, speaker, text):
        self.lines.append(f"[{date}] {speaker}: {text}")

    def retrieve(self, question):
        return list(self.lines)


class RagAdapter:
    """RagMemory：每句作为一条记忆入向量库，按问题相似度召回 top_k。"""
    name = "rag"

    def __init__(self, top_k):
        from memory import RagMemory
        self.mem = RagMemory(persist_path=str(DATA_DIR / "eval_rag.json"), top_k=top_k)
        self.mem.clear()
        self.top_k = top_k

    def ingest(self, date, speaker, text):
        self.mem.add(speaker, f"[{date}] {speaker}: {text}")

    def retrieve(self, question):
        return [h["content"] for h in self.mem.get_context(question, top_k=self.top_k)]


class Mem0Adapter:
    """Mem0：每句先抽取事实 + 四操作入库，再按问题检索。"""
    name = "mem0"

    def __init__(self, top_k):
        from memory import RagMemory, Mem0
        mem = RagMemory(persist_path=str(DATA_DIR / "eval_mem0.json"), top_k=top_k)
        mem.clear()
        self.m = Mem0(mem, llm_fn=deepseek_llm, search_top_k=top_k)

    def ingest(self, date, speaker, text):
        self.m.add(f"{speaker}（{date}）说：{text}")

    def retrieve(self, question):
        return [h["content"] for h in self.m.search(question)]


def build_adapter(name, top_k):
    if name == "fullcontext":
        return FullContextAdapter()
    if name == "rag":
        return RagAdapter(top_k)
    if name == "mem0":
        return Mem0Adapter(top_k)
    raise ValueError(f"未知 adapter：{name}")


# ============================================================
# 作答 + 判分
# ============================================================
def answer_question(question, context_lines):
    context = "\n".join(context_lines) if context_lines else "（无相关记忆）"
    messages = [
        {"role": "system", "content":
            "你要根据【记忆片段】回答关于一段对话的问题。涉及对话中的具体事实必须以记忆为准；"
            "若问题需要常识/世界知识来补全（例如“马拉松全程多长”），可结合常识作答。"
            f"简洁给出答案。如果记忆里没有任何相关线索，就原样回答：{NO_INFO}"},
        {"role": "user", "content": f"记忆片段：\n{context}\n\n问题：{question}"},
    ]
    return deepseek_llm(messages).strip()


def judge(question, gold, predicted):
    """LLM-as-judge：语义等价即算对；不可答题需预测也表示无信息。返回 True/False。"""
    messages = [
        {"role": "system", "content":
            "你是阅卷老师。给你【问题】【标准答案】【预测答案】，判断预测是否正确。"
            "只要预测传达了与标准答案一致的关键信息即算对（允许措辞不同、更详细）。"
            "若标准答案表示‘没有相关信息/无法回答’，则预测也必须表示没有相关信息才算对。"
            '只输出 JSON：{"correct": true/false}'},
        {"role": "user", "content":
            f"问题：{question}\n标准答案：{gold}\n预测答案：{predicted}"},
    ]
    raw = deepseek_llm(messages)
    try:
        import re
        m = re.search(r"\{.*\}", raw, re.S)
        return bool(json.loads(m.group(0)).get("correct")) if m else False
    except Exception:
        return False


# ============================================================
# 主流程
# ============================================================
def main():
    here = Path(__file__).resolve().parent
    default_data = here / "locomo10.json"
    if not default_data.exists():
        default_data = here / "locomo_sample.json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="rag", choices=["fullcontext", "rag", "mem0"])
    ap.add_argument("--data", default=str(default_data))
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--samples", type=int, default=1,
                    help="评测前几段对话（真实 LoCoMo 有 10 段，很大；0=全部）")
    ap.add_argument("--limit", type=int, default=10,
                    help="每段对话最多评测几道题（真实每段上百题；0=不限）")
    args = ap.parse_args()

    if not has_key():
        print("未配置 DeepSeek key，无法评测（作答与判分都要调用模型）。")
        return

    dataset = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.samples:
        dataset = dataset[:args.samples]
    print(f"数据集：{args.data}  评测 {len(dataset)} 段对话  记忆后端：{args.adapter}"
          f"  每段至多 {args.limit or '∞'} 题\n")

    total, correct = 0, 0
    by_cat = {}   # category -> [对, 总]

    for sample in dataset:
        adapter = build_adapter(args.adapter, args.top_k)

        # 1) 灌入整段对话
        turns = list(iter_turns(sample["conversation"]))
        print(f"[{sample.get('sample_id','?')}] 灌入 {len(turns)} 轮对话到 {adapter.name} ……")
        for date, speaker, _dia, text in turns:
            adapter.ingest(date, speaker, text)

        # 2) 逐题：检索 -> 作答 -> 判分
        qa = sample["qa"]
        if args.limit:
            qa = qa[:args.limit]
        for q in qa:
            gold = gold_answer(q)
            ctx = adapter.retrieve(q["question"])
            pred = answer_question(q["question"], ctx)
            ok = judge(q["question"], gold, pred)

            cat = q.get("category", 0)
            by_cat.setdefault(cat, [0, 0])
            by_cat[cat][1] += 1
            by_cat[cat][0] += int(ok)
            total += 1
            correct += int(ok)

            mark = "✓" if ok else "✗"
            print(f"  {mark} [{CATEGORY_NAMES.get(cat, cat)}] {q['question']}")
            print(f"      预测：{pred}")
            if not ok:
                print(f"      标准：{gold}")

    # 3) 汇总
    print("\n===== 评测结果 =====")
    print(f"总体准确率：{correct}/{total} = {correct/total:.1%}" if total else "无题目")
    print("按类别：")
    for cat in sorted(by_cat):
        c, n = by_cat[cat]
        print(f"  {CATEGORY_NAMES.get(cat, cat):10} {c}/{n} = {c/n:.0%}")


if __name__ == "__main__":
    main()
