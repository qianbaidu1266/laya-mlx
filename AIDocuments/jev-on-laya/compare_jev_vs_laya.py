"""Jev（云端 API） vs Laya（本地 MLX）准确性对比。

四个测试集全部来自 ~/Desktop/myHub/RAG-Engineeing/jev/ 的原题与原始数据，
一字未改；只是把调用方式换成本地 laya，并给每条补了人工标注的期望答案。

三个被测对象：
    jev              TypeSafe AI 云端服务（model="jev-latest"）
    laya-multilingual  本地 models/laya-multilingual-mlx
    laya-english       本地 models/laya-mlx（参照列：中文场景不该选它）

评分口径：
    strict   只认 gold
    loose    gold 或 alt 都算对（alt = 人工判定的"另一个也说得通"的答案）

跑法：
    cd AIDocuments/jev-on-laya
    /opt/anaconda3/envs/laya/bin/python compare_jev_vs_laya.py

结果同时写入 compare-results.json，便于文档引用与复现。
"""

import json
import os
import pathlib
import time

import laya_mlx as laya

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_FILE = pathlib.Path.home() / "Desktop/myHub/RAG-Engineeing/.env"

# ────────────────────────────── 测试数据 ──────────────────────────────
# 题目定义与 messages 全部照抄 jev 原脚本；gold / alt 是本次新增的人工标注。

RIDE_CANCEL_CRITERIA = {
    "cancel_by_passenger":
        "乘客主动取消订单，询问如何取消、取消操作或取消规则",
    "cancel_fee_by_passenger":
        "乘客主动取消订单后，被收取了取消费，询问为什么收费或要求退还取消费",
    "driver_cancel":
        "司机主动取消订单，乘客询问司机为什么取消、司机取消订单或司机取消导致订单结束",
    "driver_no_show":
        "司机已经接单，但长时间没有到达上车点，乘客想取消订单或询问司机为什么不来",
    "driver_asked_passenger_cancel":
        "司机要求或诱导乘客主动取消订单，例如让乘客自己取消、让乘客取消后重新叫车",
    "order_auto_cancel":
        "订单没有被乘客或司机主动取消，而是因为超时、系统原因等自动取消",
}

ECOMMERCE_CRITERIA = {
    "return_refund": "用户要求退货退款，需要把商品寄回给商家",
    "refund_only": "用户要求仅退款，不寄回商品，商品自行处理",
    "exchange": "用户要求更换同款商品，不涉及退款",
    "reship": "用户要求补发缺少的配件或商品，不退款不退货",
    "price_protection": "用户要求退还降价差价，商品保留不退货",
}

BILLING_QIDS = {
    "refund_request": "用户要求退款或退费",
    "duplicate_charge": "用户反馈被重复扣款",
    "invoice_issue": "发票开具或修改问题",
}
TECHNICAL_QIDS = {
    "login_failure": "用户无法登录账号",
    "app_crash": "App 闪退或无法打开",
    "payment_gateway_error": "支付网关报错",
}
LOGISTICS_QIDS = {
    "delivery_delay": "订单配送延迟",
    "wrong_item": "收到错误商品",
    "return_process": "退货流程咨询",
}
DOMAIN_CRITERIA = {
    "billing": "付款、退款、账单、扣款相关问题",
    "technical": "技术故障、系统异常、登录问题",
    "logistics": "配送、物流、退换货相关问题",
}

SUITES = [
    {
        "name": "similar_qids（网约车取消，6 选 1）",
        "instructions": "这条用户消息最匹配哪个标准问？",
        "criteria": RIDE_CANCEL_CRITERIA,
        "items": [
            {"text": "我不想打车了，怎么取消订单？", "gold": "cancel_by_passenger"},
            {"text": "我取消订单为什么还要收我钱？", "gold": "cancel_fee_by_passenger"},
            {"text": "司机怎么把我的订单取消了？", "gold": "driver_cancel"},
            {"text": "司机接单半天了还没过来，我能取消吗？", "gold": "driver_no_show"},
            {"text": "司机让我自己把订单取消掉，说他不想接了。", "gold": "driver_asked_passenger_cancel"},
            {"text": "订单怎么突然自己没了，我也没取消啊？", "gold": "order_auto_cancel"},
            {"text": "司机让我取消订单重新叫一个，这是什么情况？", "gold": "driver_asked_passenger_cancel"},
            {"text": "我把订单取消了，被扣了20块取消费，能退吗？", "gold": "cancel_fee_by_passenger"},
            {"text": "司机已经接单了，但是一直不来，我取消后还扣费了。",
             "gold": "cancel_fee_by_passenger", "alt": "driver_no_show"},
        ],
    },
    {
        "name": "ecommerce_intents（电商售后，5 选 1）",
        "instructions": "这条用户消息最匹配哪个售后处理方式？",
        "criteria": ECOMMERCE_CRITERIA,
        "items": [
            {"text": "这个东西我不要了，把钱退给我。", "gold": "return_refund", "alt": "refund_only"},
            {"text": "我收到的商品和描述不符，怎么处理？", "gold": "return_refund", "alt": "exchange"},
            {"text": "少了一个零件，怎么办？", "gold": "reship"},
            {"text": "买完就降价了，能退我钱吗？", "gold": "price_protection"},
        ],
    },
    {
        "name": "conversation_classifier（多轮对话，6 选 1）",
        "instructions": "这条用户消息最匹配哪个标准问？",
        "criteria": RIDE_CANCEL_CRITERIA,
        "items": [
            {
                "text": (
                    "用户：司机接单十几分钟了还没到。\n\n"
                    "AI：您好，请问司机现在还没有到达上车点吗？\n\n"
                    "用户：对，一直没来。\n\n"
                    "AI：您后来有取消订单吗？\n\n"
                    "用户：等了二十多分钟我就取消了。\n\n"
                    "AI：好的，请问您是想咨询取消订单的问题吗？\n\n"
                    "用户：我主要想问，司机一直不来，我取消为什么还扣我20块钱？"
                ),
                "gold": "cancel_fee_by_passenger",
                "alt": "driver_no_show",
            },
        ],
    },
    {
        "name": "classifer·阶段一（一级意图域，3 选 1）",
        "instructions": "这条用户消息属于哪个业务领域？",
        "criteria": DOMAIN_CRITERIA,
        "items": [
            {"text": "我被重复扣款了，同一个订单扣了两次，请帮我退款。", "gold": "billing"},
        ],
    },
    {
        "name": "classifer·阶段二（域内 qid，3 选 1）",
        "instructions": "在“billing”领域下，这条用户消息最匹配哪个标准问？",
        "criteria": BILLING_QIDS,
        "items": [
            {"text": "我被重复扣款了，同一个订单扣了两次，请帮我退款。",
             "gold": "duplicate_charge", "alt": "refund_request"},
        ],
    },
]


# ────────────────────────────── 引擎 ──────────────────────────────

def build_jev():
    from dotenv import load_dotenv
    from typesafe_sdk import TypeSafeClient, Choice

    load_dotenv(ENV_FILE)
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError(f"{ENV_FILE} 里没有 TYPESAFE_API_KEY")
    client = TypeSafeClient(api_key=api_key)

    def run(state, instructions, criteria):
        t0 = time.perf_counter()
        resp = client.system_one(
            model="jev-latest",
            state=state,
            questions={"q": Choice(instructions=instructions, criteria=criteria)},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        a = resp.answers["q"]
        probs = dict(a.probabilities)
        return {
            "choice": a.choice,
            "confidence": getattr(a, "confidence", None),
            "top_prob": probs.get(a.choice),
            "probabilities": probs,
            "ms": round(elapsed_ms, 1),
        }

    return run


def build_laya(model_path, dtype="float16"):
    agent = laya.load(str(model_path), dtype=dtype)

    # 预热一次，避免把首次编译/加载时间算进单条延迟
    agent.system_one("预热", {"q": {"type": "choice", "instructions": "x",
                                    "criteria": {"a": "a", "b": "b"}}})

    def run(state, instructions, criteria):
        t0 = time.perf_counter()
        resp = agent.system_one(
            state, {"q": {"type": "choice", "instructions": instructions,
                          "criteria": criteria}}
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        a = resp["answers"]["q"]
        probs = dict(a["probabilities"])
        return {
            "choice": a["choice"],
            "confidence": a.get("confidence"),
            "top_prob": probs.get(a["choice"]),
            "probabilities": probs,
            "ms": round(elapsed_ms, 1),
        }

    return run


# ────────────────────────────── 主流程 ──────────────────────────────

def evaluate(engines):
    results = {name: [] for name in engines}
    for suite in SUITES:
        for item in suite["items"]:
            for name, run in engines.items():
                r = run(item["text"], suite["instructions"], suite["criteria"])
                hit_strict = r["choice"] == item["gold"]
                hit_loose = hit_strict or (
                    "alt" in item and r["choice"] == item["alt"])
                results[name].append({
                    "suite": suite["name"],
                    "text": item["text"][:40],
                    "gold": item["gold"],
                    "alt": item.get("alt"),
                    "choice": r["choice"],
                    "hit_strict": hit_strict,
                    "hit_loose": hit_loose,
                    "confidence": r["confidence"],
                    "top_prob": r["top_prob"],
                    "ms": r["ms"],
                })
    return results


def summarize(results):
    summary = {}
    for name, rows in results.items():
        n = len(rows)
        s_hit = sum(r["hit_strict"] for r in rows)
        l_hit = sum(r["hit_loose"] for r in rows)
        ok_p = [r["top_prob"] for r in rows if r["hit_strict"] and r["top_prob"] is not None]
        bad_p = [r["top_prob"] for r in rows if not r["hit_loose"] and r["top_prob"] is not None]
        conf_bad = [r["confidence"] for r in rows
                    if not r["hit_loose"] and r["confidence"] is not None]
        summary[name] = {
            "n": n,
            "strict": s_hit,
            "loose": l_hit,
            "strict_rate": round(s_hit / n, 3),
            "loose_rate": round(l_hit / n, 3),
            "avg_ms": round(sum(r["ms"] for r in rows) / n, 1),
            "avg_top_prob_when_right": round(sum(ok_p) / len(ok_p), 3) if ok_p else None,
            "avg_top_prob_when_wrong": round(sum(bad_p) / len(bad_p), 3) if bad_p else None,
            "avg_confidence_when_wrong": round(sum(conf_bad) / len(conf_bad), 3) if conf_bad else None,
        }
    return summary


def main():
    engines = {}
    print("加载本地权重…")
    engines["laya-multilingual"] = build_laya(ROOT / "models" / "laya-multilingual-mlx")
    engines["laya-english"] = build_laya(ROOT / "models" / "laya-mlx")
    print("连接 Jev…")
    engines["jev"] = build_jev()

    print("跑测试…")
    results = evaluate(engines)
    summary = summarize(results)

    out = pathlib.Path(__file__).with_name("compare-results.json")
    out.write_text(json.dumps({"summary": summary, "detail": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print("%-20s %6s %8s %8s %10s %14s %14s" % (
        "引擎", "题数", "严格", "宽松", "平均延迟", "答对时P(top1)", "答错时P(top1)"))
    for name, s in summary.items():
        print("%-20s %6d %7.1f%% %7.1f%% %8.0fms %14s %14s" % (
            name, s["n"], s["strict_rate"] * 100, s["loose_rate"] * 100,
            s["avg_ms"], s["avg_top_prob_when_right"], s["avg_top_prob_when_wrong"]))

    print("\n" + "=" * 78)
    print("逐题明细")
    print("=" * 78)
    rows = results["jev"]
    for i, base in enumerate(rows):
        print("\n[%d] %s  ← 期望 %s%s" % (
            i + 1, base["text"], base["gold"],
            f"（也可接受 {base['alt']}）" if base["alt"] else ""))
        for name in ("jev", "laya-multilingual", "laya-english"):
            r = results[name][i]
            mark = "OK  " if r["hit_strict"] else ("~OK " if r["hit_loose"] else "MISS")
            print("    %-20s %s %-28s P=%-7s conf=%-7s %6.0fms" % (
                name, mark, r["choice"],
                "%.4f" % r["top_prob"] if r["top_prob"] is not None else "-",
                "%.4f" % r["confidence"] if r["confidence"] is not None else "-",
                r["ms"]))

    print("\n结果已写入 %s" % out.name)


if __name__ == "__main__":
    main()
