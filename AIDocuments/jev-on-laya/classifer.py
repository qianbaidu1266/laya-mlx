"""两阶段意图识别：先判一级意图域，再在命中的域内匹配具体 qid。

抄自 ~/Desktop/myHub/RAG-Engineeing/jev/classifer.py，只把云端 SDK 换成本地 laya：
    client.system_one(model="jev-latest", state=msg, questions={"x": Choice(...)})
        -> agent.system_one(msg, {"x": {"type": "choice", "instructions": ..., "criteria": ...}})
    response.answers["x"].choice  ->  response["answers"]["x"]["choice"]
题目定义与测试数据保持原样。

跑法：
    cd AIDocuments/jev-on-laya
    /opt/anaconda3/envs/laya/bin/python classifer.py
"""

import pathlib

import laya_mlx as laya

MODEL = pathlib.Path(__file__).resolve().parents[2] / "models" / "laya-multilingual-mlx"


# 模拟的 qid 层级结构（实际场景中可从数据库加载）
QID_TREE = {
    "billing": {
        "qids": {
            "refund_request": "用户要求退款或退费",
            "duplicate_charge": "用户反馈被重复扣款",
            "invoice_issue": "发票开具或修改问题",
        }
    },
    "technical": {
        "qids": {
            "login_failure": "用户无法登录账号",
            "app_crash": "App 闪退或无法打开",
            "payment_gateway_error": "支付网关报错",
        }
    },
    "logistics": {
        "qids": {
            "delivery_delay": "订单配送延迟",
            "wrong_item": "收到错误商品",
            "return_process": "退货流程咨询",
        }
    },
}


def classify_intent(agent, user_message: str) -> dict:
    """
    两阶段意图识别：
    1. 用 Choice 判断一级意图域
    2. 在命中的域内，用 Choice 匹配具体 qid
    """
    # ── 阶段一：一级意图域分类 ──
    domain_response = agent.system_one(
        user_message,
        {
            "domain": {
                "type": "choice",
                "instructions": "这条用户消息属于哪个业务领域？",
                "criteria": {
                    "billing": "付款、退款、账单、扣款相关问题",
                    "technical": "技术故障、系统异常、登录问题",
                    "logistics": "配送、物流、退换货相关问题",
                },
            }
        },
    )

    domain_answer = domain_response["answers"]["domain"]
    domain = domain_answer["choice"]
    domain_confidence = domain_answer["confidence"]

    # ── 阶段二：域内 qid 匹配 ──
    candidates = QID_TREE[domain]["qids"]

    qid_response = agent.system_one(
        user_message,
        {
            "qid": {
                "type": "choice",
                "instructions": f"在“{domain}”领域下，这条用户消息最匹配哪个标准问？",
                "criteria": candidates,
            }
        },
    )

    qid_answer = qid_response["answers"]["qid"]
    matched_qid = qid_answer["choice"]
    qid_confidence = qid_answer["confidence"]

    return {
        "domain": domain,
        "domain_confidence": domain_confidence,
        "domain_probabilities": domain_answer["probabilities"],
        "qid": matched_qid,
        "qid_confidence": qid_confidence,
        "qid_probabilities": qid_answer["probabilities"],
        "qid_description": candidates.get(matched_qid, ""),
    }


def main():
    agent = laya.load(MODEL, dtype="float16")

    # 模拟用户输入
    user_message = "我被重复扣款了，同一个订单扣了两次，请帮我退款。"

    result = classify_intent(agent, user_message)

    print("=" * 50)
    print(f"用户消息：{user_message}")
    print("=" * 50)
    print(f"一级意图域：{result['domain']}")
    print(f"  置信度：{result['domain_confidence']}")
    print("  候选域概率分布：")
    for name, prob in sorted(result["domain_probabilities"].items(), key=lambda x: -x[1]):
        print(f"    {name:20s} {prob:.4f}  {'█' * int(prob * 40)}")
    print(f"匹配 qid：{result['qid']}")
    print(f"  qid 描述：{result['qid_description']}")
    print(f"  置信度：{result['qid_confidence']}")
    print("  候选 qid 概率分布：")
    for name, prob in sorted(result["qid_probabilities"].items(), key=lambda x: -x[1]):
        print(f"    {name:20s} {prob:.4f}  {'█' * int(prob * 40)}")

    # 置信度阈值判断示例
    if result["qid_confidence"] < 0.7:
        print("\n⚠️ qid 置信度较低，建议转人工或进入下一轮澄清。")


if __name__ == "__main__":
    main()
