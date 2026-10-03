"""同一批相近 qid，但 state 是一段多轮对话。

抄自 ~/Desktop/myHub/RAG-Engineeing/jev/conversation_classifier.py，
只把云端 SDK 换成本地 laya（改法与 similar_qids.py 相同）。
题目定义与对话内容保持原样。

跑法：
    cd AIDocuments/jev-on-laya
    /opt/anaconda3/envs/laya/bin/python conversation_classifier.py

函数名不用 test_ 前缀（原因见 similar_qids.py 的说明）：它不是 pytest 用例。
"""

import pathlib

import laya_mlx as laya

MODEL = pathlib.Path(__file__).resolve().parents[2] / "models" / "laya-multilingual-mlx"


def run_similar_qids(agent, messages: str):
    """测试 laya 对语义相近 qid 的区分能力"""

    # 语义高度相近的三个 qid
    qid_criteria = {
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


    print(f"\n{'='*60}")
    print(f"测试:")
    response = agent.system_one(
        messages,
        {
            "qid": {
                "type": "choice",
                "instructions": "这条用户消息最匹配哪个标准问？",
                "criteria": qid_criteria,
            }
        },
    )

    answer = response["answers"]["qid"]
    print(f"匹配 qid: {answer['choice']}")
    print(f"置信度:   {answer['confidence']:.3f}")
    print(f"概率分布:")
    for qid, prob in sorted(answer["probabilities"].items(),
                            key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {qid:20s} {prob:.4f}  {bar}")


def main():
    agent = laya.load(MODEL, dtype="float16")

    state = """
    用户：司机接单十几分钟了还没到。

    AI：您好，请问司机现在还没有到达上车点吗？

    用户：对，一直没来。

    AI：您后来有取消订单吗？

    用户：等了二十多分钟我就取消了。

    AI：好的，请问您是想咨询取消订单的问题吗？

    用户：我主要想问，司机一直不来，我取消为什么还扣我20块钱？
    """

    run_similar_qids(agent, state)


if __name__ == "__main__":
    main()
