"""测试本地 laya 对语义高度相近 qid 的区分能力。

抄自 ~/Desktop/myHub/RAG-Engineeing/jev/similar_qids.py，只把云端 SDK 换成本地 laya：
    client.system_one(model="jev-latest", state=msg, questions={"qid": Choice(...)})
        -> agent.system_one(msg, {"qid": {"type": "choice", ...}})
    answer.choice / answer.confidence / answer.probabilities
        -> answer["choice"] / answer["confidence"] / answer["probabilities"]
题目定义与测试数据保持原样。

跑法：
    cd AIDocuments/jev-on-laya
    /opt/anaconda3/envs/laya/bin/python similar_qids.py

函数名不用 test_ 前缀：这不是 pytest 用例（它要 agent / messages 两个参数，
pytest 收集不到），加了前缀 PyCharm 会当成测试文件、硬走 pytest 运行器。
"""

import pathlib

import laya_mlx as laya

MODEL = pathlib.Path(__file__).resolve().parents[2] / "models" / "laya-multilingual-mlx"


def run_similar_qids(agent, messages: list[str]):
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

    for idx, msg in enumerate(messages, 1):
        print(f"\n{'='*60}")
        print(f"测试 {idx}: {msg}")
        print(f"{'='*60}")

        response = agent.system_one(
            msg,
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

    test_messages = [
        "我不想打车了，怎么取消订单？",

        "我取消订单为什么还要收我钱？",

        "司机怎么把我的订单取消了？",

        "司机接单半天了还没过来，我能取消吗？",

        "司机让我自己把订单取消掉，说他不想接了。",

        "订单怎么突然自己没了，我也没取消啊？",

        "司机让我取消订单重新叫一个，这是什么情况？",

        "我把订单取消了，被扣了20块取消费，能退吗？",

        "司机已经接单了，但是一直不来，我取消后还扣费了。",
    ]

    run_similar_qids(agent, test_messages)


if __name__ == "__main__":
    main()
