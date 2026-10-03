"""电商售后意图识别：语义高度重叠的 qid 测试。

抄自 ~/Desktop/myHub/RAG-Engineeing/jev/test_ecommerce_intents.py，
只把云端 SDK 换成本地 laya（改法与 similar_qids.py 相同）。
题目定义与测试数据保持原样。

跑法：
    cd AIDocuments/jev-on-laya
    /opt/anaconda3/envs/laya/bin/python test_ecommerce_intents.py

函数名不用 test_ 前缀（原因见 similar_qids.py 的说明）：它不是 pytest 用例。
"""

import pathlib

import laya_mlx as laya

MODEL = pathlib.Path(__file__).resolve().parents[2] / "models" / "laya-multilingual-mlx"


def run_ecommerce_intents(agent, messages: list[str]):
    """电商售后意图识别：语义高度重叠的 qid 测试"""

    # 五个语义高度相近的售后 qid
    qid_criteria = {
        "return_refund": "用户要求退货退款，需要把商品寄回给商家",
        "refund_only": "用户要求仅退款，不寄回商品，商品自行处理",
        "exchange": "用户要求更换同款商品，不涉及退款",
        "reship": "用户要求补发缺少的配件或商品，不退款不退货",
        "price_protection": "用户要求退还降价差价，商品保留不退货",
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
                    "instructions": "这条用户消息最匹配哪个售后处理方式？",
                    "criteria": qid_criteria,
                }
            },
        )

        answer = response["answers"]["qid"]
        print(f"Top-1 qid: {answer['choice']}")
        print(f"置信度:    {answer['confidence']:.3f}")
        print(f"完整概率分布:")
        for qid, prob in sorted(answer["probabilities"].items(),
                                key=lambda x: -x[1]):
            bar = "█" * int(prob * 40)
            print(f"  {qid:20s} {prob:.4f}  {bar}")


def main():
    agent = laya.load(MODEL, dtype="float16")

    test_messages = [
        "这个东西我不要了，把钱退给我。",
        "我收到的商品和描述不符，怎么处理？",
        "少了一个零件，怎么办？",
        "买完就降价了，能退我钱吗？",
    ]

    run_ecommerce_intents(agent, test_messages)


if __name__ == "__main__":
    main()
