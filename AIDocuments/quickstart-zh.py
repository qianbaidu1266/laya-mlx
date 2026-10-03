"""laya-mlx 本机最小可跑示例（中文输入，多语言 checkpoint）。

跑法：
    cd laya-mlx
    /opt/anaconda3/envs/laya/bin/python AIDocuments/quickstart-zh.py

权重已经下到仓库的 models/ 目录，全离线运行。要重新下载才需要设镜像：
    export HF_ENDPOINT=https://hf-mirror.com
    hf download aac6fef/laya-multilingual-mlx --local-dir models/laya-multilingual-mlx
"""

import json
import time
from pathlib import Path

import laya_mlx as laya

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models" / "laya-multilingual-mlx"  # 中文必须用多语言版，英文版 laya-mlx 读不了

STATE = "发票被重复扣款了，我已经付了两次，请尽快退款。"

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "这个问题该由哪个团队处理？",
        "criteria": {
            "billing": "发票、付款、退款",
            "technical": "系统故障与报错",
            "sales": "新购买",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "这个请求有多紧急？",
        "criteria": ["不紧急", "尽快", "非常紧急"],
    },
    "refund": {
        "type": "noul",
        "instructions": "客户是否要求退款？",
    },
}


def main():
    t0 = time.perf_counter()
    agent = laya.load(MODEL, dtype="float16")
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    result = agent.predict(STATE, QUESTIONS)
    t3 = time.perf_counter()

    rounds = 5
    t4 = time.perf_counter()
    for _ in range(rounds):
        agent.predict(STATE, QUESTIONS)
    t5 = time.perf_counter()

    print(f"权重加载      {t1 - t0:.2f} s")
    print(f"首次预测      {(t3 - t2) * 1000:.1f} ms（含 MLX 预热）")
    print(f"稳态单次      {(t5 - t4) / rounds * 1000:.1f} ms（{rounds} 次平均）")
    print(f"输入 token    {result['usage']['input_tokens']}，输出 token {result['usage']['output_tokens']}")
    print("---")
    print(json.dumps(result["answers"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
