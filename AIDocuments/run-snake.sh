#!/usr/bin/env bash
#
# 用 conda 环境 `laya` 跑 laya-snake 演示。
#
# 用法：
#   ./run-snake.sh                              交互式游玩（需要在真终端里跑）
#   ./run-snake.sh --max-speed                  交互式，每步都重新决策、不做节奏控制
#   ./run-snake.sh --headless --steps 300       无界面跑一段，结束后打印统计
#   ./run-snake.sh --record run.jsonl           记录每一步的决策与棋盘，之后可 export 成图
#   ./run-snake.sh benchmark --seeds 3          跑基准测试
#   ./run-snake.sh export run.jsonl --output run.png    把录制渲染成图/视频
#   ./run-snake.sh --help                       看完整参数
#
# 两个必须项已经替你填好：解释器（conda laya）和权重路径（models/laya-multilingual-mlx）。
# 中文必须用多语言版 checkpoint，英文版 laya-mlx 读不了中文。

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAKE_BIN="/opt/anaconda3/envs/laya/bin/laya-snake"
MODEL_DIR="$REPO_DIR/models/laya-multilingual-mlx"

if [[ ! -x "$SNAKE_BIN" ]]; then
    echo "找不到 laya-snake：$SNAKE_BIN" >&2
    echo "conda 环境 laya 可能被删了。重建：" >&2
    echo "  cd $REPO_DIR && /opt/anaconda3/envs/laya/bin/pip install --no-compile -e '.[demo]'" >&2
    exit 1
fi

sub="${1:-}"

# export 只读 JSONL 渲染画面，不需要模型，也不需要终端。
if [[ "$sub" == "export" ]]; then
    exec "$SNAKE_BIN" "$@"
fi

if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
    echo "找不到本地权重：$MODEL_DIR" >&2
    echo "下载（本机直连 huggingface.co 不通，必须走镜像）：" >&2
    echo "  HF_ENDPOINT=https://hf-mirror.com hf download aac6fef/laya-multilingual-mlx --local-dir \"$MODEL_DIR\"" >&2
    exit 1
fi

# 你显式传了 --model 就听你的，否则用本地权重。
has_model=0
for arg in "$@"; do
    case "$arg" in
        --model|--model=*) has_model=1 ;;
    esac
done

# benchmark 是子命令，argparse 要求它排在选项前面，所以 --model 得跟在子命令后面。
if [[ "$sub" == "benchmark" ]]; then
    shift
    if [[ $has_model -eq 1 ]]; then
        exec "$SNAKE_BIN" benchmark "$@"
    else
        exec "$SNAKE_BIN" benchmark --model "$MODEL_DIR" "$@"
    fi
fi

# 交互界面要在真 TTY 上画；被重定向或在 IDE 运行窗口里跑时提前说一声。
if [[ ! -t 0 || ! -t 1 ]]; then
    case " $* " in
        *" --headless "*|*" --help "*|*" -h "*) ;;
        *) echo "提示：当前不是真终端，交互界面画不出来。加 --headless，或在 Terminal 里跑。" >&2 ;;
    esac
else
    # 棋盘有硬门槛：至少 104 列 × 35 行（ui.py 里 max(104,...)，--width 也降不下去）。
    # 不够大时 snake 只会打一句 "Resize terminal..." 然后干等，什么都看不到。
    read -r cur_rows cur_cols <<<"$(stty size 2>/dev/null || echo '0 0')" || true
    if (( cur_rows > 0 && cur_cols > 0 )) && (( cur_cols < 104 || cur_rows < 35 )); then
        echo "提示：当前终端 ${cur_cols} 列 × ${cur_rows} 行，snake 至少要 104 列 × 35 行，画面画不出来。" >&2
        echo "      把终端窗口拉大或字号调小（全屏通常够了），也可以加 --headless 看统计。" >&2
    fi
fi

if [[ $has_model -eq 1 ]]; then
    exec "$SNAKE_BIN" "$@"
else
    exec "$SNAKE_BIN" --model "$MODEL_DIR" "$@"
fi
