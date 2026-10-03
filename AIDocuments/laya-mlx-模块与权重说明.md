# laya-mlx 模块与权重说明

这份文档只讲两件事：**代码里每个模块是干什么的**，以及 **`models/` 下两个权重目录为什么是两个、有什么区别**。
原理解释和上手步骤在《[laya-mlx 项目梳理](laya-mlx-项目梳理.md)》里，这份不重复。

---

## 一、代码模块

### 1.1 运行时核心：`laya_mlx/`

这是整个项目的主体，14 个文件约 2360 行。按职责分四组。

**第一组：真正跑模型的那条链**（缺一个都跑不起来）

| 文件 | 行数 | 作用 |
|---|---|---|
| `agent.py` | 296 | 对外主 API。`load()` 建模型、`Agent.predict()`（等价于 `system_one()`）出结果、`prepare()` 预热。文件末尾还把 `Agent` 别名成 `RLAgent`、把 `system_one` 别名成 `predict`，两种写法都能用。 |
| `model.py` | 249 | 神经网络本身：ModernBERT 编码器 + 三个决策头（choice / score / noul）。纯 `mlx.nn` 实现，没有任何 torch 依赖。 |
| `tokenizer.py` | 27 | 加载 checkpoint 里的 Rust tokenizer。存在的意义是绕开 `transformers`，这样整个运行时不带 torch。 |
| `common.py` | 138 | 提示词拼装与概率处理：`build_prefix()` / `build_sequence()` 负责把 state 和问题拼成模型输入，`confidence_from_probs()` 算归一化熵，还有温度钳制。名字叫 common，其实是 agent 和 shortlist 共用的底层，不是通用工具箱。 |
| `prepared.py` | 59 | 提示词前缀缓存 `PrefixCache`。只缓存 tokenize 过的前缀，不缓存编码器状态和预测结果。 |

**第二组：命令行**

| 文件 | 行数 | 作用 |
|---|---|---|
| `cli.py` | 55 | `laya-mlx` 命令的实现，两个子命令：`predict`（读 state 和问题 JSON 出结果）、`convert`（转换 checkpoint）。`python -m laya_mlx` 走的也是它。 |
| `convert.py` | 48 | 把上游 checkpoint 的权重名改成 MLX 的命名并转成指定精度。只在发布权重时用得到。 |

**第三组：业务场景辅助**（不参与核心推理，但对外导出）

| 文件 | 行数 | 作用 |
|---|---|---|
| `email.py` | 115 | 邮件场景：`clean_email_body()` 去掉免责声明一类的样板文字，`email_state()` 把邮件拼成 state，`email_questions()` 生成邮件分类的问题模板。 |
| `presets.py` | 193 | 五套现成问题模板：工单分诊 `triage_questions()`、邮件、`guard_questions()`、内容审核 `moderation_questions()`、路由 `router_questions()`。 |
| `lang.py` | 463 | 语言与文字系统检测。核心判断是「这段文本英文版 checkpoint 读不读得了」，判断依据是 Unicode 区块 + 拉丁文语言启发式。纯正则和字符表，不加载模型。 |
| `router.py` | 404 | `Router` 类：按 `lang.py` 的检测结果自动挑 checkpoint，并负责按需加载 / 淘汰。`lang.py` 是它的判断依据。 |
| `shortlist.py` | 270 | 选项上百个时的嵌入粗排。把选项先用模型自身编码器按余弦相似度筛一遍再交给决策模型。实测在 100 选项的场景下是帮倒忙的，属于可选开关。 |

**第四组：门面**

`__init__.py`（38 行）把上面各模块的公共 API 全部 re-export 出来，所以写代码时一个 `import laya_mlx` 就够了，不用关心东西在哪个文件里。`__main__.py`（3 行）只是把 `python -m laya_mlx` 转到 `cli.main`。

### 1.2 终端演示：`laya_mlx/snake/`

贪吃蛇演示完全自成一块，6 个文件约 1390 行。

| 文件 | 行数 | 作用 |
|---|---|---|
| `cli.py` | 294 | 命令入口（`laya-snake`）。三个子命令：默认的交互式游戏、`benchmark` 跑分、`export` 把录制渲染成图。键盘监听、主循环、终端尺寸检查也在这里。 |
| `game.py` | 156 | 棋盘逻辑：蛇的移动、吃食物、判定碰撞，以及给每个方向算「安不安全」。 |
| `ui.py` | 169 | 画面渲染。终端布局在这里写死：**最少 104 列 × 35 行**（`layout_size()` 里的 `max(104, ...)`），不够大只显示一句提示。 |
| `policy.py` | 220 | 把游戏局面翻译成 Laya 的问题，拿模型概率决定往哪走。里面对 `from laya_mlx import Agent` 是**函数内导入**，用它拿模型。 |
| `replay.py` | 256 | 录制与回放：把每一步存成 JSONL，再渲染成 PNG / MP4 / GIF。 |
| `benchmark.py` | 293 | 速度与稳定性跑分，多 seed 对比、统计死亡数和安全接管次数。 |

### 1.3 仓库其他目录

| 目录 | 内容 |
|---|---|
| `benchmarks/` | 正式基准测试。`run.py` 调度、`accuracy.py` 对比上游 FP32 与本地 FP16 的准确率、`latency.svg`/`.png` 是产出的延迟图、`report.py` 生成 `BENCHMARKS.md`。`snake_*.py` 三个是蛇的专项对比。 |
| `experiments/` | 研究性实验，不进生产。`engineering/` 下是编译、量化（q4/q8）、手写 Metal kernel 的成对实验记录；`math_costs.py`、`math_spectrum.py` 是纯 CPU 的静态算力估算；`snake_runtime.py` 是蛇的编译优化消融实验。 |
| `examples/` | 官方最小示例：`quickstart.py` + 配套的 `state.json`、`questions.json`。 |
| `scripts/` | `prepare_hub.py`，把转换好的权重整理成可上传的目录结构（只写本地文件，不上传）。 |
| `tests/` | 7 个文件的测试集。`test_model.py` 验权重、`test_runtime.py` 验推理、`test_router.py` 验路由与语言检测、`test_shortlist.py`、`test_email.py`、`test_snake.py`（逐帧回放录制文件校验）。 |
| `docs/` | 研究文档：`MATH_10X_RESEARCH.md`、`ENGINEERING_10X_RESEARCH.md`、`PERFORMANCE_RESEARCH.md`，以及蛇的 `SNAKE_DEMO.md` / `SNAKE_BENCHMARKS.md` / `SNAKE_OPTIMIZATION.md`。 |

### 1.4 谁依赖谁

```
laya_mlx/snake/            ← 演示层，只用 Agent（policy.py 里函数内导入）
    └── policy.py ────────┐
                          ▼
laya_mlx/agent.py ──────► model.py ──► tokenizer.py
    │        │                 common.py
    │        └── prepared.py   （model 和 shortlist 共用 common）
    ├── shortlist.py ──► common.py
    └── cli.py / convert.py

laya_mlx/router.py ──► lang.py          （路由只用语言检测，不加载模型）
laya_mlx/email.py、presets.py           （纯文本处理，不依赖别的模块）
```

有一个副作用值得知道：`snake/policy.py` 里写的是 `from laya_mlx import Agent`，这会触发 `__init__.py`，把 email / presets / lang / router / shortlist 全部连带加载进来。蛇本身一个都不用它们。

---

## 二、权重

### 2.1 为什么是两个目录

**不是同一个模型切成了两份，是两个不同的上游模型，各自转成 MLX 后发成了两个独立的 HuggingFace 仓库，下载下来自然就是并列的两个目录。**

命名规则很简单：**上游仓库名 + `-mlx`**，`-mlx` 表示「这是给它做的 MLX 移植版」。

```
上游 convaiinnovations/laya               → 移植版 aac6fef/laya-mlx               → 本地 models/laya-mlx/
上游 convaiinnovations/laya-multilingual  → 移植版 aac6fef/laya-multilingual-mlx  → 本地 models/laya-multilingual-mlx/
```

这两个上游模型出自同一个训练框架、同一套 `<answer>|<types>` 决策契约（题库、输出格式、三种问题原语完全一致），但**编码器换了个底座**：英文版用 ModernBERT-large，多语言版换成 mmBERT-base。所以它们不是大小号关系，是两个独立训练出来的模型。

顺带说一句，上游 `convaiinnovations/laya` 那个仓库本身是个 bundle —— 三个 checkpoint 都塞在里面用子目录区分（`multilingual/`、`typed-decisions/`）。发布者同时也给每个模型开了独立仓库，本机这两个目录就是按独立仓库下的，所以目录名和 HF 仓库名一样。

### 2.2 两个 checkpoint 的区别

以下数据全部来自本机 `models/` 下的实际文件（`encoder/config.json`、`rl_agent_config.json`、`manifest.json`）。

| | `models/laya-mlx/` | `models/laya-multilingual-mlx/` |
|---|---|---|
| 转自 | `convaiinnovations/laya` | `convaiinnovations/laya-multilingual` |
| 编码器 | `answerdotai/ModernBERT-large` | `jhu-clsp/mmBERT-base` |
| 编码器规模 | hidden 1024 / 28 层 / 16 头 | hidden 768 / 22 层 / 12 头 |
| 词表大小 | 50,368（英文 BPE） | 256,000（多语言） |
| 参数量 / 权重张量数 | 421M / 206 个 | 322M / 170 个 |
| `model.safetensors` | 843 MB | 644 MB |
| 目录总大小 | 807 MB | 647 MB |
| 总上下文 `max_len` | 512 | 1024 |
| 问题区上限 `head_max_len` | 192 | 256 |
| 训练量 | 1 epoch / 7313 updates | 4 epoch / 15987 updates |
| 校准温度 | 分档校准（按选项数细分 6 档） | 全是 1.0（未做分档校准） |
| 语言覆盖 | 仅英文 | 100+ 种语言 |

**能力上差在哪**（数据出自 `laya_mlx/router.py` 的注释，同一套 17,416 道题的对比基准）：

- 英文：ModernBERT-large 版明显更强 —— MASSIVE 意图分类英文 0.783 对 0.657，XNLI 英文 0.860 对 0.843。
- 非英文：多语言版反超 —— XNLI 非英文 0.731 对 0.521，差 21 个点。
- 最要命的是英文版在非拉丁文字上的表现：印地语 0.100、韩语 0.103（20 选项，随机猜是 0.050），**而且它自己还报高置信度**（印地语的 ECE 是 0.855）。所以英文版不能用中文、日文、韩文、阿拉伯文这类输入 —— 不是效果差一点，是直接崩且不自知。

结论很直接：**中文场景必须用多语言版**。

### 2.2.1 为什么必须分开：词表层面的硬证据

上面这些准确率差距只是结果，根本原因是**英文版的词表里压根没有汉字**。它的词表是 50,368 个纯英文 BPE，遇到中文只能退回到 UTF-8 字节级切分。本机实测两个 tokenizer 对同一批输入的切分（已去掉首尾特殊 token）：

| 输入 | 英文版（50,368 词表） | 多语言版（256,000 词表） |
|---|---|---|
| `refund` | `refund` — 1 个 | `refund` — 1 个 |
| `退款` | `éĢ` `Ģ` `æ¬` `¾` — 4 个碎片 | `退` `款` — 2 个 |
| `重复扣款` | `éĩį` `å¤` `į` `æī` `£` `æ¬` `¾` — 8 个碎片 | `重` `复` `扣` `款` — 4 个 |
| `こんにちは` | `ãģĵ` `ãĤĵ` `ãģ«` `ãģ¡` `ãģ¯` — 5 个碎片 | `こんにちは` — 1 个 |

左列那些乱码不是显示问题 —— 它们是汉字 UTF-8 编码的三个字节被当成独立符号后显示的字符。也就是说，把中文喂给英文版，模型看到的不是"退款"，而是四个毫无语义的符号，而且同样的话要多花两倍的 token 预算。

所以这不是「训练数据里中文少、效果差一点」的问题，**是分词器层面就不认识汉字**。这也解释了为什么英文版在非拉丁文字上会掉到接近随机（印地语 0.100）却仍然报出高置信度：它压根没读懂输入，只是在瞎猜，而校准机制给不出"我不知道"这个信号。

### 2.3 第三个：`typed-decisions`

`router.py` 里其实登记了三个 checkpoint，本机只下了两个。第三个叫 typed-decisions（`convaiinnovations/laya-typed-decisions`，同样是 ModernBERT-large 底座），专门为四类工作流微调过：客服工单、发票处理、安全事件、Agent 链路可观测性。

它**不会被自动选中**，除非显式传 `task="typed_decisions"` 或打开 `auto_task_detection=True`。这是故意的 —— 它只擅长那四个固定场景，做默认值会是个静默的坑。

### 2.4 怎么选

一句话：**看输入是什么语言。**

- 全是英文 → `models/laya-mlx`，又快又准（参数多但上下文短，编码器更宽）。
- 有中文或其他语言 → `models/laya-multilingual-mlx`。本机所有中文实验用的都是它。
- 不确定，或者要做服务 → 用 `Router`，让 `lang.py` 按文字系统自动分派；服务场景记得 `Router(preload=True)`，否则跨语言请求会反复冷加载。

顺带一个数字上的差异：多语言版的 `head_max_len` 是 256（英文版 192），所以两者在「选项很多」的场景下能塞进的问题描述长度不一样 —— 实测多语言版在 100 个选项时每个选项只剩 3~4 个 token 的空间。

### 2.4.1 中英混排：模型能处理，自动路由原本会送错（已修）

**混排文本本身对多语言版没有障碍。** 它的词表同时含中文和英文，实测切分完全正常：

```
输入：重复扣款了请把钱退给我。By the way, the login page keeps showing an error.
切分：重 复 扣 款 了 请 把 钱 退 给我 。 By the way , the login page keeps showing an error .
```

**但 `Router` 原本会把混排判给英文版** —— 只要拉丁字母多于汉字就判错，不管汉字占多少。本机实测（修改前）：

| 输入 | 汉字占字母比 | 修改前选中 | 修改后 |
|---|---|---|---|
| 我上个月被重复扣款了，请尽快退款 | 100% | multilingual | multilingual |
| 我上个月被重复扣款 duplicate charge，请尽快 refund | 36% | **english** | multilingual |
| I was billed twice 重复扣款 last month, please 尽快 refund | 14% | **english** | multilingual |
| I was billed twice, please refund 尽快 | 7% | **english** | multilingual |
| I was billed twice last month, please refund asap | 0% | english | english（未误伤） |

原因是两层：`lang.detect_script()` 返回的是**字母数最多的那种文字**，混排时拉丁字母稍多就判成 `latin`；而 `router.route()` 只看 `script` 和 `is_english`，**完全没用 `analyse()` 已经算出来的 `non_latin_fraction`**（上面第二行那个 36% 就是被丢掉的信号）。后果就是混排输入被送到英文版，汉字部分变成字节碎片 —— 实测纯中文上英文版偶尔还能蒙对（概率 0.96），但掺进英文后把握明显下滑（0.96 → 0.84），多语言版基本不动（0.994 → 0.998）。

**修改**（`laya_mlx/router.py`，2026-10-03）：加常量 `MIXED_NON_LATIN_FRACTION = 0.05`，在 `route()` 的英文分支前插一条判断 ——

```python
elif float(det["non_latin_fraction"]) >= MIXED_NON_LATIN_FRACTION:
    key = "multilingual"
    reason = (
        "mixed script: %.0f%% of letters are not Latin; the English checkpoint "
        "cannot read them" % (100 * float(det["non_latin_fraction"]))
    )
```

改完 141 个测试全通过（1 个 skip 是原有的），纯英文、纯法文、纯日文的判定都没被误伤。

**残留的边界**：阈值是 5%，所以很长的英文里夹一两个汉字（比如末尾一句「谢谢」）仍会判英文版 —— 占比低于阈值。真要覆盖这种情况就显式指定模型。混排场景最稳的写法：

```python
agent = laya.load("models/laya-multilingual-mlx")     # 直接指定
router.predict(state, questions, model="multilingual") # 或强制路由
```

另外两个细节：`state` 传 dict 时，`lang.py` 只按 value 判断、忽略 key（注释写的是「key 通常是英文」），所以英文的字段名不会干扰路由；反过来，如果中文只出现在 key 里就会被漏掉。还有问题定义（instructions 和 criteria）的语言尽量和 state 一致，之前实测中英混搭的标签是会掉分的。

### 2.5 权重目录里都有什么

两个目录结构完全一致，都是自包含的，可以直接用 `--model` 或 `laya.load()` 指过去：

```
models/<name>/
├── model.safetensors        权重本体（FP16）
├── mlx_config.json          格式版本、来源仓库、源 revision
├── rl_agent_config.json     编码器名、上下文长度、温度配置、训练记录
├── encoder/config.json      编码器架构参数
├── tokenizer/               词表与 tokenizer 配置
├── manifest.json            每个文件的字节数与 sha256（校验用）
├── validation.json          数值一致性与稳定性实测记录
├── README.md / LICENSE / NOTICE
└── .gitattributes
```

`agent.py` 的 `resolve_model()` 会检查 `model.safetensors`、`rl_agent_config.json`、`encoder/config.json` 三个文件是否齐全，缺任何一个都会直接报「Not a complete Laya checkpoint」，不会带着半个权重跑出莫名其妙的结果。

### 2.6 一处容易踩的差异：温度

英文版的 `rl_agent_config.json` 里有一张 `temperature_by_options` 表，按问题类型和选项数量细分成 6 档校准温度（比如 2 个选项用 1.906，11 个以上用 0.101）；多语言版这张表是空的、三档温度全是 1.0。

这影响的是概率的绝对值 —— 同样是「最高分选项」，两个模型给出的概率数值不能直接拿来互相比较，`confidence` 更不行（它本来就不是选中项的概率，而是归一化熵）。

---

## 三、速查

```bash
# 模型在哪
models/laya-mlx/                 # 英文
models/laya-multilingual-mlx/    # 多语言（中文用这个）

# 模块怎么用
import laya_mlx as laya          # 门面，全部 API 从这里进
laya.load(path)                  # agent.py
laya.Router()                    # router.py + lang.py
laya.predict_shortlist(...)      # shortlist.py
laya.email_state(...)            # email.py
laya.triage_questions()          # presets.py

# 命令
laya-mlx predict ...             # cli.py
laya-snake                       # snake/cli.py
```
