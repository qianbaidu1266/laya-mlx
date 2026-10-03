# laya-mlx 项目梳理

> 本文对应仓库 HEAD `0a85951`（2026-10-03 读取）。所有结论都取自当时读到的代码、配置文件和 `benchmarks/results/` 里的原始测量数据，不引用 README 首页的宣传口径。

## 图目录

| 图号    | 标题                         | 所在章节 |
| ----- | -------------------------- | ---- |
| 图 2-1 | 单次推理链路                     | 2.3  |
| 图 2-2 | 输入序列是怎么拼出来的                | 2.4  |
| 图 5-1 | Jev / Laya / laya-mlx 三者关系 | 5.1  |

---

## 1. 它是不是本地启动的模型

**是。** laya-mlx 是一个纯本地的 Python 推理库，模型权重下载到本地后，整个推理过程不发一个网络包。它运行时依赖只有四个：`mlx`（Apple 的机器学习框架，限定 macOS arm64）、`numpy`、`huggingface-hub`（只在首次拉取权重时用）、`tokenizers`（Hugging Face 的 Rust 分词器）。没有 PyTorch，没有 Transformers 运行时，没有云端 API，也没有常驻服务进程——你在自己进程里 `import laya_mlx`，它就在你的进程里跑。

需要网络的只有两件事：一是首次 `laya.load("aac6fef/laya-mlx")` 会把 safetensors 权重拉到本地 HF 缓存；二是你显式调用 `convert` 导出一份权重。之后可以整机断网运行。

硬件上有硬门槛：**必须是 Apple Silicon 的 Mac**（`mlx>=0.32.2` 这条依赖写了 `sys_platform == 'darwin' and platform_machine == 'arm64'`），Python 3.11+，macOS 14+。Intel Mac、Linux、Windows 都装不上——这不是没做适配，而是 MLX 本身就只为 Apple 芯片存在。

---

## 2. 基本原理

### 2.1 它不是生成模型

常规 LLM 回答分类问题的路子是：生成一段文本（最好情况下是 JSON），你的代码再去解析出一个标签。laya-mlx 走的是另一条路：

```text
state + 结构化问题 → 双向编码器 → 决策头 → 概率分布
```

信息流是**单次前向**：没有 token-by-token 解码循环，输出的 `output_tokens` 恒为 0。你在请求里就把答案空间定义完了（有几个选项、选项叫什么、怎么描述），模型的活儿只是在这些预设位置之间算一个概率分布。这也是它能做到十几毫秒的主要原因——没有解码步数乘以单步延迟。

代价很直接：它**不会给你理由**，也说不出训练语料之外的话。可解释性、开放生成、多轮对话这些能力它一概没有。

### 2.2 三种问题原语

所有能用它做的事，都归结为三种原语：

| 原语       | 你问什么             | 返回什么                        | 代码里的 type id |
| -------- | ---------------- | --------------------------- | ------------ |
| `choice` | 从一组命名选项里选一个      | 命中标签 + 每个标签的概率 + confidence | 0            |
| `score`  | 在有序档位上打分（2–10 档） | 期望档位（浮点）+ 每档概率 + legend     | 1            |
| `noul`   | 一句话是真还是假         | P(true)，0–1                 | 2            |

一个 `predict` 请求可以同时带多个问题，它们会被拼成一个 batch 一起前向。`confidence` 不是概率本身，而是归一化香农熵：`1 - H(p) / log(k)`，`k` 是选项个数——它衡量的是「分布有多集中」，不是「有多大概率答对」。

### 2.3 一次推理走哪几步

**图 2-1　单次推理链路**

> 从你传入的 state 和问题定义，到最终的概率结果。注意第 3 步：每个问题各自走一遍编码器，state 不会被编码一次然后复用。

<svg viewBox="0 0 680 246" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,'PingFang SC','Helvetica Neue',sans-serif" height="246">
<title>单次推理链路</title>
<desc>输入经过 prompt 拼接、ModernBERT 编码、marker 打分、温度校准，输出 choice / score / noul 结果。</desc>
<defs>
<marker id="ar1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<g>
<rect x="40" y="50" width="180" height="56" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="130" y="68" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#0C447C">输入</text>
<text x="130" y="86" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#185FA5">state + 带类型的问题</text>
</g>
<g>
<rect x="250" y="50" width="180" height="56" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="340" y="68" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#0C447C">拼 prompt</text>
<text x="340" y="86" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#185FA5">每个选项前插 [MASK]</text>
</g>
<g>
<rect x="460" y="50" width="180" height="56" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="550" y="68" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#0C447C">ModernBERT 编码</text>
<text x="550" y="86" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#185FA5">MLX 双向一次前向</text>
</g>
<path d="M220 78 H244" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar1)"/>
<path d="M430 78 H454" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar1)"/>
<path d="M550 106 V140 H130 V164" fill="none" stroke="#888780" stroke-width="1.5" stroke-linejoin="round" marker-end="url(#ar1)"/>
<g>
<rect x="40" y="170" width="180" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="130" y="188" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#085041">取 marker 向量</text>
<text x="130" y="206" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0F6E56">scorer 打到每个选项上</text>
</g>
<g>
<rect x="250" y="170" width="180" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="340" y="188" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#085041">温度校准 + softmax</text>
<text x="340" y="206" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0F6E56">异常温度截断到 [0.5, 5]</text>
</g>
<g>
<rect x="460" y="170" width="180" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="550" y="188" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#085041">输出</text>
<text x="550" y="206" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0F6E56">choice / score / noul</text>
</g>
<path d="M220 198 H244" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar1)"/>
<path d="M430 198 H454" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar1)"/>
</svg>

落到代码上，`Agent.system_one`（`predict` 是它的别名，`laya_mlx/agent.py:233`）做四件事：`prepare` 把每个问题拼成 token 序列；按 `batch_size`（默认 16）切块；`collate_items` 补齐成矩形张量；`forward` 跑一次模型。最后在 NumPy 侧做温度缩放和 softmax，拼出答案字典。

### 2.4 输入序列怎么拼的

这是理解它各种限制的钥匙。`build_sequence`（`laya_mlx/common.py:88`）产出的序列长这样：

```text
[CLS] <类型> question: <指令> [SEP] [MASK] 选项1 [MASK] 选项2 ... [SEP] <state 正文> [SEP]
```

**图 2-2　输入序列是怎么拼出来的**

> 每个选项前面插一个 `[MASK]`，模型只在这些位置取隐藏向量交给 scorer。上方两条括号分别是两段 token 预算。

<svg viewBox="0 0 680 260" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,'PingFang SC','Helvetica Neue',sans-serif">
<title>输入序列结构</title>
<desc>序列由 CLS、问题说明、若干 MASK 选项、SEP 与 state 正文组成；head_max_len 覆盖到第二个 SEP，max_len 覆盖整条序列。</desc>
<defs>
<marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<path d="M40 68 V62 H504 V68" fill="none" stroke="#888780" stroke-width="0.5"/>
<text x="272" y="54" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#5F5E5A">head_max_len（所有选项共享这一段预算）</text>
<path d="M40 92 V86 H636 V92" fill="none" stroke="#888780" stroke-width="0.5"/>
<text x="338" y="78" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#5F5E5A">max_len（含 state 正文）</text>
<rect x="40" y="110" width="52" height="44" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="66" y="132" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0C447C">[CLS]</text>
<rect x="95" y="110" width="76" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="133" y="132" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#444441">类型 + 指令</text>
<rect x="174" y="110" width="52" height="44" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="200" y="132" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0C447C">[SEP]</text>
<rect x="229" y="110" width="94" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="276" y="132" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#444441">[MASK] 选项A</text>
<rect x="326" y="110" width="94" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="373" y="132" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#444441">[MASK] 选项B</text>
<rect x="423" y="110" width="26" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="436" y="132" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#444441">…</text>
<rect x="452" y="110" width="52" height="44" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="478" y="132" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0C447C">[SEP]</text>
<rect x="507" y="110" width="74" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="544" y="132" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#444441">state 正文</text>
<rect x="584" y="110" width="52" height="44" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="610" y="132" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0C447C">[SEP]</text>
<path d="M276 154 V190" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar2)"/>
<path d="M373 154 V190" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar2)"/>
<rect x="150" y="196" width="380" height="44" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="340" y="218" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#085041">scorer 只在这些 [MASK] 位置取隐藏向量打分</text>
</svg>

由此可以推出三条重要行为：

- **每个选项是一个 `[MASK]` 位置**，`markers` 记录这些位置的下标。模型前向完以后，scorer 只在这几个位置上取隐藏向量，过一个 `LayerNorm → Linear → GELU → Linear(1)` 得到每个选项的 logit（`laya_mlx/model.py:209`）。padding 位置被填成 `-1e4`，softmax 后自然归零。
- **所有选项共享 `head_max_len` 这段预算**（默认 192）。单个选项还会被硬截断到 48 个 token（`common.py:71`）。所以选项一多，每个选项分到的 token 就少——选项太多时报 `Question 'x' has too many options for the token budget`，这不是 bug，是预算不够。
- **`max_len` 管整条序列**（默认 512，多语言版 1024）。超出的 state 正文会被截断（`common.py:103`，默认截尾不截头）。

### 2.5 概率是怎么校准出来的

模型输出的是 logit，要经过温度缩放才成为可解释的概率。`Agent` 在加载时读 checkpoint 自带的 `temperature`（三种问题类型各一个）和 `temperature_by_options`（按 `类型:选项数分桶`，桶为 `2 / 3-5 / 6-10 / 11+`），按 `(类型, 桶)` 选一个温度除下去再 softmax。

这里有一个值得单独说的工程决定（`common.py:126`）：**温度被强制截断到 `[0.5, 5.0]`**。原因是官方 checkpoint 里 `choice:11+` 这一桶的值是 `0.1006`——除以 0.1 等于把 logit 放大约十倍，一个 0.24 的 top 概率会被发布成 0.99。任何拿 confidence 做阈值判断的调用方都会被告知「一个抛硬币的结果是确定结论」。laya-mlx 拒绝照搬，截断后加载时会发 `RuntimeWarning` 点名被改的桶，同时把原值留在 `agent.temperature_raw` 和 `agent.temperature_by_options_raw` 上供查验。

### 2.6 返回值长什么样

`predict` 返回一个字典：

```python
{
  "model": "laya-rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.42,
      "probabilities": {"billing": 0.91, "technical": 0.05, "sales": 0.02, "other": 0.02},
      "action": {"act_probability": 0.13}
    }
  },
  "usage": {"input_tokens": 93, "output_tokens": 0}
}
```


几个实操要点：`choice` 的概率值保留上游的四位小数；`score` 的 `score` 是零基期望档位（float），`legend` 给回档位到描述的映射；`noul` 返回 `noul` 即 P(true)。`action.act_probability` 来自一个独立的 action head，它吃的是「CLS 向量 + top1 概率 + top1-top2 差距 + 归一化熵 + 选项数」这几个特征（`model.py:231`），跟上面那套类别概率不是一回事。

---

## 3. 功能模块

### 3.1 代码地图

```
laya_mlx/                 包本体，共 3750 行
├── agent.py                Agent / load：加载、准备、推理、后处理（主干，296 行）
├── common.py               prompt 构造、选项渲染、温度校准、confidence（主干，138 行）
├── model.py                ModernBERT 编码器 + 决策头 + 权重名映射（主干，249 行）
├── tokenizer.py            HF Rust 分词器薄封装（27 行）
├── prepared.py             前缀缓存：重复问题只 tokenize 一次（59 行）
├── convert.py              导出一份本地 MLX checkpoint（48 行）
├── router.py               多 checkpoint 路由 + 生命周期管理（404 行）
├── lang.py                 依赖 Unicode 区块的语种/文字检测（463 行）
├── shortlist.py            大选项集的嵌入粗排（270 行）
├── email.py                邮件正文清洗、构造 email state（115 行）
├── presets.py              业务预设问题集：triage / moderation / guard 等（193 行）
├── cli.py                  predict / convert 两个子命令（55 行）
└── snake/                  贪吃蛇演示（762 行）
tests/                      1345 行，7 个文件
benchmarks/                 基准脚本 + 已提交的原始测量 JSON
experiments/                性能实验脚本与数据
docs/                       7 份研究/评测文档，共 1171 行
```

### 3.2 主干四件套

`agent.py` 是唯一入口，`laya.load()` 返回的 `Agent` 一次性完成：解析模型路径（本地目录 or HF repo）→ 校验配置完整性 → 读 `rl_agent_config.json` 与 `encoder/config.json` → 构造 `DecisionModel` → `strict=True` 加载权重（名字或形状对不上直接报错）→ 转 dtype → `mx.eval`。加载完成后 `Agent` 视为冻结：要换权重就重建一个。

`model.py` 里的 ModernBERT 是手写的，不是从 Transformers 继承。它保留了原实现的几个容易做错的点：global/local 交替注意力、滑窗边界是**闭区间**（`<= local_attention // 2`）、全局与局部用**不同**的 RoPE base、第一层不做 attention 前的 LayerNorm。加载时还会显式拒绝非 `modernbert` 的 model_type、非 GELU 激活、以及任何非 default 的 RoPE 缩放配置。

### 3.3 语言路由（`router.py` + `lang.py`）

三份 checkpoint 各有分工，路由就是替你选：

| 名称                     | 底座               |   参数 |  上下文 | 适用                        |
| ---------------------- | ---------------- | ---: | ---: | ------------------------- |
| `laya`                 | ModernBERT-large | 421M |  512 | 纯英文                       |
| `laya-multilingual`    | mmBERT-base      | 322M | 1024 | 多语言含中文，日常默认选这个            |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | 上游 typed-decisions 工作流微调版 |

路由的依据是**文字**（script），而不是语种本身。`lang.py` 用 Unicode 区块做精确判定（汉字、假名、谚文、阿拉伯、天城体……都各自有区间），拉丁字母再用停用词 + 变音符号比例做启发式猜。这么设计有实测依据：英文 checkpoint 在非拉丁文字上不是「略微变差」，而是**崩到接近随机**（20 选项意图任务上印地语 0.100、韩语 0.103，随机基线是 0.050），而且崩的时候还给出高置信度。所以与其猜语种，不如先判文字。

`typed-decisions` 默认不会被自动选中，必须显式传 `task="typed_decisions"` 或打开 `auto_task_detection`——它在四类合成工作流上微调过，不该当沉默默认值。多语言路径还处理了罗马尼亚语、波兰语这类「拉丁字母但看不出是不是英文」的情况，会因为出现普通英文没有的字母而落到多语言 checkpoint，而不是被想当然当成英文。

**但这条启发式有实测漏洞（本机跑 `Router` 得到）**：

| 输入 | 实际路由 | 结果 |
| --- | --- | --- |
| `Grüße aus Wien, ich brauche Hilfe mit meiner Rechnung.` | multilingual | 对（4% 非英语字母触发） |
| `Gracias por su ayuda con la factura, necesito un reembolso.` | multilingual | 对（`por`/`su`/`con` 命中停用词） |
| `Bonjour, je voudrais annuler mon abonnement.` | **english** | **错**：法语停用词不在表里，句子里又没有触发变音符判定的字符 |
| `您好，请问我的退款进度如何？` | multilingual | 对（汉字脚本） |

也就是说：**非拉丁文字必对，靠停用词的语言看表覆盖，只靠变音符的语言会漏**。法语、意大利语这类拼写里可以完全没有变音符的语言，会被当成英文送进英文 checkpoint——而英文 checkpoint 读不了它们。**结论：多语言场景里如果输入可能包含法语/意大利语，别信自动路由，显式传 `lang=` 或 `model=`。**

### 3.4 大选项集短名单（`shortlist.py`）

针对 2.4 节说的 token 预算问题。选项成百上千时每个选项分不到几个 token，先用你给的嵌入函数把 state 和每个标签各算一个向量，按余弦相似度取 top-k，再对这 k 个跑一次真正的 `predict`。这是**opt-in**：`Agent.predict` 仍然老老实实给它拿到的每个选项打分，不会偷偷给你截断。注意短名单上的概率是**只在保留下来的标签之间归一化的**，别拿它当全量分布用。

**实测警告（100 个中文标签 vs 一条明确正文）**：用 `embed_fn_from_agent`（拿决策模型自己的编码器取均值向量）做粗排，结果是**帮倒忙**——

| 路径 | 耗时 | 选中 | 置信度 |
| --- | ---: | --- | ---: |
| 全量 100 个标签 | 78 ms | 退款到账进度查询（正确） | 0.779 |
| 短名单 k=20 | 169 ms | 退款到账申请办理（错误） | 0.9989 |

粗排给出的余弦 top-5 是「银行卡解绑撤销申请 0.7622 / 套餐降级撤销申请 0.7507 / 账号冻结申请办理 0.7476 …」，全与正文无关；正确标签**根本没进** top-20。而且短名单还多花了一倍时间（要额外跑 1 条 query + 100 个标签的嵌入）。所以 README 那句「专门的 bi-encoder 通常粗排更好」不是客套话：**没有真正的双编码器就别用短名单**。另外注意短名单把概率重新归一化后报了 0.9989，比全量的 0.779 自信得多，但答案是错的。

### 3.5 Snake 演示（`snake/`）

不是玩具 demo。游戏每一步真实调用模型做决策，同时用一个哈密顿回路做安全兜底——模型提的方向要是会撞死就纠正。它把「模型有多不靠谱」变成了一个可以计数的指标（接管次数），比看一堆 accuracy 数字直观。仓库里连当时的真实运行记录都提交了（`snake-showcase.jsonl`），测试会逐帧回放校验。本机怎么跑见 4.9。

### 3.6 转换与发布

`convert.py` 做的是参数名和 dtype 的转换（`in_proj_weight` → `in_proj.weight` 这类），**不是量化也不是重训练**。输出目录必须不存在——不做覆盖，失败还会自动清理半成品。`scripts/prepare_hub.py` 负责产出待发布的模型卡与校验过的权重。

---

## 4. 快速使用

### 4.1 环境要求

Apple Silicon Mac（M 系列）、macOS 14+、Python 3.11+。本机实测环境：arm64、macOS 15.6、Python 3.13.12、MLX 0.32.3。

### 4.2 环境（本机已实测通过）

本机统一用 conda 环境 **`laya`**，跑任何脚本都用它的解释器：

```bash
/opt/anaconda3/envs/laya/bin/python <脚本>
```

里面已经装好 mlx 0.32.3 / tokenizers / numpy / huggingface-hub / rich / Pillow，并且用可编辑方式装了仓库本体（`pip install -e .`），所以 `laya-snake`、`laya-mlx` 这些命令可以直接调。**注意这个环境实际是 Python 3.14.8**（`python --version` 在有些 shell 里会被 PATH 上的别的解释器覆盖，别被误导）。

从零重建：

```bash
conda create -n laya python=3.13 -y
cd laya-mlx
/opt/anaconda3/envs/laya/bin/pip install --no-compile -e ".[demo]"
```

（仓库根曾经建过一个 `.venv`，已经删掉了；`.venv/` 本来就在 `.gitignore` 里。）

（也可以不装本地包，直接 `pip install laya-mlx`——PyPI 上最新是 0.3.0，本地仓库版本号是 0.2.0。）

**唯一的坑：huggingface.co 在你这台机器上直连不通。** 实测 `curl https://huggingface.co/api/models/aac6fef/laya-mlx` 返回 `000`，走代理是 `502 Bad Gateway`，`laya.load(...)` 会直接抛 `httpx.ProxyError`。必须把下载端点指到镜像，否则后面的代码一行都跑不动：

```bash
export HF_ENDPOINT=https://hf-mirror.com    # 实测可达（200）
```

（建议写进 shell profile，或每次跑之前 export。）这样下载的权重落在 `~/.cache/huggingface/hub/`，多语言版 663 MB，**只在第一次下载，之后完全离线可跑**。

### 4.2.1 把权重单独放到项目的 models/ 目录

不想让权重散在 HF 缓存里、想跟着项目走，就用 `hf download --local-dir`。`models/` 已经在 `.gitignore` 里，不会误提交。

```bash
cd laya-mlx
export HF_ENDPOINT=https://hf-mirror.com        # 镜像，本机直连不通，见上
hf download aac6fef/laya-multilingual-mlx --local-dir models/laya-multilingual-mlx
hf download aac6fef/laya-mlx              --local-dir models/laya-mlx
```

下载完是这样（**已在本机下好，直接可用，不用重复下**）：

| 目录 | 大小 | 用途 |
| --- | ---: | --- |
| `models/laya-multilingual-mlx` | 657 MB | 中文及一切非拉丁文字，日常默认 |
| `models/laya-mlx` | 810 MB | 纯英文 |

之后加载时传目录路径即可，`laya.load` 判定「这个路径存在」就直接当本地权重用，不发任何网络请求：

```python
import laya_mlx as laya
agent = laya.load("models/laya-multilingual-mlx", dtype="float16")
```

一个容易踩的细节：**相对路径依赖当前工作目录**。脚本里最好像示例那样用 `Path(__file__).resolve().parents[2] / "models" / ...` 拼绝对路径，避免换个目录执行就报 `FileNotFoundError`。

第三种拿权重的方式是 `laya-mlx convert --model convaiinnovations/laya --output models/xxx`，那是从官方仓库拉原始权重再转成 MLX 格式（需要联网），本仓库的 `aac6fef/*-mlx` 已经是转好的，没特殊需求不用走这条路。

然后跑一个最小脚本：

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-multilingual-mlx", dtype="float16")   # 中文业务必须用这个，别用英文版

result = agent.predict(
    "发票被重复扣款了，我已经付了两次，请尽快退款。",
    {
        "department": {
            "type": "choice",
            "instructions": "这个问题该由哪个团队处理？",
            "criteria": {"billing": "发票、付款、退款", "technical": "系统故障与报错", "sales": "新购买"},
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
    },
)
print(result["answers"])
```

现成的可执行脚本放在 `AIDocuments/quickstart-zh.py`，`cd laya-mlx && /opt/anaconda3/envs/laya/bin/python AIDocuments/quickstart-zh.py` 即可。

本机实测输出（原样摘录，不是 README 抄的）：

```json
{
  "department": { "choice": "billing", "confidence": 0.7199,
                  "probabilities": { "billing": 0.9138, "technical": 0.0828, "sales": 0.0034 } },
  "urgency":    { "score": 1.3298, "probabilities": { "0": 0.0771, "1": 0.516, "2": 0.4069 } },
  "refund":     { "noul": 0.9929, "confidence": 0.9929 }
}
```

同一台机器上的耗时（多语言 FP16，一条中文 state + 三个问题一次调用）：

| 阶段 | 实测 |
|---|---:|
| 权重加载 | 1.49 s |
| 首次预测（含预热） | 110.1 ms |
| 稳态单次（3 问合计） | 20.1 ms |

20.1 ms 是三个问题一起算完的时间（同一批内并行），折合单问题约 7 ms，与 README 标的 7.39 ms 量级吻合。多次运行之间有 ±5 ms 抖动。**首次调用一定比稳态慢一个数量级**，这是 MLX 的惰性编译，不是模型慢。

### 4.3 Python API

```bash
pip install laya-mlx
```

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx")          # 首次会下载权重
result = agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": ["billing", "technical", "sales"],
        }
    },
)
print(result["answers"]["department"])
```

常用参数：`dtype="float16"`（默认，要数值一致性改用 `"float32"`）、`batch_size=16`（单批问题数上限，大了快但要内存）、`cache_prompts=True`（重复问题只 tokenize 一次）、`compile=True`、`pad_to_multiple=16`。后三个默认关闭，都有适用前提，别无脑全开。

三个 checkpoint 的选择：做中文业务用 `aac6fef/laya-multilingual-mlx`；拿不准就用 `Router`，它会按输入文字自动跳。

### 4.4 命令行

```bash
laya-mlx predict \
  --model aac6fef/laya-multilingual-mlx \
  --state-file examples/state.json \
  --questions examples/questions.json

# 或直接用文本
laya-mlx predict --model aac6fef/laya-mlx --state '发票被重复扣款，请退款。' --questions examples/questions.json
```

导一份本地权重：

```bash
laya-mlx convert --model convaiinnovations/laya --dtype float16 --output models/laya-mlx-fp16
```

### 4.5 Router 用法

```python
from pathlib import Path
from laya_mlx import Router, triage_questions

router = Router(
    models={                                    # 指向本地目录，不联网
        "english": Path("models/laya-mlx"),
        "multilingual": Path("models/laya-multilingual-mlx"),
    },
    max_loaded=2,              # 两个都常驻，切语言不重新加载
    default="multilingual",    # 输入里没有字母时（纯数字等）走这个
)
result = router.predict({"message": "发票被重复扣款，请退款。"}, triage_questions())
print(result["routing"]["model"])     # multilingual
print(result["routing"]["reason"])    # 判定依据，中文是 "non-Latin script (han, 100% of letters)..."

# 只想知道会选哪个、不想加载模型：
router.route({"message": "..."}, triage_questions())
```

`routing` 是个字典，含 `model` / `repo` / `reason` / `detection` / `workflow` 五个键；直接 print 整个字典会很长，看 `model` 和 `reason` 就够。`max_loaded` 限制常驻内存的 checkpoint 数量；模型生命周期用可重入锁保护，多线程共享同一个 `Agent` 而不会重复加载，但推理本身不做串行化。

### 4.6 shortlist 用法

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx")
embed_fn = laya.embed_fn_from_agent(agent)     # 复用已加载的编码器，不额外下权重
result = laya.predict_shortlist(agent, state, questions, embed_fn, k=20)
print(result["shortlist"])     # 保留了哪些标签，各自的余弦分数
```

专门的双编码器当 `embed_fn` 通常比用决策模型自己的编码器粗排效果更好。

### 4.7 跑测试（本机已验证）

> ⚠️ 下面这段是当时的记录，临时环境 `/tmp/laya-venv` **已经删掉**。现在要跑测试就直接用 conda `laya`，
> 但它只装了运行依赖、没装 pytest，得先补一条 `/opt/anaconda3/envs/laya/bin/pip install --no-compile pytest`。

仓库里没有 `.venv`，我用隔离环境装依赖跑过一次完整测试：

```bash
python3 -m venv /tmp/laya-venv
/tmp/laya-venv/bin/pip install mlx tokenizers pytest numpy huggingface-hub Pillow rich
cd laya-mlx
LAYA_MLX_TEST_DEVICE=cpu TOKENIZERS_PARALLELISM=false /tmp/laya-venv/bin/python -m pytest -q
```

结果 **141 passed，1 skipped**。跳过的是 `tests/test_model.py:17`——它要和 PyTorch Transformers 做数值对比，属于 `reference` 可选依赖。单元测试用的是临时构造的随机小模型（64 维隐层、3 层），不需要下载真实权重，所以 CPU 上也能跑完。

### 4.8 你原来那批 jev 例子，跑在本地 laya 上（`AIDocuments/jev-on-laya/`）

`~/Desktop/myHub/RAG-Engineeing/jev/` 下的四个脚本抄了过来，**只把调用方式换成 laya**——
题目定义、测试数据、打印格式一个字没动。改动就这三处：

| 原来（Jev 云端） | 现在（本地 laya） |
| --- | --- |
| `client = TypeSafeClient(api_key=...)` | `agent = laya.load(MODEL, dtype="float16")` |
| `client.system_one(model="jev-latest", state=msg, questions={"qid": Choice(instructions=..., criteria=...)})` | `agent.system_one(msg, {"qid": {"type": "choice", "instructions": ..., "criteria": ...}})` |
| `resp.answers["qid"].choice` | `resp["answers"]["qid"]["choice"]` |

`load_dotenv()` 和 `TYPESAFE_API_KEY` 那几行删掉了——本地不需要 key。
`Choice(...)` 这个包装换成直接写 dict（laya 要 `{"type","instructions","criteria"}`）。

```bash
cd AIDocuments/jev-on-laya
/opt/anaconda3/envs/laya/bin/python classifer.py     # 另外三个同理
```

实测（多语言 checkpoint，FP16）：

| 例子 | 内容 | 结果 |
| --- | --- | --- |
| `classifer.py` | 两阶段：先判域，再域内匹配 qid | 域 `billing` 0.9875、qid `duplicate_charge` 0.9339，**全对** |
| `similar_qids.py` | 6 个高度相近的取消类意图 × 9 条消息 | **命中 2/9** |
| `conversation_classifier.py` | 同一批意图，喂一段多轮对话 | `driver_no_show` 0.5996，说得通 |
| `test_ecommerce_intents.py` | 5 个售后意图 × 4 条消息 | 明确对 2 条，明确错 2 条 |

两个结论值得单独记：

- **标签语言有影响**：取消类那批，只把标签从英文换成中文（`cancel_by_passenger` → `乘客自己取消`），
  描述一个字不动，命中从 **2/9 升到 4/9**。混搭「英文标签 + 中文描述」在拖后腿。
- **不是 token 截断**：多语言版 `head_max_len` 实际是 **256**（不是英文版的 192），
  6 个选项平均 30.5 token、合计 183，没触发压缩，每个选项都完整。掉分掉在语义区分本身。

另外 `AIDocuments/quickstart-zh.py` 是不依赖 Jev 的最小示例（一个 state、三个问题、带计时）。

### 4.9 跑 Snake 演示（`AIDocuments/run-snake.sh`）

Snake 是命令行工具（入口 `laya_mlx/snake/cli.py`），不是单个脚本，**必须带 `--model`** 指到本地权重，否则会去找 Hub 默认缓存。封装了一个脚本，省掉记这些：

```bash
cd laya-mlx
./AIDocuments/run-snake.sh                                   # 交互式游玩（需要真终端）
./AIDocuments/run-snake.sh --max-speed                       # 交互式，每步都重新决策
./AIDocuments/run-snake.sh --headless --steps 300            # 无界面跑一段，结束打印统计
./AIDocuments/run-snake.sh --record run.jsonl                # 顺带录制每一步
./AIDocuments/run-snake.sh export run.jsonl --output run.png # 把录制渲染成图 / mp4
./AIDocuments/run-snake.sh benchmark --seeds 3               # 基准测试
```

它固定用 `/opt/anaconda3/envs/laya/bin/laya-snake`、权重固定指向 `models/laya-multilingual-mlx`；你显式传了 `--model` 就以你的为准。脚本会自己算项目根，在哪个目录调用都行。

交互界面要在真 TTY 上画：PyCharm 的 Run 窗口不是 TTY，要么用它底部的 Terminal，要么加 `--headless`。非 TTY 下跑又忘了加 `--headless`，脚本会先提示一句。

本机实测（多语言 FP16）：`--headless --steps 100` → 100 步 / 8.7 秒 / 11.48 步每秒 / 得分 1 / 零死亡 / 平均推理 30.5 ms / `network: offline`。录制 60 步产出 62 行 JSONL，`export --output x.png` 正常出图。

---

## 5. 与 Jev 的联系和区别

你 `~/Desktop/myHub/RAG-Engineeing/jev/` 下那几个脚本用的是 **TypeSafe AI 的 Jev**，走 `typesafe_sdk.TypeSafeClient` + `TYPESAFE_API_KEY`，调 `model="jev-latest"`。它是一个**闭源云端 API**。

### 5.1 三者关系

**图 5-1　Jev / Laya / laya-mlx 三者关系**

> 三者说的是同一套调用契约。差别在「谁替你算」和「要不要联网」。

<svg viewBox="0 0 680 330" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,'PingFang SC','Helvetica Neue',sans-serif" height="330">
<title>Jev / Laya / laya-mlx 三者关系</title>
<desc>Jev 是 TypeSafe 的闭源云端 API，Laya 是 Convai Innovations 的开源权重，laya-mlx 是把 Laya 权重搬到 Apple MLX 上做本地推理。</desc>
<defs>
<marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<rect x="40" y="44" width="600" height="48" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="340" y="62" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#854F0B">同一套调用契约</text>
<text x="340" y="80" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#BA7517">state + 结构化问题 → choice / score / noul</text>
<path d="M130 92 V110" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar3)"/>
<path d="M340 92 V110" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar3)"/>
<path d="M550 92 V110" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar3)"/>
<rect x="40" y="114" width="180" height="104" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="130" y="138" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#0C447C">TypeSafe Jev</text>
<text x="130" y="160" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#185FA5">TypeSafe AI</text>
<text x="130" y="180" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#185FA5">闭源，仅云端 API</text>
<text x="130" y="200" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#185FA5">64k 上下文 · 255 选项</text>
<rect x="250" y="114" width="180" height="104" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="340" y="138" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#085041">Convai Laya</text>
<text x="340" y="160" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0F6E56">Convai Innovations</text>
<text x="340" y="180" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0F6E56">开源权重 Apache-2.0</text>
<text x="340" y="200" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0F6E56">512–1024 上下文</text>
<rect x="460" y="114" width="180" height="104" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text x="550" y="138" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#3C3489">laya-mlx</text>
<text x="550" y="160" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#534AB7">Apple MLX 本地推理</text>
<text x="550" y="180" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#534AB7">加载上一步的权重</text>
<text x="550" y="200" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#534AB7">无 key，数据不出本机</text>
<path d="M130 218 V258" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar3)"/>
<path d="M340 218 V258" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar3)"/>
<path d="M550 218 V258" fill="none" stroke="#888780" stroke-width="1.5" marker-end="url(#ar3)"/>
<rect x="40" y="262" width="600" height="48" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="280" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#444441">你的业务代码</text>
<text x="340" y="298" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#5F5E5A">questions 字典两边通用，换的是「谁在算」</text>
</svg>

**联系**：Jev 和 Laya 是两家公司先后做出了同一件东西——2026 年 9 月 TypeSafe 先发布闭源的 Jev，几天后 Convai Innovations 开源了 Laya。两者用同一套三种原语（`choice` / `score` / `noul`），连方法名都一样（Jev SDK 的 `client.system_one()` 对应 Laya/laya-mlx 的 `agent.system_one()`），输入输出结构几乎可以对照。你那套 qid 分级 + 置信度阈值判断的写法，换到 laya-mlx 上是改调用端，不是改题目设计。
实测下来要改的只有调用方式，题目设计不用动，见 5.3；
唯一真的要改的是「置信度阈值」那一句的含义，见 6.4。

**根本区别**：Jev 是你把数据发给别人算并按 token 付钱；Laya 是把权重给你自己算；laya-mlx 是 Laya 在 Mac 上的本地实现。

### 5.2 逐项对比

| 维度     | TypeSafe Jev               | laya-mlx（跑 Laya 权重）            |
| ------ | -------------------------- | ------------------------------ |
| 交付形态   | 闭源托管 API，需 API key         | 本地 pip 包，权重自己下载                |
| 运行环境   | 有网就行                       | 必须 Apple Silicon Mac           |
| 数据流向   | state 出网到 TypeSafe         | 不出本机                           |
| 计费     | 输入约 $0.042 / 百万 token，输出免费 | 免费，付的是电费和你自己的机器                |
| 上下文长度  | 文档口径数十 k token             | 512（英文）/ 1024（多语言）token        |
| 单问题选项数 | 上限 255                     | 受 `head_max_len` 预算约束，选项多了要短名单 |
| 典型延迟   | 几十到几百毫秒（含网络）               | 本机 P50 约 18 ms（实测，见第 6 章）      |
| 数值可复现  | 取决于服务端版本                   | 完全可控，可以 pin revision           |
| `confidence` 语义 | 校准概率，可当「判断对的概率」用 | **归一化熵**，只表示分布集中度，见 6.4 |
| 微调校准   | 不支持（闭源）                    | 权重开源，可以自己 fine-tune 并重拟合温度     |


### 5.3 你那段 jev 脚本迁过来要改什么

实测过了（4.8）：**只改调用方式，题目设计不用动。** 对照表：

| Jev 云端 | 本地 laya |
| --- | --- |
| `client = TypeSafeClient(api_key=...)` | `agent = laya.load(模型路径, dtype="float16")` |
| `client.system_one(model="jev-latest", state=msg, questions={"x": Choice(instructions=..., criteria=...)})` | `agent.system_one(msg, {"x": {"type": "choice", "instructions": ..., "criteria": ...}})` |
| `resp.answers["x"].choice` | `resp["answers"]["x"]["choice"]` |
| `answer.confidence` | `answer["confidence"]`（**含义不同**，见 6.4） |
| `load_dotenv()` + `TYPESAFE_API_KEY` | 删掉 |

`questions` 的 key（也就是你写的 qid 名）和 `criteria` 的内容可以原样搬过来。

三点要注意：一是中文业务必须走 multilingual checkpoint；二是 `confidence` 的**含义不同**（见 6.4），
Jev 那套「< 0.7 转人工」的阈值不能照搬到本地；三是 Jev 的延迟是网络延迟、本地是计算延迟，
两者抖动来源完全不同。

### 5.4 怎么选

- **数据出不了本机 / 合规要求明确**：只能 laya-mlx（或者任一本地部署的 Laya）。
- **你已经有 Mac、量不小、意图体系稳定**：laya-mlx 的边际成本是零，Jev 的 token 账单会一直线性增长。
- **要零样本效果最强的那个**：开源阵营自己也承认，base checkpoint 在某些 typed-decisions 任务上接近随机，需要先 specialize；云端 Jev 开箱效果通常更好。第三方评测也提到 Jev 零样本准确率明显更高。
- **团队没有 ML 运维能力、或者机器上不仅是 Mac**：Jev 的「替你把模型跑起来」是有实际价值的，别为了开源而开源。

更务实的做法是分层：两者 `questions` 定义通用，可以先在 Mac 上用 laya-mlx 快速验证题目设计是否合理（改一行描述立刻生效），效果确认后再决定终局走哪条路。

---

## 6. 已知边界与注意事项

### 6.1 README 首页的延迟数字比当前基准结果旧

README 表格说 Laya 421M 单问题端到端 P50 是 **13.42 ms**、多语言版 **7.39 ms**。但我从提交在仓库里的原始样本 `benchmarks/results/laya-mlx-float16.json`（生成于 2026-09-22）重算，同配置的 P50 是：

| checkpoint        | 口径         | 当前 JSON 实测 P50 | README 首页口径 |
| ----------------- | ---------- | -------------: | ----------: |
| laya FP16         | forward    |       15.82 ms |           — |
| laya FP16         | end-to-end |   **17.75 ms** |    13.42 ms |
| multilingual FP16 | forward    |        8.72 ms |           — |
| multilingual FP16 | end-to-end |   **10.91 ms** |     7.39 ms |

`benchmarks/latency` 表和 `BENCHMARKS.md` 的主表给的也是 17.75 / 10.91，与 JSON 一致。13.42 / 7.39 这两个数出自 `docs/PERFORMANCE_RESEARCH.md`（2026-09-19 的早期一轮测量）。**结论：README 首页那两个数是旧一轮的结果，引用性能时应以 `BENCHMARKS.md` 主表或 `benchmarks/results/*.json` 为准。**

### 6.2 使用上的硬约束

- **问题数量线性影响延迟**：每个问题各自走一遍编码器，state 不会被编码一次复用。本机实测（多语言 FP16，端到端口径）：1 问 13.1 ms、5 问 28.3 ms、20 问 104.5 ms、40 问 203.5 ms——摊下来是**固定开销约 8 ms + 每问约 5 ms**，所以问题越多摊得越薄，但总量一定随问题数线性涨。官方 50 题吞吐（laya FP16 约 143 q/s、multilingual 约 402 q/s）是 `batch_size=64` 测的，而 API 默认 16，默认配置达不到。
- **`head_max_len` 是共享预算，而且是静默降级的**：见 2.4 节和 3.4 节。本机实测：多语言版给 100 个选项时，`agent.prepare()` 出来的每条选项**只占 4 个 token**（含 1 个 MASK，真正留给标签文字的只剩 3 个），整条序列 428 token——没有报错、没有警告。机制是 `build_prefix` 先给每个选项最多 48 token，预算不够时再统一压到 `max(4, (head_max_len-16)//选项数)`。所以「选项太多」的后果是悄悄变差，不是抛异常，得自己盯着选项数和标签长度。
- **`Agent` 加载后即冻结**：开了 `compile=True` 还会按输入形状做特化，padding 策略改了可能触发重新编译；`pad_to_multiple` 在某些负载上反而更慢。三者默认关闭是对的。
- **Python 侧没有服务化封装**：它是一个库，不是 API 服务。要做成 HTTP 服务得你自己包一层（社区里有人用 Jev 兼容协议包了本地版，但那是第三方工作，不在本仓库范围）。
- **一致性验证的边界**：仓库宣称三个 checkpoint 在 FP32/FP16 下与上游在 63/63 个验证问题上 argmax 一致（共 378 次比对）——这是在**那批 fixtures 上的保真度**，不等于在任意问题上的准确率。AG News 256 条采样上的准确率见 `BENCHMARKS.md`，最高 0.9648。
- **上游同步靠人工**：最新提交 `0a85951` 是一次手工同步到上游 v0.3.5。没有自动 diff 机制，上游再改动需要人工再跟一轮。
- **拉权重必须走镜像**：见 4.2。本机直连 huggingface.co 不通，不设 `HF_ENDPOINT=https://hf-mirror.com` 会直接抛 `httpx.ProxyError`。
- **解释器就用 conda `laya`**：`/opt/anaconda3/envs/laya/bin/python`。用别的解释器直接跑仓库里的脚本会 `ModuleNotFoundError: No module named 'laya_mlx'`——Python 加进 `sys.path` 的是脚本所在目录，不是当前目录；靠的是 `pip install -e .` 写进 site-packages 的那条路径。另外 `similar_qids.py` / `conversation_classifier.py` / `test_ecommerce_intents.py` 三个脚本里的函数**不能**用 `test_` 前缀，否则 PyCharm 会当它们是 pytest 用例、硬走 pytest 运行器（而这些函数要 `agent` / `messages` 参数，pytest 根本收集不了）。
- **PyCharm 运行方式**：右键脚本选 **Run '<脚本名>'**（Python 图标那个）。如果菜单里同时出现 `pytest in xxx.py`，别选它。
- **snake 演示怎么跑**：它是命令行工具，不是单个脚本，必须带 `--model` 指到本地权重（否则会去找 Hub 默认缓存）：
  ```bash
  cd laya-mlx
  /opt/anaconda3/envs/laya/bin/laya-snake --headless --steps 80 --model models/laya-multilingual-mlx
  ```
  交互式界面（方向键 / 空格 / R / Q）需要真 TTY，在普通终端里跑；PyCharm 的 Run 窗口不是 TTY，只能加 `--headless`。
- **加载英文 checkpoint 会打印一条 RuntimeWarning**：内容是把 `choice:11+=0.1006` 截断到区间内。这是设计如此（见 2.5），不是错误，用英文版时看到它属正常。多语言版没有这条，因为它的 `temperature` 全是 1.0、没有 `temperature_by_options`。

### 6.3 中文正文在 `noul` 上判定偏保守（本机实测）

同一个问题、同一个 checkpoint（多语言版），只换正文语言，用 `triage_questions()` 里现成的 `is_urgent` 问题：

| 指令语言 | 正文语言 | P(表达了时间压力) |
| --- | --- | ---: |
| 中文 | 中文 | 0.0965 |
| 英文 | 中文 | 0.1007 |
| 中文 | 英文 | 0.9205 |
| 英文 | 英文 | 0.7449 |

正文「你们承诺 48 小时到账，现在都一周了，再不处理我就投诉到监管部门」是明显带时间压力的，但中文正文只给到 0.10 左右，而对应的英文正文能到 0.74–0.92。**差别来自正文语言，不是指令语言**（指令换语言几乎不改变结果，0.0965 vs 0.1007）。

注意限定：这是单组对拍，不构成准确率结论；而且同一批测试里 `choice` 类问题在中文上表现正常（「重复扣款」→ billing，置信度 0.9997）。所以更准确的说法是：**中文的粗粒度分类可用，但 `noul` 这种需要细粒度立场判断的问题，中文上明显比英文保守，阈值不能照搬英文的设置**。上生产前建议用你自己的真实语料分别标定中英文的阈值。

### 6.4 `confidence` 和 Jev 的 `confidence` 不是一个东西

把 jev 例子搬过来实测才暴露的（见 4.8）：**字段同名，语义不同。**

Jev 那边 `answer.confidence` 是**校准概率**（RLCD 训练出来的，ECE 约 0.07），可以当「这个判断有多大概率是对的」用。
laya 这边是同名代码 `confidence_from_probs()` 算的**归一化熵** `1 - H(p) / log(k)`，量的是「这堆概率有多集中」，
跟「对不对」无关。

实测验证（多语言 FP16，4.8 里电商售后那批的第 2 条）：`confidence` 报 **0.306**，而同一条的
`return_refund` 真实概率是 **0.5438**。手算 `1 - H/log(5)` 得到 0.3068，确认是熵，不是概率。

后果是 `jev/classifer.py` 结尾那句：

```python
if result["qid_confidence"] < 0.7:
    print("⚠️ qid 置信度较低，建议转人工或进入下一轮澄清。")
```

**在本地不能直接用**——它会把上面那条 0.5438 的合理判断直接判成「低置信度」。
要在本地做阈值路由，用 `answer.probabilities[answer.choice]`，那个才对应 Jev 的 confidence。

---

## 参考资料

- 仓库内：`README.md`、`BENCHMARKS.md`、`docs/PERFORMANCE_RESEARCH.md`、`docs/SNAKE_OPTIMIZATION.md`、`benchmarks/results/*.json`
- 第三方对比 Jev 与 Laya 的文章：`runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models`、`alphamatch.ai/blog/jev-vs-laya-system-one-2026`
