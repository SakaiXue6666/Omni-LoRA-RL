# Omni-LoRA-RL

Qwen3-Omni Thinker + LoRA 的强化学习训练工程。任务是 **S2TT**（英语语音 → 中文文本），
算法用 GRPO，奖励用句级 BLEU，推理侧走 sglang 的 LoRA adapter 热加载 —— 训练每步把更新
后的 adapter 直接从显存推给推理引擎，不落盘。

**最好的结果：4×A100-80GB 上跑满 100 步，BLEU 从 0.287 升到 0.487。**
当前代码路径已复现前 40 步（0.294 → 0.391）。四次实验的完整逐步数据在
[`docs/results/experiments.md`](docs/results/experiments.md)。

本仓库是**入口 / hub**，真正的代码以 submodule 指向两个 fork。

## 仓库组成

| 目录 | 来源 | 分支 | 作用 |
|---|---|---|---|
| `Relax/` | fork 自 `redai-infra/Relax` | `lora-omni-v2` | 训练侧（Megatron-Bridge + LoRA + rollout） |
| `sglang/` | fork 自 `sgl-project/sglang` `v0.5.12.post1` | `lora-omni-v2` | 推理侧（Omni 的 LoRA serving） |
| `omni_s2tt/` | 本仓库自带 | — | 训练脚本、BLEU 奖励、数据准备、曲线统计 |
| `scripts/`、`docs/` | 本仓库自带 | — | Modal 入口与迁移期探针；文档与实验数据 |

之所以还要 fork，是因为有五处改动还没进上游（PR 都已提，见下面「当前进度」）。
一旦合并，fork 就能退化成"上游 + Omni 专属那两处"。

**硬件**：4 张 80GB 卡（A100/H100）。模型是 30B 的 MoE（激活 3B），配置是 colocate ——
训练与推理共用同一批卡，TP4 / EP4 / PP1。单机即可，不需要多机。显存吃紧先调
`SGLANG_MEM_FRACTION`（默认 0.7），它决定推理引擎占多少。

---

# 做了什么实验，结果如何

数据集统一是 **FLEURS**（`google/fleurs`，`en_us` 音频 + `cmn_hans_cn` 文本按 `id` 平行对齐），
奖励是 sacreBLEU（中文 tokenizer）/ 100。指标是每步 `rollout/raw_reward`，也就是那一批样本
BLEU 的均值。**判据一律用十步窗口均值，不看单步** —— 单步噪声很大，见过从 0.539 掉到 0.397。

| 实验 | 数据 | 步数 | 结果（首窗 → 末窗） |
|---|---|---|---|
| zh→en 翻译 + BLEU（验证奖励设计） | 自造长难句 256 条 | 10 | 0.463 → 0.523 |
| **en→zh S2TT（离线单轮）** | FLEURS 128 条 | **100** | **0.287 → 0.487** |
| 同传（960ms 定长块，多轮） | FLEURS 97 条整段 | 20 | 0.155 → 0.265 |
| en→zh S2TT（当前代码路径） | FLEURS 128 条 | 40 | 0.294 → 0.391 |

还有一次有价值的失败：数学 MCQ + 0/1 奖励，32/32 全对、组内零方差、advantage=0，学不动。
它决定了后面所有实验为什么用连续奖励，理由见
[`docs/design/reward-design.md`](docs/design/reward-design.md)。

## 主线：S2TT 100 步

十步窗口均值，单调上升：

| 区间 | BLEU | | 区间 | BLEU |
|---|---|---|---|---|
| 1–10 | 0.2868 | | 51–60 | 0.4137 |
| 11–20 | 0.3442 | | 61–70 | 0.4400 |
| 21–30 | 0.3448 | | 71–80 | 0.4822 |
| 31–40 | 0.3886 | | 81–90 | 0.4890 |
| 41–50 | 0.4020 | | 91–100 | 0.4873 |

区间 [0.234 @step3, 0.610 @step95]。旁证：response 长度从 28 token 降到 20（译文变紧凑）、
log_probs 上升。逐步原始数据 `docs/results/s2tt-100step-curve.json`。

## 边界：哪些没验过

- **当前代码路径只跑到 40 步。** 那条 100 步曲线是在旧实现上跑出来的；40 步之后能不能
  继续爬到 0.487，在当前代码上没验过。前 40 步两者逐段吻合（差值都在 ±0.013 内）。
- **续训没验证过。** 最近一次想从 `iter_0000004` 接着跑，实际从 0 开始了 —— 上次被提前
  收掉，`latest_checkpointed_iteration.txt` 没写出来。LoRA 的续训路径至今没单独查过。
- **同传代码已移过来，但一次没跑过。** `omni_s2tt/simul/`，四处必要改动已做；已知风险与
  预期报错见 [`docs/design/simul-port-notes.md`](docs/design/simul-port-notes.md)，上卡前先读。
- **口径**：`omni_s2tt/curve.py` 按 `rollout_result` 的逐样本 reward 统计，上面的数字来自
  训练日志的 `rollout/raw_reward`。两者理论上相等，但没交叉核对过。

---

# 当前进度

> 截至 2026-09-05

**已闭环**：S2TT 训练链路在当前代码上端到端跑通（2026-08-12，40 步）。sglang 的
`should_apply_lora` 门控、Omni 的 `_lora_pattern`、Relax 的通配符展开与 PEFT 前缀、
以及 adapter 传输，全部被这次训练隐式验证过。

**进行中**：五个上游 PR，全部 open，自 8 月中旬起无进展。

| PR | 仓库 | 状态 |
|---|---|---|
| [#34428](https://github.com/sgl-project/sglang/pull/34428) Honor `should_apply_lora` | sglang | Open，**CI 卡在缺 `run-ci` label，测试从未真正执行**；5 位 code owner 未响应 |
| [#34595](https://github.com/sgl-project/sglang/pull/34595) CPU 张量 reduce 越界守卫 | sglang | Open，同样卡 `run-ci` label |
| [#261](https://github.com/redai-infra/Relax/pull/261) 通配符 target modules 展开 | Relax | Open，CI 全绿，等 code owner 批准 |
| [#262](https://github.com/redai-infra/Relax/pull/262) adapter 按 PEFT key 布局导出 | Relax | Open，CI 全绿，等 code owner 批准 |
| [#265](https://github.com/redai-infra/Relax/pull/265) adapter 传输改内联字节 | Relax | Open，等 code owner 批准 |

PR 正文存档在 `docs/upstream-prs/`。两个 sglang PR 卡的不是技术问题，是需要 maintainer
加个标签 —— 这是最容易推动的一件事。

**下一步**（按优先级）：

1. 推动 sglang 两个 PR 的 `run-ci` label
2. 排查 LoRA 续训不生效
3. 跑通同传（代码已就位，见 `docs/design/simul-port-notes.md` 的风险清单）
4. 当前代码路径跑满 100 步，确认能复现 0.487

---

# 如何运行

两条路径，**验证状态不同，先看清楚**：

| | 状态 |
|---|---|
| **A. Modal** | ✅ **已验证**。上面所有实验都是这条路跑出来的 |
| **B. 自建服务器 / Docker** | ⚠️ **从源码推导，一次没实跑过**。逻辑与 A 一一对应，但没人走通过 |

B 那条路欢迎第一个跑通的人回来改这份文档 —— 有出入的地方直接提 PR，那比留着一份没验证的
手册有用得多。

## A. 在 Modal 上跑（已验证）

```bash
modal run scripts/modal_train_s2tt.py::check                      # 先查数据与权重（CPU，几十秒）
modal run scripts/modal_train_s2tt.py --num-rollout 40 --detach   # 起训练，spawn 出去与本地解耦
modal run scripts/modal_train_s2tt.py::result --call-id <ID>      # 取结果
```

**`--detach` 不要省。** 用 `remote()` 会把训练的生命周期绑在本地那个 modal 进程上，本地一断
app 就被收掉 —— 上次就是这么在第 5 步被杀的，只留下一个 `iter_0000004`。

`scripts/modal_train_s2tt.py` 里的镜像、PYTHONPATH、环境变量与下面 B 的每一节一一对应，
可以对着看。

## B. 在自建服务器 / Docker 上跑（未实跑）

### 一、拉代码

```bash
git clone --recursive https://github.com/SakaiXue6666/Omni-LoRA-RL.git
cd Omni-LoRA-RL

# 已经 clone 但忘了 --recursive：
git submodule update --init --recursive
```

核对两个 submodule 落在预期的提交上（对不上就是没 init 干净）：

```bash
git submodule status
# +9e94202... Relax  (heads/lora-omni-v2)
#  02044692... sglang (v0.5.12.post1-5-g02044692cc)
```

### 二、起容器

用 Relax 官方镜像，按 digest 钉死——`:latest` 会漂移，旧的那套实现就是因为这个变得不可复现。
镜像里已经有 Megatron-LM、Megatron-Bridge、flashinfer、transformer-engine，不用自己装。

```bash
docker run --gpus all -it --rm \
  --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v $PWD:/workspace/Omni-LoRA-RL \
  -v /path/to/qwen3-omni:/models/qwen3-omni \
  -v /path/to/s2tt-data:/data/s2tt \
  ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23 \
  bash
```

`--shm-size` 不要省。默认的 64MB 会让 NCCL 与 Ray 的共享内存路径随机失败。

容器里补一个包（BLEU 奖励要用它的中文 tokenizer）：

```bash
pip install --no-cache-dir sacrebleu
```

#### PYTHONPATH

这是最容易翻车的一步，三条都必须有：

```bash
cd /workspace/Omni-LoRA-RL
export PYTHONPATH=$PWD:$PWD/sglang/python:$PYTHONPATH
```

- `$PWD`：让 `--custom-rm-path omni_s2tt.bleu_rm.compute_bleu_reward` 能 import 到奖励模块
- `$PWD/sglang/python`：**顶掉镜像预装的 sglang**，否则跑的是没有 Omni LoRA 支持的那份
- 结尾的 `$PYTHONPATH`：镜像原本的 `/root/Megatron-LM:/pkg:/root` 必须保留，`megatron` 和
  `megatron.bridge` 都在那儿。第一次跑漏了它，Ray job 起来后在 `from megatron.core import mpu`
  直接 ModuleNotFoundError，十次重试全废在同一个地方

烧卡之前先自检一遍，几秒钟的事：

```bash
python3 -c "import megatron.core, relax, sglang, omni_s2tt.bleu_rm as b; \
print('megatron', megatron.core.__file__); print('relax', relax.__file__); \
print('sglang ', sglang.__file__); print('reward ', b.__file__)"
```

`sglang` 那行必须指向 `/workspace/Omni-LoRA-RL/sglang/python/...`。指到别处就是被镜像里
那份盖住了，LoRA 会挂到 audio/vision 塔上去。

### 三、准备权重

```bash
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /models/qwen3-omni
```

两个必须存在的文件，缺了都不会立刻报错，而是以奇怪的方式失败：

```bash
# 1. tokenizer.json：sgl-router 是 Rust 写的，注册 tokenizer 时只认 fast 格式
python3 -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('/models/qwen3-omni', trust_remote_code=True, use_fast=True).save_pretrained('/models/qwen3-omni')"

# 2. chat_template.json：Qwen3-Omni 的模板在 processor 里，裸 AutoTokenizer 在
#    transformers 5.x 下读不到，会拿到空模板 —— 模型于是直接吐 <|im_end|>，rollout 全空
test -f /models/qwen3-omni/chat_template.json && echo OK
```

### 四、准备数据

一行一条 JSON，四个字段：

```json
{
  "prompt": "<audio>\nPlease translate the English speech into Chinese. Only output the Chinese translation.",
  "audios": ["/data/s2tt/audio/fleurs_00000123_en.wav"],
  "label": {"ground_truth": "参考译文"},
  "metadata": {"src_lang": "en", "tgt_lang": "zh", "rm_type": "bleu", "src_text": "原文"}
}
```

`prompt` 里的 `<audio>` 是占位符，由 `--multimodal-keys '{"audio": "audios"}'` 把 `audios`
里的路径填进去。`audios` 用绝对路径，单声道 16kHz wav。

复现那条曲线的数据（FLEURS en→zh，validation 前 128 条）：

```bash
pip install "datasets>=2.19,<3" "numpy<2" soundfile librosa "huggingface_hub<0.26"
python3 omni_s2tt/prep_fleurs_s2tt.py --out-dir /data/s2tt --limit 128
```

128 条配 `rollout-batch 8` 是每 16 步一个 epoch，40 步约 2.5 个 epoch。

### 五、跑训练

```bash
cd /workspace/Omni-LoRA-RL

export HF_CKPT=/models/qwen3-omni
export DATA=/data/s2tt/train_s2tt.jsonl
export NUM_ROLLOUT=40
export NUM_GPUS=4
export SAVE_DIR=/data/s2tt/ckpt/s2tt_run1     # 留空则不存档
export SAVE_INTERVAL=5

# chat template 显式喂进去（见第三步）
export CHAT_TEMPLATE_KWARGS=$(python3 -c "
import json; t = json.load(open('$HF_CKPT/chat_template.json'))['chat_template']
print(json.dumps({'chat_template': t}, ensure_ascii=False))")

bash omni_s2tt/run-qwen3-omni-lora-s2tt-4gpu.sh 2>&1 | tee train.log
```

Ray 不用自己起。脚本会 source `Relax/scripts/entrypoint/local.sh`，那里会清理残留进程、
起单机 Ray head、探测 NVLink 并设好 `RUNTIME_ENV_JSON`。**如果你已经有一个 Ray 集群**，
设好 `RAY_ADDRESS` 即可，它检测到之后会转去 `ray-job.sh` 而不是另起一个 head。

> **在共用服务器上注意**：`local.sh` 的清理段是 `pkill -9 python` + `pkill -9 ray`，
> 不区分是谁的进程。在容器里跑没问题（PID namespace 隔离），直接在裸机上跑会连同事的
> 任务一起杀掉。裸机跑的话自己起 Ray 并跳过那一段，`scripts/modal_train_s2tt.py` 走的就是这条路：
>
> ```bash
> export RELAX_ENTRYPOINT_MODE=local        # 让脚本别再 source local.sh
> export RAY_ADDRESS=http://127.0.0.1:8265
> export RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$PYTHONPATH\", \
>   \"PYTHONUNBUFFERED\": \"1\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \
>   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\"}}"
> ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 \
>   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
> ```
>
> 这条路上 `RUNTIME_ENV_JSON` 必须自己给：Ray job 是另一个进程，PYTHONPATH 不传过去
> 就又回到那个 `ModuleNotFoundError: megatron`。

后台跑就套 `nohup ... &` 或 tmux。管道那里注意 `set -o pipefail`（脚本里已经有了）：
不套的话退出码取的是 `tee` 的，训练崩了也报成功。

### 六、记录与看结果

跑在服务器上不像 Modal 有面板兜着，所以先说清楚每样东西落在哪。**只要设了 `SAVE_DIR`，
下面前三样都是自动的**，不用加开关。

#### 逐样本的 reward（最该收的一份）

`$SAVE_DIR/rollout_result/train/<step>.jsonl`，每步一个文件，每行一条样本：

```
rollout_id / sample_index / group_index / prompt / response / reward / label
prompt_length / response_length / total_length / status
```

BLEU 曲线只是这份数据的均值，而排查问题要的是原始 response——译文空了、带 `<|im_end|>`、
还是组内八条完全一样（那样 GRPO 没有梯度），都得翻这里。没设 `SAVE_DIR` 的话显式给
`--rollout-result-dir <目录>` 也行。

统计曲线：

```bash
python3 omni_s2tt/curve.py $SAVE_DIR/rollout_result/train --csv curve.csv
```

它按步打印均值与直方条，并算首尾窗口均值。参照值见上面「做了什么实验」那一节。

#### TensorBoard

默认就开着（`--use-tensorboard` 默认为真）。落盘目录有优先级：`TENSORBOARD_DIR` 环境变量
> `$SAVE_DIR/tensorboard_log` > `tensorboard_log/<项目>/<实验>`（相对路径，会跟着 Ray job
的工作目录跑，不好找）。建议在跑之前显式钉死绝对路径：

```bash
export TENSORBOARD_DIR=/data/s2tt/tb/run1        # 要在 ray start 之前 export
tensorboard --logdir /data/s2tt/tb --host 0.0.0.0 --port 6006
```

曲线在 `rollout/raw_reward`（这一步的 BLEU 均值）、`rollout/rewards`、`response_len/*`
以及 `perf/*` 下面。

#### 文本日志

两处，作用不同：

- `log/qwen3-omni-lora-s2tt-<时间戳>.log` —— 脚本自己 tee 的，是 Ray driver 的输出，
  训练主线程与每步的 metrics 行都在这儿
- `/tmp/ray/session_latest/logs/` —— Ray worker 的日志。**训练崩了要看的是这里**，
  driver 那边往往只剩一句 actor died，真正的 traceback 在 worker 的 `python-core-worker-*.log`
  和 `worker-*.err` 里

服务器上记得把这两处也落到能长期保存的盘：`/tmp` 一重启就没了。

#### wandb（可选）

实验室机器常没外网，用离线模式，回头再 `wandb sync`：

```bash
export WANDB_API_KEY=...          # 在线才需要
# 在训练脚本的 WANDB_ARGS 里加：--use-wandb --wandb-mode offline --wandb-dir /data/s2tt/wandb
```

#### checkpoint

`$SAVE_DIR/iter_XXXXXXX`，含 optimizer 状态，`--load` 指向同一目录即可续跑。默认
`--max-actor-ckpt-to-keep 1`，只留最新一个，想多留就调 `MAX_CKPT_KEEP`。
注意续训路径至今没验证过，见上面「边界」。

## 想调什么，去哪调

超参全部走环境变量，不用改脚本：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LORA_RANK` / `LORA_ALPHA` | 16 / 32 | 记录在案的曲线都是这个配置；改了就没法跟它们比 |
| `LR` | 1e-4 | constant 衰减 |
| `ROLLOUT_TEMPERATURE` | 1.1 | 略升温，让组内译文有方差，GRPO 才有梯度。**改小之前先读 `docs/design/reward-design.md`** |
| `ROLLOUT_BATCH` / `N_SAMPLES` | 8 / 8 | 每步 64 条序列 |
| `GLOBAL_BATCH` | 64 | |
| `SGLANG_MEM_FRACTION` | 0.7 | 推理引擎占的显存比例 |
| `PROJECT_NAME` | Relax/v2/omni-lora-s2tt | tensorboard 项目名 |

换数据集只要换 `DATA`，字段对齐第四节即可；换奖励则改 `--custom-rm-path` 指向你自己的
函数（签名见 `omni_s2tt/bleu_rm.py`）。两者都不需要动 Relax 的代码。

## 踩过的坑

按出现频率排，都是真在这条链路上遇到过的：

1. **`ModuleNotFoundError: No module named 'megatron'`** —— `PYTHONPATH` 覆盖掉了镜像原有的，
   见 B 的第二节。Ray job 里报错，不是主进程。
2. **rollout 全是空输出 / 满屏 `<|im_end|>`** —— chat template 没喂进去。
3. **`RuntimeError: unable to open shared memory object </torch_...>`，而且只死一个 TP rank** ——
   adapter 传输走了共享内存引用。`Relax` submodule 必须在 `lora-omni-v2`（含 `19aea461`）上，
   那个提交把 adapter 改成内联字节。原因见 `docs/migration-v2.md` 的探针九。
4. **sgl-router 起不来，抱怨 tokenizer** —— 缺 `tokenizer.json`，见 B 的第三节。
5. **日志里出现 `[bleu_rm] 响应里出现特殊 token`** —— sglang 返回的文本带 `<|im_end|>`
   之类，拼进 response 后会把 BLEU 压到真实值的约四成。奖励模块**只报不改分**，因为记录在案
   的所有曲线都是在同样不去污染的条件下跑的，改了就没法比。偶发几条可以不管，成片出现
   说明 rollout 那边不对 —— 去查 rollout，别去调奖励。
6. **NCCL 挂在初始化** —— `--shm-size` 太小，或多网卡时要指定 `NCCL_SOCKET_IFNAME`。

---

# 深入

| 文档 | 内容 |
|---|---|
| [`docs/results/experiments.md`](docs/results/experiments.md) | 四次实验的完整记录：配置、逐步曲线、结论、失败的那次为什么失败 |
| [`docs/design/reward-design.md`](docs/design/reward-design.md) | 为什么奖励是 BLEU、温度为什么是 1.1、组内方差怎么造 |
| [`docs/design/simul-port-notes.md`](docs/design/simul-port-notes.md) | 同传移植：改了哪四处、六条已知风险与各自的报错长什么样 |
| [`docs/migration-v2.md`](docs/migration-v2.md) | 当前实现是怎么来的：九个探针的结论、五个上游 PR 的动机与证据 |
| [`scripts/probes/`](scripts/probes/) | 上面每条结论对应的可复现脚本，附索引 |
| `docs/design/my_plan.md` | 早期的规划与参数快照 |
| `docs/results/*.json` | 四条曲线的逐步原始数据，带 `_meta` 说明口径 |

早期实现冻结在本仓库的 `v1` 分支（tag `v1-frozen`），只读，那边有当时的镜像 digest 与
三份归档 patch。
