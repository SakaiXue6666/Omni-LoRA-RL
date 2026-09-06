# Omni-LoRA-RL

Qwen3-Omni Thinker + LoRA 的强化学习训练工程。当前跑通的任务是 **S2TT**（英语语音 →
中文文本），奖励用句级 BLEU，算法用 GRPO，推理侧走 sglang 的 LoRA adapter 热加载。

已验证的结果：4×A100-80GB 上 40 步，BLEU 从前 10 步均值 0.294 升到后 10 步 0.392。
详细的迁移记录、探针结论与逐段对照见 `docs/migration-v2.md`。

本仓库是**入口/hub**，真正的代码以 submodule 指向两个 fork。

## 组成

| 目录 | 来源 | 分支 | 作用 |
|---|---|---|---|
| `Relax/` | fork 自 `redai-infra/Relax` | `lora-omni-v2` | 训练侧（Megatron-Bridge + LoRA + rollout） |
| `sglang/` | fork 自 `sgl-project/sglang` `v0.5.12.post1` | `lora-omni-v2` | 推理侧（Omni 的 LoRA serving） |
| `omni_s2tt/` | 本仓库自带 | - | S2TT 的训练脚本、BLEU 奖励、数据准备、曲线统计 |

之所以还要 fork，是因为有五处改动还没进上游（PR 已提，见 `docs/migration-v2.md` 的「上游 PR」）。
一旦合并，fork 就能退化成"上游 + Omni 专属那两处"。

## 硬件

4 张 80GB 卡（A100/H100）。模型是 30B 的 MoE（激活 3B），配置是 colocate：训练与推理
共用同一批卡，TP4 / EP4 / PP1。单机即可，不需要多机。

显存吃紧的话先调 `SGLANG_MEM_FRACTION`（默认 0.7），它决定推理引擎占多少。

---

## 一、拉代码

```bash
git clone --recursive https://github.com/SakaiXue6666/Omni-LoRA-RL.git
cd Omni-LoRA-RL

# 已经 clone 但忘了 --recursive：
git submodule update --init --recursive
```

核对两个 submodule 落在预期的提交上（对不上就是没 init 干净）：

```bash
git submodule status
# +19aea461... Relax  (heads/lora-omni-v2)
#  02044692... sglang (v0.5.12.post1-5-g02044692cc)
```

## 二、起容器

用 Relax 官方镜像，按 digest 钉死——`:latest` 会漂移，v1 就是因为这个变得不可复现。
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

### PYTHONPATH

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

## 三、准备权重

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

## 四、准备数据

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

## 五、跑训练

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

## 六、记录与看结果

跑在服务器上不像 Modal 有面板兜着，所以先说清楚每样东西落在哪。**只要设了 `SAVE_DIR`，
下面前三样都是自动的**，不用加开关。

### 逐样本的 reward（最该收的一份）

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

它按步打印均值与直方条，并算首尾窗口均值。判据用**窗口均值**而不是单步——单步噪声
很大，v1 的曲线单步能从 0.539 掉到 0.397。健康的 40 步大致是：

| 区间 | 参照值 |
|---|---|
| 前 10 步 | 0.29 左右 |
| 第 31–40 步 | 0.39–0.41 |

### TensorBoard

默认就开着（`--use-tensorboard` 默认为真）。落盘目录有优先级：`TENSORBOARD_DIR` 环境变量
> `$SAVE_DIR/tensorboard_log` > `tensorboard_log/<项目>/<实验>`（相对路径，会跟着 Ray job
的工作目录跑，不好找）。建议在跑之前显式钉死绝对路径：

```bash
export TENSORBOARD_DIR=/data/s2tt/tb/run1        # 要在 ray start 之前 export
tensorboard --logdir /data/s2tt/tb --host 0.0.0.0 --port 6006
```

曲线在 `rollout/raw_reward`（这一步的 BLEU 均值）、`rollout/rewards`、`response_len/*`
以及 `perf/*` 下面。

### 文本日志

两处，作用不同：

- `log/qwen3-omni-lora-s2tt-<时间戳>.log` —— 脚本自己 tee 的，是 Ray driver 的输出，
  训练主线程与每步的 metrics 行都在这儿
- `/tmp/ray/session_latest/logs/` —— Ray worker 的日志。**训练崩了要看的是这里**，
  driver 那边往往只剩一句 actor died，真正的 traceback 在 worker 的 `python-core-worker-*.log`
  和 `worker-*.err` 里

服务器上记得把这两处也落到能长期保存的盘：`/tmp` 一重启就没了。

### wandb（可选）

实验室机器常没外网，用离线模式，回头再 `wandb sync`：

```bash
export WANDB_API_KEY=...          # 在线才需要
# 在训练脚本的 WANDB_ARGS 里加：--use-wandb --wandb-mode offline --wandb-dir /data/s2tt/wandb
```

### checkpoint

`$SAVE_DIR/iter_XXXXXXX`，含 optimizer 状态，`--load` 指向同一目录即可续跑。默认
`--max-actor-ckpt-to-keep 1`，只留最新一个，想多留就调 `MAX_CKPT_KEEP`。

## 想调什么，去哪调

超参全部走环境变量，不用改脚本：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LORA_RANK` / `LORA_ALPHA` | 16 / 32 | 与 v1 对齐；改了就没法跟那条曲线比 |
| `LR` | 1e-4 | constant 衰减 |
| `ROLLOUT_TEMPERATURE` | 1.1 | 略升温，让组内译文有方差，GRPO 才有梯度 |
| `ROLLOUT_BATCH` / `N_SAMPLES` | 8 / 8 | 每步 64 条序列 |
| `GLOBAL_BATCH` | 64 | |
| `SGLANG_MEM_FRACTION` | 0.7 | 推理引擎占的显存比例 |
| `PROJECT_NAME` | Relax/v2/omni-lora-s2tt | tensorboard 项目名 |

换数据集只要换 `DATA`，字段对齐第四节即可；换奖励则改 `--custom-rm-path` 指向你自己的
函数（签名见 `omni_s2tt/bleu_rm.py`）。两者都不需要动 Relax 的代码。

## 踩过的坑

按出现频率排，都是真在这条链路上遇到过的：

1. **`ModuleNotFoundError: No module named 'megatron'`** —— `PYTHONPATH` 覆盖掉了镜像原有的，
   见第二节。Ray job 里报错，不是主进程。
2. **rollout 全是空输出 / 满屏 `<|im_end|>`** —— chat template 没喂进去。
3. **`RuntimeError: unable to open shared memory object </torch_...>`，而且只死一个 TP rank** ——
   adapter 传输走了共享内存引用。`Relax` submodule 必须在 `lora-omni-v2`（含 `19aea461`）上，
   那个提交把 adapter 改成内联字节。原因见 `docs/migration-v2.md` 的探针九。
4. **sgl-router 起不来，抱怨 tokenizer** —— 缺 `tokenizer.json`，见第三节。
5. **日志里出现 `[bleu_rm] 响应里出现特殊 token`** —— sglang 返回的文本带 `<|im_end|>`
   之类，拼进 response 后会把 BLEU 压到真实值的约四成（v1 在同传里量过）。奖励模块只报
   不改分：S2TT 单轮当时没打去污染补丁，这里**故意保持一致**，否则曲线没法跟 v1 比。
   偶发几条可以不管，成片出现说明 rollout 那边不对。
6. **NCCL 挂在初始化** —— `--shm-size` 太小，或多网卡时要指定 `NCCL_SOCKET_IFNAME`。

## 在 Modal 上跑

这套东西最初就是在 Modal 上验的，那条路径仍然保留：

```bash
modal run scripts/modal_train_s2tt.py::check                      # 先查数据与权重（CPU，几十秒）
modal run scripts/modal_train_s2tt.py --num-rollout 40 --detach   # 起训练，spawn 出去与本地解耦
modal run scripts/modal_train_s2tt.py::result --call-id <ID>      # 取结果
```

`scripts/modal_train_s2tt.py` 里的镜像、PYTHONPATH、环境变量与上面几节是一一对应的，可以对着看。

## 相关文档

- `docs/migration-v2.md` —— v2 的迁移全过程：九个探针的结论、五个上游 PR、40 步与 v1 的逐段对照
- `docs/design/my_plan.md` —— 设计文档与实验记录
- v1 在本仓库的 `v1` 分支（tag `v1-frozen`），那边有 `README_v1.md`（镜像 digest 与
  复现步骤）和 `patches/`（v1 相对上游的三份归档 patch）。v1 只读，是这条链路上唯一
  一份在真机上完整跑通过的参考实现，v2 出问题时先去那里对照
