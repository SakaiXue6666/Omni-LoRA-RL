# omni-lora-rl

Qwen3-Omni Thinker + LoRA 的强化学习训练工程（当前任务：同传 speech-to-text，simul S2TT）。

本仓库是**入口/hub**，真正的代码以 submodule 形式指向两个定制 fork。

## 组成

| 目录 | 来源 | 分支 | 作用 |
|---|---|---|---|
| `Relax/` | fork 自 `redai-infra/Relax` | `lora-omni-baseline` | 训练侧（Megatron + LoRA + rollout + reward） |
| `sglang/` | fork 自 `sgl-project/sglang` | `lora-omni-baseline` | 推理侧（Omni LoRA serving） |
| `IMPORTANT/` | 本仓库自带 | - | 设计文档与实验记录（`my_plan.md`） |

## 一键获取

```bash
git clone --recursive https://github.com/<你的账号>/omni-lora-rl.git
# 若已 clone 但忘了 --recursive：
git submodule update --init --recursive
```

## 环境（不入 git，走 Docker 镜像）

- 训练/推理镜像：`<你的镜像 name:tag>`（含 Megatron-Bridge / flashinfer 等依赖）
- 依赖细节见 `Relax/pyproject.toml`

## 模型与数据（不入 git）

- 基座模型：`<Qwen/Qwen3-Omni-30B-A3B-...>`（HF 下载或实验室共享盘）
- 训练数据：`<HF dataset repo 或服务器路径>`

## 跑起来

1. 拉环境：`docker pull <镜像>`（或在 Modal 用同一镜像）
2. 下模型到 `<路径>`
3. 启动训练：

```bash
bash Relax/scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-simul.sh
```

## 关键改动一览

- 训练侧（`Relax/`）：Qwen3-Omni thinker 挂 LoRA、权重转换/同步契约、BLEU reward、同传多轮 rollout。
- 推理侧（`sglang/`）：Omni LoRA serving（should_apply_lora gate + audio pad + 热加载兼容）。
- 详见 `IMPORTANT/my_plan.md`。

---

| 功能 | 标准 SGLang | 当前 SGLang-Omni backend |
|---|---:|---:|
| Relax 启动推理服务 | 支持 | 不支持，必须预先启动 |
| 固定拓扑 rollout | 支持 | 支持 |
| tensor LoRA 热更新 | 支持 | 已接通 |
| 动态增加推理副本 | 支持 | 暂不支持 |
| rollout offload/onload | 支持 | 暂不支持 |
| fully async/DCS | 支持 | 暂不支持 |
| PD disaggregation | 支持 | 暂不支持 |
| worker 崩溃后自动重建服务 | 较完整 | 只能检查/重连，不能重建完整 Omni |
| Talker/Code2Wav | 不属于普通 SGLang rollout | 支持 |

因此目前主要影响三个方面：
不能在训练途中动态扩推理容量
比如训练开始时是一个 Thinker TP4 Omni 服务，后面不能让 Relax 自动变成两个 Omni 副本。
但你现在每个 prompt 采样 4 条，只是向同一个 TP4 服务并发发 4 个请求，不等于需要 4 个推理副本，所以没有问题。
故障恢复能力弱一些
标准 SGLang Ray-native 模式下，某个 rollout engine 死掉后，Relax 有机会：
重新申请 GPU
→ 重新启动 SGLang server
→ 同步当前权重
→ 注册回 router
当前 Omni proxy 只能：
检查外部 Omni 是否还活着
→ 活着就重新连接
→ 真正的 Thinker/Talker 进程死了则无法自行重建
在 Modal 上通常表现为整个任务失败后重跑，而不是单独恢复 Omni。
不能开启部分标准 SGLang 高级参数
当前 Omni 明确拒绝：
--fully-async
--offload-rollout
elastic scale-out
多个 external Omni router 地址
PD worker
如果以后训练脚本误开这些参数，会提前报错，而不是悄悄跑错。
对标准 SGLang 本身没有影响，因为选择逻辑是：
普通 sglang_rollout.py
→ 没有 ROLLOUT_ENGINE_CLASS
→ 继续使用原来的 SGLangEngine
→ 所有标准能力保持不变

sglang_omni_rollout.py
→ 显式声明 SGLangOmniEngine
→ 只在 Omni 训练中使用功能较窄的 proxy
所以最合理的策略是：
现在保留 Omni 的固定 external 模式，先完成训练和语音 reward；
对不支持的功能明确报错；
不为了“看起来和 SGLang 一样”而实现假的 scale-out；
真正需要吞吐扩容或容错时，再为 Omni 单独实现完整 launcher。
未来 Omni 的真实扩容不能只是多建 proxy，而应该是：
申请一整组 GPU
→ 启动新的 Omni Coordinator
→ 启动新的 Thinker TP ranks
→ 启动 Talker/Code2Wav/encoder stages
→ 加载当前 LoRA
→ 把整套 Omni 副本注册到 Omni router
