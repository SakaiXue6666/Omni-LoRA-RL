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
