#!/usr/bin/env bash
# 同传（simultaneous S2TT）训练：整段音频按定长块多轮 rollout。
#
# 代码在 Relax/examples/simul_s2tt/（submodule 里）。本脚本只是 run-qwen3-omni-lora-s2tt-4gpu.sh
# 的薄包装 —— 超参、LoRA、并行、奖励全部复用那份，
# 只加两件事：
#   1. --custom-generate-function-path 换成同传的多轮 generate
#   2. --custom-config-path 指向 max_turns / simul_chunk_ms
# 这样单轮那条基线改了参数，同传自动跟着变，不会各自漂移。
#
# ⚠️ 这条路径**从未在 v2 上跑过**。移植后的已知风险与预期报错见
#    docs/design/simul-port-notes.md，上卡之前先读一遍。
#
# 用法：
#   HF_CKPT=/models/qwen3-omni DATA=/s2tt/train_s2tt.jsonl NUM_ROLLOUT=20 \
#   SAVE_DIR=/data/s2tt/ckpt/simul_run1 \
#   bash omni_s2tt/run-qwen3-omni-lora-simul-4gpu.sh
#
# 注意数据要求：每条样本恰好一段**整段**音频（AudioChunkEnv 会自己切块）。
# 单轮那份 128 条数据可以直接用；v1 当时用的是卷里 97 条整段音频，平均 9.6 秒。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 同传专属参数。CUSTOM_CONFIG_PATH 可覆盖（改 chunk 时长 / 轮数上限）。
export EXTRA_ARGS="${EXTRA_ARGS:-} \
   --custom-generate-function-path examples.simul_s2tt.rollout.generate \
   --custom-config-path ${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/Relax/examples/simul_s2tt/config.yaml}"

# 日志和 tensorboard 项目名分开，别和单轮那条曲线混在一起。
export LOG_NAME="${LOG_NAME:-qwen3-omni-lora-simul}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/v2/omni-lora-simul}"

# v1 的同传是 base 冻结两端常驻（比带 offload 快约 27~35%），代价是显存更紧，
# 所以推理引擎的占比要降下来。想回到带 offload 的配置就把这三行注释掉。
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.55}"
export EXTRA_ARGS="${EXTRA_ARGS} --no-offload-train --no-offload-rollout"

exec bash "${SCRIPT_DIR}/run-qwen3-omni-lora-s2tt-4gpu.sh"
