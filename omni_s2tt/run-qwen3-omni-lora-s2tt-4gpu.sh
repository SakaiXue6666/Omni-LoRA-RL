#!/bin/bash
#
# Qwen3-Omni-30B-A3B thinker-LoRA 的 S2TT 训练脚本（colocate，4 卡）。
#
# 这是 v1 `run-qwen3-30B-A3B-omni-lora-smoke.sh` 的 v2 移植版。超参逐项对齐 v1 那次
# 100 步 S2TT 实验（rank 16 / alpha 32、lr 1e-4、temperature 1.1、rollout-batch 8、
# n-samples 8、global-batch 64），因为这次跑 40 步就是为了和 v1 的 BLEU 曲线比。
#
# 相对 v1 的三处变化，都是因为上游变好了或者环境变了：
#
#   1. LoRA 开关变了：v1 是 `--lora-enable --lora-name policy`，v2 上游没有这两个
#      flag —— LoRA 由 `--lora-rank > 0` 打开，adapter 名字是代码里的常量
#      LORA_ADAPTER_NAME='relax_policy_lora'。新增 `--lora-adapter-mode` 走 adapter
#      模式（每步只推 adapter，不推底模）。
#   2. sglang 的 LoRA 参数不用手写了：v1 要显式给 --sglang-enable-lora /
#      --sglang-max-lora-rank / --sglang-lora-target-modules，v2 的 sglang_engine.py
#      会自己从训练侧的 --lora-* 推出来（target 经 convert_megatron_to_hf_target_modules
#      转成 q/k/v/o_proj）。
#   3. 不再禁 cuda graph / custom all-reduce：v1 那两条是因为往旧 slime 镜像里注入了
#      更新的 sglang，内核二进制对不上；v2 用的是 Relax 官方镜像 + 同版本(0.5.12.post1)
#      的 sglang fork，不存在这个错配。
#
# 保留 v1 的 PEFT 安全默认：**不开 sequence-parallel、不开 recompute**。v1 记录过
# recompute + SP + LoRA 会让 lora_B 的 backward 出 NaN。注意 TP>1 时 Omni 的 provider
# 会在 finalize() 里自己把 sequence_parallel 打开，这里不额外叠加。
#
# 用法：
#   HF_CKPT=/models/qwen3-omni DATA=/s2tt/train_s2tt.jsonl NUM_ROLLOUT=40 \
#   bash omni_s2tt/run-qwen3-omni-lora-s2tt-4gpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../Relax/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR:-${SCRIPT_DIR}/../Relax/scripts/models}/qwen3-omni-30B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/v2/omni-lora-s2tt}"
NUM_ROLLOUT="${NUM_ROLLOUT:=40}"
HF_CKPT="${HF_CKPT:-/models/qwen3-omni}"

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${HF_CKPT}
   --megatron-to-hf-mode bridge
)

# 存档/续跑（抗 Modal 抢占）：--save 与 --load 指向同一目录，首跑是空目录就从
# hf-checkpoint 全新开始，被抢占重试时该目录已有 Megatron ckpt，自动接着上次的
# rollout_id 跑（含 optimizer 状态）。
if [ -n "${SAVE_DIR:-}" ]; then
   CKPT_ARGS+=(
      --save "${SAVE_DIR}"
      --load "${SAVE_DIR}"
      --save-interval "${SAVE_INTERVAL:-5}"
      --max-actor-ckpt-to-keep "${MAX_CKPT_KEEP:-1}"
      # 续训时改总步数会让 checkpoint 里 LR 调度器的 total-iters 与新配置冲突，
      # 触发 Megatron 断言。constant 衰减下这个覆盖对 LR 本身没有影响。
      --override-opt_param-scheduler
   )
fi

SYSTEM_PROMPT="You are a helpful assistant."
PROMPT_SET="${DATA:-/s2tt/train_s2tt.jsonl}"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   # v1 是改 Relax 源码加了个 rm_type=bleu 分支；v2 用扩展点，不动框架。
   --custom-rm-path omni_s2tt.bleu_rm.compute_bleu_reward

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH:-8}
   --n-samples-per-prompt ${N_SAMPLES:-8}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-512}
   --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN:-4096}
   # 略升温：组内译文更分散 -> 组内 BLEU 有方差 -> GRPO 才有梯度
   --rollout-temperature ${ROLLOUT_TEMPERATURE:-1.1}
   --global-batch-size ${GLOBAL_BATCH:-64}
   --multimodal-keys '{"audio": "audios"}'
   --balance-data
   --use-fault-tolerance
   --system-prompt "${SYSTEM_PROMPT}"
)

# Qwen3-Omni 的 chat template 在 processor 的 chat_template.json 里，裸 AutoTokenizer
# 在 transformers 5.x 下读不到，tokenizer.chat_template 会是空的。由外部把模板 JSON
# 注入 CHAT_TEMPLATE_KWARGS 显式喂进去。（v1 同样的处理。）
if [ -n "${CHAT_TEMPLATE_KWARGS:-}" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")
fi

if [ -n "${DUMP_DETAILS:-}" ]; then
   ROLLOUT_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi

# lora-target-modules 用裸名字而不是 v1 的通配符：探针七在真模型上确认过塔不会被
# 误挂（v2 的 audio/vision 是 HF 模块，叫 q_proj/k_proj/v_proj，压根不叫 linear_qkv），
# 而通配符会原样落进 adapter_config.json 导致导出侧点不到模块（已提 Relax #261）。
LORA_ARGS=(
   --lora-rank ${LORA_RANK:-16}
   --lora-alpha ${LORA_ALPHA:-32}
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-adapter-mode
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-4}
   --lr-decay-style constant
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.95
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION:-0.7}
)

# colocate 4 卡：TP4 / EP4（128 个 expert 能整除）/ PP1。
PERF_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1
   --micro-batch-size 1
)

WANDB_ARGS=(
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-omni-lora-s2tt-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address=${RAY_ADDRESS:-"http://127.0.0.1:8265"} \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   ${RUNTIME_ENV_JSON:+--runtime-env-json="${RUNTIME_ENV_JSON}"} \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 4], "rollout": [1, 4]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-omni-lora-s2tt-${now}.log
