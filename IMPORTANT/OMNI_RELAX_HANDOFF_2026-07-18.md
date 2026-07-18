# Relax × SGLang-Omni LoRA 当前状态与接手指南

> 更新时间：2026-07-19 01:55（Asia/Shanghai）
> 用途：换 Codex/AI 会话、交给其他开发者、或中断后继续工作时的当前事实来源。
> 精确代码始终以 Git 为准；本文件记录架构、验证边界、提交位置和下一步。

## 1. 当前仓库与提交

父仓库：

- 路径：`D:\Li_Lab\RL\omni-lora-rl`
- 远端：`https://github.com/SakaiXue6666/omni-lora-rl.git`
- 分支：`main`
- 本轮提交前基线：`109eefa69e8449da7726302f12d96bf26f75b9f3`
- 当前提交以 `git log -1 --oneline` 为准。

子仓库：

| 仓库 | 分支 | 当前提交 | 远端 |
|---|---|---|---|
| `sglang-omni` | `lora-omni-baseline` | `7316e0ce9085a1a4c207a5c6320596a85a58473e` | `SakaiXue6666/sglang-omni` |
| `Relax` | `lora-omni-baseline` | `bb103df` | `SakaiXue6666/Relax` |
| 标准 `sglang` | `lora-omni-baseline` | `d8a1f27e721a0859f5e61b0916280ab2c4dff48f` | `SakaiXue6666/sglang` |

父仓库的 gitlink 已精确指向上表两个新提交。正常状态下父仓库只会看到以下历史排障日志未跟踪：

```text
e2e_last.log
e2e_run3.log
e2e_run4.log
e2e_run5.log
e2e_stream.log
```

不要删除、暂存或把这些日志当作最终成功证据。

### 1.1 2026-07-18 至 19 日后续改动

以下改动已包含在 `Relax bb103df` 和本轮父仓库提交中，接手时不要重复应用：

- 父仓库新增 `modal_relax_omni_lora_smoke.py`。
- Relax 修改：
  - `examples/simul_s2tt/omni_rollout.py`
  - `relax/distributed/ray/rollout.py`
  - `relax/engine/rollout/sglang_rollout.py`
  - `relax/engine/rollout/sglang_omni_rollout.py`
  - `scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-omni-simul.sh`
  - `tests/distributed/ray/test_rollout_engine_selection.py`
  - `tests/engine/rollout/test_sglang_omni_rollout.py`

这些改动解决：

1. Relax 实际 sampling params 中 Omni 不接受的字段会导致首请求 422：
   现在显式白名单过滤，并把 `sampling_seed` 映射为 `seed`。
2. custom generate 不再继承标准 SGLang group wrapper 预编码的整段音频 base64，
   避免重复编码和每个 Sample 携带无用大对象。
3. 每轮 rollout 结束不再对单个 Omni 服务错误调用标准 router 的
   `/workers` + `/abort_request`；Omni marker 声明独立 abort hook，
   使用 `/pause_generation(mode=abort)` 后 `/continue_generation`。
4. 输入残留 `<image>`/`<video>` 会 fail-fast；当前 Omni S2TT 只支持 audio + text，
   避免默认 AVQA 数据被静默降级。
5. rollout 返回前显式检查 tokens、loss mask、logprobs、response length 对齐。
6. Qwen3-Omni 的 `AutoTokenizer.chat_template` 为空、但
   `AutoProcessor.chat_template` 包含官方模板时，Omni rollout 会把 processor
   模板回填给 tokenizer；显式 `apply_kwargs["chat_template"]` 和 tokenizer
   自带模板仍优先，原有标准 SGLang 路径不受影响。
7. 训练脚本允许用环境变量覆盖 `CUSTOM_CONFIG_PATH`，并允许通过
   `MAX_GLOBAL_RESTART` 把 smoke 的失败重试关闭；未设置时仍保持原有配置与
   health check 行为。

## 2. 已完成的系统关系

### 2.1 Thinker tensor LoRA 热更新

`sglang-omni` 已具备独立 tensor LoRA 控制链：

```text
POST /load_lora_adapter_from_tensors
→ Client
→ Coordinator
→ AdminMessage / msgpack / ZMQ
→ StageRuntime leader
→ TPLeaderFanout
→ Thinker rank 0 + follower ranks
→ OmniScheduler
→ ModelWorker
→ 标准 SGLang ModelRunner / LoRAManager
```

关键约束：

- 只加载 Thinker LoRA。
- `lora_path="__tensor__"`。
- 使用标准 SGLang 的 `MultiprocessingSerializer.deserialize`、安全 unpickler、`monkey_patch_torch_reductions` 和 `FlattenedTensorBucket`。
- 没有使用不受限制的 `pickle.loads`。
- TP 广播复用 Omni 原有 admin control plane，没有只在 rank 0 加载。
- `/update_weights_from_tensor` 的全量 base 权重数据面仍不在本项目范围内。

对应提交：

- `sglang-omni b6e5f2c`：tensor LoRA HTTP→TP→ModelRunner 主链。
- `sglang-omni 7316e0c`：Relax Omni rollout 所需的 structured messages 和 `/flush_cache`。

### 2.2 Relax 独立 Omni backend

Relax 保留原有标准 SGLang backend，同时新增独立 Omni 路径：

```text
rollout_function_path
→ _resolve_rollout_engine_class()
→ 默认仍是 SGLangEngine
→ Omni rollout module 显式声明 SGLangOmniEngine
```

Omni backend 当前定位：

- 连接外部 SGLang-Omni router，不负责在 Relax actor 内启动模型。
- 当前只支持 regular rollout。
- 不宣称支持 fully async、offload、elastic scale-out。
- 支持 health、model info、flush cache，以及训练侧 updater 已使用的 LoRA admin endpoints。

关键文件：

- `Relax/relax/backends/sglang_omni/omni_engine.py`
- `Relax/relax/engine/rollout/sglang_omni_rollout.py`
- `Relax/relax/distributed/ray/rollout.py`
- `Relax/examples/simul_s2tt/omni_rollout.py`
- `Relax/scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-omni-simul.sh`

对应提交：`Relax 6d55611`。

### 2.3 多轮音频请求契约

Relax Omni rollout 发给 `/generate` 的请求为：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio"},
        {"type": "text", "text": "..."}
      ]
    },
    {"role": "assistant", "content": "..."},
    {
      "role": "user",
      "content": [{"type": "audio"}]
    }
  ],
  "metadata": {
    "audios": [
      "data:audio/wav;base64,...",
      "data:audio/wav;base64,..."
    ]
  },
  "stage_params": {
    "thinker": {
      "lora_name": "policy"
    }
  },
  "return_logprob": true,
  "output_modalities": ["text"]
}
```

规则：

- structured message 中的 audio marker 按遍历顺序与 `metadata.audios` 一一对应。
- marker 数量和 audio 数量不相等时硬失败。
- 旧字符串 messages 继续保留原有自动注入行为。
- Relax 共享编码器仍产出标准 SGLang 依赖的 `data:,<base64>`；只有 Omni rollout 边界将其改写为 `data:audio/wav;base64,<base64>`。
- 这保证了原有标准 SGLang rollout 请求格式没有变化。

训练对齐规则：

- 模型生成 token：`loss_mask=1`，保存服务端真实 logprob。
- 后续 audio observation 经 processor 展开的 token：`loss_mask=0`，logprob 填 `0.0`。
- 初始 prompt 不计入 `response_length`。
- 缺少 logprob、token/logprob 数量不一致或 malformed finish reason 时硬失败。

### 2.4 Cache 生命周期

训练侧更新前后的缓存控制为：

```text
pause / drain（由现有训练生命周期负责）
→ /flush_cache
→ /unload_lora_adapter
→ /load_lora_adapter_from_tensors
→ continue generation
```

`OmniScheduler.flush_cache()` 是 Omni 原生实现，检查其实际 request/build/result 队列是否为空，并清理：

- prefix tree cache
- request-to-token pool
- KV allocator
- grammar manager
- metrics
- 可选 draft worker cache
- CUDA allocator cache

不要改回直接委托标准 Scheduler `flush_cache()`；标准实现依赖 OmniScheduler 没有构造的 pipeline-parallel `ps` 状态。

## 3. 已执行的验证

### 3.1 CPU

- sglang-omni LoRA/admin/flush 直接相关测试：`37 passed`
  - Modal app：`ap-yjxyjKcRQa4Xwsn1fNc3qI`
- Relax Omni rollout/data URL 直接相关测试：`5 passed`
  - Modal app：`ap-YEmLs18Gpenu1XSCnKTGWZ`
- 更早的分层回归：
  - sglang-omni：`31 passed, 167 deselected`
  - Relax：`14 passed`
- 本轮真实 launcher 相关 CPU 验证：
  - 镜像/版本隔离 probe：`PASS`
    - Modal app：`ap-u7uhSHraEviXze0KLEbaxt`
    - Relax 使用 Transformers `5.3.0`
    - Omni 独立 venv 使用 Transformers `5.6.0`
  - 第一轮 targeted tests：`18 passed`
    - Modal app：`ap-31TfQ7JcUVdawdbrZEH3Oa`
  - 加入 modality fail-fast 和长度断言后：`20 passed`
    - Modal app：`ap-OqEyNgNmaVBW6y2qLQZXfm`
  - 加入 processor chat template 回填回归后：`21 passed, 26 warnings`
    - Modal app：`ap-P9otWUBHYDjUK6fJSk6yB7`
- 本次 BLEU smoke helper 与 shell 覆盖 probe：`PASS`
  - Modal app：`ap-3Jq2yxgLkvEuOpT5UlV11P`
  - Relax Transformers `5.3.0`、Omni venv Transformers `5.6.0`
  - FLEURS 数据转换、非零 `grad_norm` evidence guard、`bash -n` 均通过
- py_compile / AST、bash syntax、`git diff --check` 通过。
- 当时环境没有 ruff，因此没有运行 ruff。

### 3.2 GPU：已通过的真实边界

最终 TP2 GPU 冒烟：

- Modal app：`ap-wKDAorWQE7kxUbB2fCu9aA`
- GPU：`2 × A100-80GB`
- Thinker：TP2，rank 0/1 分别绑定 GPU 0/1。

通过内容：

1. `/model_info` 返回 Thinker `tp_size=2`，包含 leader/follower rank results。
2. structured messages + 两轮真实 WAV data URL 推理成功。
3. base 两轮均返回 token logprobs。
4. `/flush_cache` 在两个 TP rank 成功。
5. 第一次随机 tensor LoRA 在两个 TP rank 成功加载。
6. `stage_params.thinker.lora_name="policy"` 路由生效。
7. LoRA 两轮均返回 token logprobs。
8. flush → unload → 第二次不同 tensor LoRA 加载成功。
9. 第二次 LoRA 两轮均成功。
10. base → update 1、update 1 → update 2 的第一轮 logprobs 均发生变化。
11. 最终输出 `PASS`。

此前单独的 tensor LoRA TP1/TP2 冒烟也已通过；TP2 app 为 `ap-CdRuwe7lChN85DZqxGMhEY`。

## 4. 必须诚实区分的验证边界

截至 2026-07-19，真实 Relax Ray/Megatron 与 SGLang-Omni 的
`num_rollout=2` 非零奖励训练闭环已经通过，不再只是 Relax-compatible 模拟客户端：

```text
真实 Relax Ray RolloutManager
→ SGLangOmniEngine
→ structured audio / omni_rollout.generate
→ 初始真实 Megatron LoRA tensor load
→ Thinker TP4 rollout
→ Megatron loss + optimizer step
→ UpdateLoRAFromTensor 导出训练后 adapter
→ TP4 unload / reload
→ 下一轮 rollout 消费训练后的 policy
→ 再次训练与 TP4 unload / reload
→ 作业成功退出
```

本次两步 `grad_norm` 都非零，且第二轮请求发生在第一次训练后 TP4 reload 之后，
因此“训练产生 adapter 更新并被下一轮消费”的关系已经验证。仍未验证的是
960ms chunk 下的真实多轮 observation，以及超过两个 step 的学习趋势；不要把
两个 reward group 夸大成收敛或稳定提升。

### 4.1 正确的真实闭环拓扑

`IMPORTANT/my_plan.md` 记录的 100-step 成功基线是：

```text
单节点 4 × A100-80GB
Megatron TP4 / EP4
标准 SGLang TP4
两者 colocate 在同一组 GPU 0–3
no-offload train + no-offload rollout
SGLang mem_fraction_static=0.55
```

外部 SGLang-Omni 也必须先按这一拓扑验证，不要再拆成 4+2、3+2 或申请
5/6 卡。Modal 的 `A100-80GB:4` 就是四张卡，不需要按 8 卡申请。

### 4.2 2026-07-18 TP4 colocate 实测

真实闭环 app：`ap-6U79MAHwQcYiyveN6CnF3C`

- 实际获得 `4 × A100-80GB`。
- SGLang-Omni Thinker TP4 成功启动，rank 0–3 分别绑定物理 GPU 0–3。
- `/model_info` 返回 `tp_size=4` 及四个 rank results。
- Relax 资源校验返回：
  `required GPUs=4, cluster GPUs=4, colocate=True`。
- actor placement group 的四个 bundle 明确落到同一节点 GPU 0、1、2、3；
  与 Omni Thinker 使用的是同一组卡。
- Megatron 四个 worker 随后实际启动，并确认 TP4 / EP4 provider override。

该次没有进入 rollout/tensor update，因为启动顺序与标准链路相反而 OOM：

```text
错误顺序：
先启动外部 Omni
→ 再启动 Relax / Megatron actor
```

定量日志：

- Omni 每 rank Thinker 权重约 `14.32 GiB`。
- `mem_fraction_static=0.55` 分配 K/V cache 各约 `14.22 GiB`，
  即 KV 合计约 `28.44 GiB`。
- Omni 启动后每卡合计先占约 `43.5 GiB`。
- Megatron 在构造 `TEColumnParallelGroupedLinear` 的临时峰值阶段将 GPU 0
  顶满，只剩 `7.75 MiB`，再申请 `20 MiB` 时 OOM。

这不是 TP placement 错误，也不是 tensor LoRA endpoint 错误；它发生在
Megatron 模型构造阶段，尚未执行 rollout。

发现 OOM 后立即停止，最终 Modal 状态为 `stopped`、`tasks=0`，
没有自动重跑。

### 4.3 Megatron-first 启动顺序已完成 GPU 复验

`modal_relax_omni_lora_smoke.py` 已改为复刻 Relax + 标准 SGLang 的成功生命周期：

```text
启动 Ray
→ 启动 Relax
→ 等待 Megatron Actor 完成初始化并释放构造期临时峰值
→ 启动 colocated SGLang-Omni Thinker TP4
→ SGLangOmniEngine 使用已有 300 秒 health retry 连接外部 Omni
```

launcher 会实时转发训练日志；若 Relax 在等待 Omni 时退出，会立即失败并清理，
避免继续空耗 GPU。

真实复验 app：`ap-OK4hEqgzY0D9VjWc59SwOk`

- Megatron Actor 先在 GPU 0–3 完成 TP4 / EP4 模型构造和 checkpoint 加载。
- 每个 rank 参数量为 `8831871600`；192 个 LoRA adapter 参数完成挂载，
  共 `2.556M` 个可训练元素。
- Actor 从 step 0 初始化完成并输出
  `[actor] Service deployed successfully` 后，launcher 才启动 Omni。
- 此时 Omni 各 rank 启动前可用显存约 `56.8–57.2 GiB`。
- Omni Thinker TP4 每 rank 权重占 `14.32 GiB`。
- `mem_fraction_static=0.55` 下，每 rank KV cache 的 K/V 各约 `8.17 GiB`，
  加载完成后每卡仍有约 `24.9–25.7 GiB` 可用。
- rank 0–3 和所有 stage 均启动成功，因此 Megatron-first 已解决此前
  Omni-first 的构建期 OOM；4×A100 同卡 colocate 不需要拆池或扩卡。

该次仍未进入 rollout，因为 Ray Serve 已占用 `127.0.0.1:8000`，Omni 自动改用
`35314`，而 Relax 仍轮询配置的 8000。确认根因后立即停止 app，最终
`tasks=0`。

launcher 随后将外部 Omni 固定改为 `127.0.0.1:30000`：

- `launch_server(..., port=30000)`；
- `_wait_for_omni()` 检查 30000；
- `OMNI_ROUTER_PORT=30000`；
- Relax 训练脚本把它传入
  `--rollout-external-engine-addrs 127.0.0.1:30000`。

该端口修复已通过 `py_compile` 和参数链静态核对。成本受控复验 app
`ap-y04TXMOduwaPYkcLwCf1On` 一直等待 `4 × A100-80GB` 容量，始终
`tasks=0`、没有获得 GPU；为避免无人监控时突然开始计费，已主动停止。

更早的 `ap-Dnwm9qjMeTeEEKVVMpaChI` 请求过 6 卡，但只排队、没有获得 GPU，
已经停止。它不是有效测试，也不应再作为拓扑依据。

### 4.4 真实 Relax × SGLang-Omni 单步闭环已通过

最终成本受控测试：

- Modal app：`ap-VXUZMqOw9PLgxreritbscf`
- Ray job：`raysubmit_PneTEERMkXu334Kq`
- 资源：单机 `4 × A100-80GB`
- 配置：Megatron TP4 / EP4 与 Omni Thinker TP4 colocate 在 GPU 0–3
- 参数：`num_rollout=1`、一个样本、极短响应
- 最终状态：Modal `stopped`、`tasks=0`

真实证据链：

1. Relax 校验 `required GPUs=4, cluster GPUs=4, colocate=True`。
2. Megatron 四个 rank 完成 checkpoint 4287/4287；每 rank
   `8831871600` 参数，挂载 192 个 LoRA tensor、`2.556M` 可训练元素。
3. Actor ready 后才启动 Omni；Omni rank 0–3 使用同一组 GPU 0–3。
4. Omni 每 rank 权重约 `14.32 GiB`，K/V cache 各约 `8.17 GiB`；
   启动后每卡仍有约 `24.95–25.74 GiB` 可用。
5. 初始同步从真实 Megatron adapter 抽取 192 个 bf16 tensor；
   `/flush_cache` 与 `/load_lora_adapter_from_tensors` 均返回 200。
6. 四个 Thinker rank 都完成 `LoRARef(policy, policy, __tensor__)` 加载；
   prefill 均记录 `lora_ids=['policy']`、`has_active_lora=True`。
7. 两个真实 structured audio `/generate` 请求均返回 200、各生成 2 token；
   rollout 结果为 `A`，reward `1.0`，并把 tokens、loss mask、真实 logprobs、
   multimodal train inputs 送入 Megatron。
8. Megatron 完成 step 0，记录 `train/loss=0.0`、
   `train/entropy_loss=0.000392913818359375` 和训练性能指标。
   这个单样本/单答案 smoke 的 advantage 为 0，因此 loss/grad norm 为 0；
   它验证执行链，不证明一次优化产生了可观的权重变化。
9. 训练后再次从 Megatron 导出 192 个 adapter tensor，并在四个 Omni rank
   完成 flush → unload → tensor reload。
10. 日志依次出现 `Actor training completed step 0/1`、
    `All training steps finished`、`Main func successfully`、
    `Job 'raysubmit_PneTEERMkXu334Kq' succeeded` 和
    `REAL RELAX OMNI SMOKE PASS (num_rollout=1)`。

首次进入真实请求时暴露并修复了 Qwen3-Omni chat template 契约：
checkpoint 的 `AutoTokenizer.chat_template` 为 `None`，官方模板位于
`AutoProcessor.chat_template`。Relax Omni rollout 现在会安全回填该模板，
修复后两次 structured audio 请求与后续训练均通过。

Modal 客户端 stderr 的 “Timed out waiting for final app logs” 是本地客户端在
远端工作已经完成后的日志等待超时；远端 Ray job 成功、launcher 返回码为 0，
且 app 最终 `stopped/tasks=0`，不是训练失败。

### 4.5 非零奖励的两步训练与下一轮消费已通过

为避免单答案 MCQ 的 group reward 全为 1、GRPO advantage 为 0，本次成本受控
测试改用已有 FLEURS 英译中样本、原有 BLEU reward，并让四个采样形成同一
reward group：

- Modal app：`ap-YUK8ye4Kyn6azypakTqz8B`
- Ray job：`raysubmit_6c5pDJSnpYpW5uay`
- 资源：单机 `4 × A100-80GB`
- 拓扑：Megatron TP4 / EP4 与 Omni Thinker TP4 colocate 在 GPU 0–3
- 参数：`num_rollout=2`、`rollout_batch_size=1`、`n_samples_per_prompt=4`、
  `global_batch_size=4`、temperature `1.3`
- reward：`bleu`
- 最终状态：Modal `stopped`、`tasks=0`

为控制 GPU 成本，临时 smoke config 使用 `max_turns=1` 和 30 秒 chunk，
把 6.54 秒音频作为一次完整 observation；两轮各发四个 `/generate` 请求，
约八次生成，而不是按 960ms chunk 放大请求数。数据与 reward 都复用已有
FLEURS/Relax 路径，没有新增训练 reward 实现。

关键证据：

1. Megatron 四个 rank 完成 checkpoint 4287/4287；每 rank
   `8831871600` 参数，挂载 192 个 LoRA tensor、`2.556M` 可训练元素。
2. 初始 192 个 bf16 adapter tensor 在四个 Thinker rank 加载成功；
   所有 rank 的 prefill 都记录 `lora_ids=['policy']`、
   `has_active_lora=True`。
3. rollout 0 四个请求均返回 200；首个翻译为
   “当你给离你成千上万里的人打电话时，你就是在使用卫星。”，
   首个 BLEU reward 为 `0.19359517339258717`，该 group 平均 raw reward
   为 `0.19961697794497013`。
4. Megatron step 0 的 `train/grad_norm=2.2268602674041342`，
   `train/entropy=0.5976698398590088`；这证明该 smoke 不再是零 advantage
   的纯执行链。
5. step 0 后完成四 rank flush → unload → tensor reload；随后才出现第二轮
   request IDs，且四个 rank 继续记录 active `policy`，证明下一轮实际消费了
   训练后 reload 的 adapter。
6. rollout 1 四个请求均返回 200，group 平均 raw reward 为
   `0.24674632865935564`；Megatron step 1 的
   `train/grad_norm=2.0462279716708838`、`train/entropy=0.5585435628890991`。
7. step 1 后再次完成四 rank unload/reload；日志出现
   `Actor training completed step 1/2`、`All training steps finished`、
   `Main func successfully`、Ray job succeeded，以及
   `[evidence] non-zero training grad_norms=[2.2268602674041342, 2.0462279716708838]`
   和 `REAL RELAX OMNI SMOKE PASS (num_rollout=2)`。

第二组平均 BLEU 高于第一组是本次观测事实，但样本、group 和 step 数都太少，
不能据此声称 reward 已形成稳定上升趋势。

## 5. 下一阶段优先目标

核心训练关系已经接通。下一次 GPU 测试应从以下两个尚缺边界中选择一个，
不要再次重复本次两步整段音频 smoke：

```text
A. 恢复 960ms chunk，验证真实 AudioChunkEnv 多轮 observation
   与累计 metadata.audios、token/loss-mask/logprob 对齐

B. 保持整段音频的低请求数配置，增加到少量 3–5 steps，
   观察非零 grad 是否持续；若要判断学习趋势，需要更长的正式训练
```

继续遵守：

- 严格使用成功基线的 `4×A100-80GB` 同卡共置：
  - Megatron TP4 + EP4 使用 GPU 0–3。
  - 外部 SGLang-Omni Thinker TP4 也使用 GPU 0–3。
- 必须先完成 Megatron Actor 初始化，再启动 Omni；不要恢复成 Omni-first。
- 先保持 `mem_fraction_static=0.55`，用正确启动顺序验证；不要在没有证据时
  先改并行度或扩卡。
- 复用现有模型和 FLEURS 数据 Volume。
- 继续使用 text output + 原有 BLEU reward；当前只训练 Thinker LoRA，
  不要把 speech/Talker reward 混入本阶段。
- 启动、协议或首个训练 step 失败时停止，不盲目重试。

仍待验证：

1. 更真实的 AudioChunkEnv 多轮 observation 是否与累计
   `metadata.audios` 对齐；当前训练 smoke 为单轮完整音频。
2. 两步以上是否持续产生合理 reward/gradient；本次只证明非零更新与下一轮消费，
   不证明长期收敛。
3. 如果使用 Omni Router 多 worker 广播，需要补一次真实 GPU router 验证；
   目前 Router 主要是 CPU 覆盖，TP fanout 已在 TP4 GPU 覆盖。

## 6. 暂不做的内容

- 不迁移 Megatron-Bridge 0.5.0；`MIGRATION_PLAN.md` 已记录暂停原因。
- 不实现 Talker LoRA。
- 不实现 audio codec action/logprobs。
- 不实现全量 base `/update_weights_from_tensor`。
- 不修改原有标准 SGLang S2TT backend 请求语义。
- 不把早期 `e2e_*.log` 当作成功基准。

## 7. 新会话启动检查

```powershell
cd D:\Li_Lab\RL\omni-lora-rl

git status -sb
git log -1 --oneline

git -C sglang-omni status -sb
git -C sglang-omni log -1 --oneline

git -C Relax status -sb
git -C Relax log -1 --oneline

git -C sglang status -sb
git -C sglang log -1 --oneline

git ls-files --stage Relax sglang-omni
```

期望基线提交：

- 父仓库：本轮提交前为 `109eefa`；当前以 `git log -1 --oneline` 为准。
- sglang-omni：`7316e0c`，干净。
- Relax：`bb103df`，干净。
- 标准 sglang：`d8a1f27e7`，干净。
- 本交接文档和新 Modal launcher 已由父仓库跟踪；只有五个
  `e2e_*.log` 保持未跟踪。

## 8. 给新 AI 的最短提示词

```text
先完整阅读 IMPORTANT/OMNI_RELAX_HANDOFF_2026-07-18.md 和 Relax/AGENTS.md。
不要修改标准 sglang，不要删除或暂存父仓库 e2e_*.log。
当前 tensor LoRA + structured audio + TP4 已通过，CPU targeted tests 21 passed。
真实 4×A100 同卡共置的非零奖励 num_rollout=2 闭环已经成功：
Modal app ap-YUK8ye4Kyn6azypakTqz8B，Ray job raysubmit_6c5pDJSnpYpW5uay。
它使用 FLEURS + BLEU、n_samples=4，完成两个 Megatron optimizer step；
grad_norm 为 2.2269、2.0462。step 0 训练后的 192 tensor adapter 在四个
Thinker rank reload 后被 rollout 1 实际消费，最后再次 reload，并输出
REAL RELAX OMNI SMOKE PASS (num_rollout=2)；app 已 stopped/tasks=0。
不要重复应用 1.1 节已经提交的修复。
本次为控成本使用单轮完整音频；下一缺口是 960ms AudioChunkEnv 多轮对齐，
或少量 3–5 steps 的持续非零梯度验证，不要重复本次测试。
使用 modal_relax_omni_lora_smoke.py；严格复刻成功基线：
Megatron TP4 + Omni Thinker TP4 colocate 在同一组 4×A100-80GB。
launcher 已改为 Megatron Actor ready 后再启动 Omni，并把 Omni 独立固定到 30000。
失败时停止分析，不盲目重跑。
```
