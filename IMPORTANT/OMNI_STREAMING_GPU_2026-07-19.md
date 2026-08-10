# Relax × SGLang-Omni 960ms 流式 GPU 验证记录

日期：2026-07-19

## 结论

真实 `AudioChunkEnv` 960ms 多轮链路已经在单机
`4 × A100-80GB` 上进入训练并完成训练后 LoRA tensor 热更新。

本次验证证明：

- Megatron TP4 / EP4 与 SGLang-Omni Thinker TP4 可以 colocate
  在同一组 GPU 0–3；
- Relax 会把 6.54 秒 FLEURS 音频切成 7 个 960ms chunk；
- Omni rollout 会逐轮累积 structured messages 和
  `metadata.audios`；
- Thinker `policy` LoRA 在全部 TP4 rank 生效；
- 多轮音频产生的 token、loss mask、rollout logprob 和音频训练特征
  能进入 Megatron；
- Megatron optimizer step 成功；
- 训练后的 192 个 LoRA tensor 能在全部 TP4 rank 执行
  flush → unload → `load_lora_adapter_from_tensors`。

这不是长期收敛或语音输出质量验证。本阶段仍使用 text output 和原有
FLEURS BLEU reward，只训练 Thinker LoRA。

## 最终 3-step / n=4 GPU 回归

在下述首次探索作业之后，又完成了一次正式的 3-step、每个 prompt
采样 4 条的流式回归：

- Modal app：`ap-jTbPhHTOwET8pjehZwCE3d`
- Ray job：`raysubmit_hApVkzEZeYMDkVrW`
- GPU：`4 × A100-80GB`
- 拓扑：Megatron TP4 / EP4 与 Omni Thinker TP4 colocate 在同一组
  GPU 0–3
- `num_rollout=3`
- `rollout_batch_size=1`
- `n_samples_per_prompt=4`
- `global_batch_size=4`
- `rollout_max_response_len=128`
- `rollout_temperature=0.8`
- `simul_chunk_ms=960`
- 音频时长：6.54 秒，共 7 个 chunk

最终结果：

```text
Job 'raysubmit_hApVkzEZeYMDkVrW' succeeded
REAL RELAX OMNI SMOKE PASS (num_rollout=3, streaming=True)
Stopped all 14 Ray processes.
[cleanup] Ray and SGLang-Omni stopped
```

### 推理和轨迹证据

- 3 个 rollout 共得到 12/12 条样本；
- 每条样本都完成 7 个流式 turn，状态均为
  `stop_reason=chunks_exhausted status=completed`；
- 共完成 `12 × 7 = 84` 个 `/generate` HTTP 200；
- 12 条输出均非空、可按 UTF-8/JSON 恢复，没有 replacement character
  或控制字符损坏；
- 每个 turn 的 TP4 prefill 都记录
  `lora_ids=['policy']`、`has_active_lora=True`。

输出确实包含“千里之外”“卫星”等目标语义，但随机样本之间质量差异明显：
部分翻译较完整，部分存在重复、音乐标记、英文或幻觉。这说明端到端推理是
正常工作的，也说明当前只跑 3 step 不能用于声称模型已经收敛。

### 三步训练指标

| step | raw reward | train loss | entropy | grad norm |
|---:|---:|---:|---:|---:|
| 0 | 0.1120084664 | -8.1956387e-08 | 0.8933291435 | 4.0776537627 |
| 1 | 0.1171396608 | -5.9604645e-08 | 0.9269951582 | 3.7085923665 |
| 2 | 0.1005691784 | 2.9802322e-08 | 0.9821163416 | 3.9447506948 |

三步的 `pg_clipfrac=0.0`、`ppo_kl=0.0`、学习率 `1e-5`，所有记录指标
均为有限值，三步梯度均非零。这里的 GRPO 组内归一化 reward/advantage
均值接近 0 是预期行为，不代表原始 BLEU reward 为 0。

### 三代 LoRA 的消费与更新证据

完整控制序列为：

```text
initial load
→ rollout 0 (28 generate)
→ train step 0
→ unload + load
→ rollout 1 (28 generate)
→ train step 1
→ unload + load
→ rollout 2 (28 generate)
→ train step 2
→ unload + final load
```

HTTP 计数精确为：

```text
load_lora_adapter_from_tensors = 4
unload_lora_adapter = 3
generate = 84
```

每次 load/unload 都在 Thinker rank 0–3 出现 ModelRunner
start/complete 日志。initial load、step 0 reload、step 1 reload 后面都存在
实际生成请求，证明三个被 rollout 使用的 adapter 版本都真正进入推理，
不是只测试 endpoint。step 2 后的最终 reload 也在全部 TP4 rank 成功，
证明最后一次训练结果可以继续交给后续 rollout。

Omni 加载后每 rank 的 Thinker 权重占 `14.32 GiB`，KV cache 的 K/V
各占 `8.17 GiB`，服务 ready 后每卡仍有约 `25 GiB` 可用。训练和多次
热更新期间没有 OOM 或 NCCL 错误。

Modal 本地客户端最后报告一次“等待最终 app logs 超时”，发生在 Ray job
succeeded、wrapper PASS、Modal app completed 之后；它不影响远端作业结果。
远端和本地清理均已完成，没有残留 GPU 任务。

## 首次 n=2 探索作业

- Modal app：`ap-54M8Q0tHaPr08RPX7S1DX9`
- Ray job：`raysubmit_79MHWquVLHrMwr1V`
- GPU：`4 × A100-80GB`
- 配置：Megatron TP4 / EP4 与 Omni Thinker TP4 colocate 在 GPU 0–3
- `num_rollout=1`
- `rollout_batch_size=1`
- `n_samples_per_prompt=2`
- `global_batch_size=2`
- `rollout_max_response_len=128`
- `simul_chunk_ms=960`
- 音频时长：6.54 秒
- chunk 数：7

作业结束后 Ray、SGLang-Omni 和 Modal task 均已停止，没有 GPU 残留。

## 成功证据

1. Relax 报告：

   ```text
   required GPUs=4, cluster GPUs=4, colocate=True
   ```

2. 四个 Megatron rank 均完成模型和 checkpoint 加载。每 rank
   `8,831,871,600` 参数，挂载 192 个 LoRA 参数张量，
   共 `2.556M` 个可训练元素。

3. Megatron Actor ready 后才启动外部 Omni，避免 Omni-first 的构建期
   OOM。Omni Thinker rank 0–3 分别使用 GPU 0–3。

4. 初始同步抽取 192 个 LoRA tensor。四个 Thinker rank 都完成：

   ```text
   LoRARef(lora_id=policy, lora_name=policy, lora_path=__tensor__)
   ```

   `/load_lora_adapter_from_tensors` 返回 HTTP 200，prefill 日志记录
   `lora_ids=['policy']` 和 `has_active_lora=True`。

5. 真实多轮 rollout 一共完成 9 个 `/generate` HTTP 200。两条随机轨迹的
   turn 数分别为 2 和 7：

   - 轨迹 A：`3(stop) + 125(length)`，第 2 轮耗尽总生成预算；
   - 轨迹 B：`10 + 27 + 9 + 10 + 31 + 10 + 31 = 128`，
     完成 7 次请求，因此第 7 次请求已经携带并消费全部 7 个音频 chunk。

6. Megatron 训练 step 0 成功：

   ```text
   train/loss=0.0
   train/ppo_kl=-2.1373125491663814e-10
   train/pg_clipfrac=0.0
   train/grad_norm=1.0849168220842416
   ```

   指标均为有限值，且梯度非零。

7. 训练后再次导出 192 个 LoRA tensor。全部 TP4 rank 先卸载旧
   `policy`，再从 `__tensor__` 成功重载；endpoint 返回 HTTP 200。

8. Ray job 最终报告 succeeded，Relax 正常输出：

   ```text
   Actor training completed step 0/1
   All training steps finished
   Main func successfully
   ```

## 为什么 Modal app 最终没有打印 PASS

当时 launcher 的第一版流式 evidence guard 要求两条随机轨迹都严格完成
7 次请求，即必须观察到 `2 × 7 = 14` 个 HTTP 200。

实际观察到 9 次：一条轨迹完整覆盖 7 chunks，另一条在第 2 轮随机生成
过长并以 `finish_reason=length` 结束。核心 Ray job、训练和 TP4 tensor
reload 都成功，但旧 guard 在 cleanup 后按预期抛出：

```text
Streaming smoke did not complete every expected chunk request;
expected=14, observed=9
```

这不是 OOM、TP fanout、LoRA 更新或训练失败。`length` 终止整条轨迹也是
原标准 SGLang simul rollout 与 Omni rollout 共同的既有语义；不能为了
测试凑满请求数而改成截断后继续。

也不应简单把预算从 128 放大到 512。轨迹 A 的 125-token runaway 很可能
变成 509-token runaway，最坏成本约放大四倍，仍不能保证 14 次请求。

## 本地待提交改动

### Relax

`examples/simul_s2tt/omni_rollout.py`

- 写入 `sample.metadata["rollout_turns"]`；
- 每条轨迹结束时打印可机器解析的：

  ```text
  [omni-simul-evidence] sample_index=... rollout_turns=...
  num_chunks=... stop_reason=... status=...
  ```
- 同时打印 JSON 转义、最多 256 字符的输出预览：

  ```text
  [omni-simul-output] sample_index=... response=...
  ```

`tests/engine/rollout/test_sglang_omni_rollout.py`

- 使用真实 `AudioChunkEnv` 和 6.54 秒 CPU waveform 验证 7 个
  960ms turn；
- 验证每轮 message 中 audio marker 与 `metadata.audios` 均按
  1..7 累积；
- 验证最后一个 tail chunk；
- 验证 Thinker `stage_params`；
- 验证双 token 流、loss mask、logprob 和 response length；
- 验证 7 份多模态训练特征的 pad/merge。

### 父仓库 Modal launcher

`modal_relax_omni_lora_smoke.py`

- 新增 `--streaming` 模式，成本保护只允许 1–3 个 rollout；
- 每个 prompt 使用 4 个采样、960ms chunk、128 总 response budget；
- 流式温度使用 0.8，降低随机 runaway 概率；
- 从 WAV 实际时长计算 expected chunk 数；
- 捕获 Omni stdout；
- 新 evidence guard 要求：
  - 每条轨迹至少 2 turn；
  - 至少有 `num_rollout` 条轨迹覆盖所有 7 chunks；
  - 只允许 `chunks_exhausted` 或明确的 `length`；
  - 拒绝 abort、failed 和请求计数不一致；
  - 每条轨迹必须有非空且无编码损坏的输出预览；
  - `load=steps+1`、`unload=steps`；
  - 每个实际被 rollout 使用的 load 与下一次 load 之间必须存在
    `/generate`，防止只测 endpoint 而没有消费新 adapter；
  - 分别报告 full-audio coverage、chunks-exhausted 和 truncated 比例；
  - 继续要求 active Thinker LoRA 和有限训练指标。

新 guard 保留标准 SGLang 的 `length` 终止语义，没有修改原 SGLang
rollout backend。

## CPU 回归

最新 Modal CPU app：`ap-Y2yrM4QCu6AHOTms7BWDX3`

```text
23 passed, 26 warnings in 0.34s
```

该 app 已 `stopped`、`tasks=0`，没有申请 GPU。

本地额外完成：

- 三个本次修改 Python 文件 `py_compile` 通过；
- evidence validator 正例和无 full-coverage 反例通过；
- 父仓库和 Relax 的 `git diff --check` 通过；
- 本机未安装 ruff，因此没有安装依赖或格式化整个仓库。

## 当前边界与下一步

最新 3-step / n=4 GPU 作业已经给出正式的 wrapper PASS，因此不需要为了
同一链路立即再烧一次 4×A100。后续仍应保持 4 卡 colocate，不拆成 4+2，
也不扩大到 5/6/8 卡。

尚未验证：

- 长时间训练稳定性；
- 更大数据集和更多 prompt 下的 reward/gradient 趋势及收敛；
- Omni Router 多 worker 的真实 GPU 广播；
- Talker/speech reward（不在当前范围）；
- 全量 base `/update_weights_from_tensor`（不在当前范围）。

截至写入本文档时，这批改动尚未 commit、尚未 push；sglang-omni 和标准
sglang 工作树干净，父仓库五个 `e2e_*.log` 仍保持未跟踪且未被修改。
