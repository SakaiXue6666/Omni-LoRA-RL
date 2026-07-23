# Relax 训练后 Tensor LoRA + Talker 语音闭环（2026-07-24）

## 结论

本轮已经把此前分开的两段能力接成一条可验证链路：

1. Relax 在 4 张 A100-80GB 上完成 1 个训练 step，每个 prompt 采样 4 条；
2. Megatron 导出的 192 个 LoRA tensor 经 `flattened_bucket` 热更新到 SGLang-Omni Thinker TP4；
3. 训练后先卸载旧 `policy`，再把新 `policy` 同步到全部 4 个 Thinker rank；
4. 最终请求使用 `stage_params.thinker.lora_name="policy"`；
5. Thinker 的 hidden states 继续交给 Talker/Code2Wav，得到有效 WAV。

训练和 tensor 同步在完整 4 卡作业中通过；训练后语音故障的修复又用一张
A100 做了针对性 GPU 复验并通过。没有重新跑一遍昂贵的 4 卡 Megatron 初始化。

## 4 卡完整训练作业

Modal app：`ap-Qo27fPoUHqtRxh1p4iLh2n`

拓扑：

- 4 × A100-80GB；
- Megatron TP4/EP4 与 SGLang-Omni Thinker TP4 colocate 在同一组 4 卡；
- Talker、Code2Wav、image/audio encoder 放在 GPU 0；
- `num_rollout=1`，每个 prompt 采样 4 条。

已确认的证据：

- 初始同步：192 个 adapter tensor 在 4 个 Thinker rank 全部加载成功；
- 4/4 rollout 请求均有 `lora_ids=['policy']`、`has_active_lora=True`；
- 4/4 推理 HTTP 200，均生成非空中文翻译；
- `rollout/raw_reward = 0.19961697794497013`；
- `train/loss = -1.1920928955078125e-07`；
- `train/grad_norm = 2.2268602674041342`；
- `train/ppo_kl = 0.0`，所有指标有限；
- 训练 step 0/1 完成；
- 训练后旧 LoRA 在 4 rank 全部 unload；
- 新 LoRA 在 4 rank 全部通过 `load_lora_adapter_from_tensors` reload，HTTP 200；
- 最终 speech 请求的 Thinker 路由仍为 `policy` 且 active。

## 暴露并修复的跨 stage IPC 问题

完整作业最后一次 Thinker → Talker 传递最初失败：

```text
module 'torch.multiprocessing.reductions' has no attribute
'_rebuild_cuda_tensor_original'
```

原因不是训练、LoRA 权重或显存。Thinker 反序列化 tensor LoRA 时调用了标准
SGLang 的 `monkey_patch_torch_reductions()`，因此随后 CUDA IPC pickle 使用了修改后的
rebuild 函数；独立的 Talker stage 进程没有初始化同一 patch，无法解开 Thinker 发来的
CUDA tensor。

最小修复位于：

- `sglang-omni/sglang_omni/pipeline/stage/runtime.py`
- `sglang-omni/tests/unit_test/pipeline/test_stage_streaming.py`

接收同卡 CUDA IPC chunk 时，在 `pickle.loads` 之前幂等调用标准 SGLang 的
`monkey_patch_torch_reductions()`。这样 Thinker、Talker 和后续同卡 stage 对 CUDA tensor
reducer/rebuilder 的定义一致。未修改 Relax，也未改变 tensor LoRA payload。

## 低成本单卡 GPU 定点复验

Modal app：`ap-c8RRkZAiJxOhTWxOzZCRxo`

入口：

```powershell
D:\Li_Lab\RL\.venv-modal\Scripts\modal.exe run `
  modal_omni_serve_lora_tensor.py::e2e_speech
```

流程：单张 A100-80GB 上启动 colocated Thinker + Talker + Code2Wav，执行随机 tensor
LoRA 第一次加载、推理、unload、第二次加载、推理，再发带 `policy` 的 text+audio 请求。

结果：

- 两轮 payload 各含 192 个 tensor，`load_lora_adapter_from_tensors` 均 HTTP 200；
- `base_vs_update_1_changed=True`；
- `update_1_vs_update_2_changed=True`；
- speech Thinker 路由：`lora_ids=['policy']`、`has_active_lora=True`；
- Thinker text：`Hello after tensor update.`；
- WAV：85,034 bytes，24,000 Hz，42,495 frames，1.771 秒；
- RMS：0.052715，非静音且数值有限；
- HTTP 200，最终 `PASS`；
- 不再出现 reduction/IPC 错误。

## CPU 与静态验证

- Relax 独立 Omni backend：`23 passed`；
- SGLang-Omni stage streaming + LoRA admin：`48 passed, 2 skipped`；
- 新 IPC 定点测试单独运行：`1 passed`；
- `modal_omni_serve_lora_tensor.py`、`modal_relax_omni_lora_smoke.py` 语法检查通过；
- 父仓库与 sglang-omni 的 `git diff --check` 通过。

## 当前未做事项

- 修复后没有再次重复完整 4 卡训练，原因是完整作业的训练、TP4 二次同步已经通过，
  单卡定点复验覆盖了唯一失败的 tensor-load 后 Thinker → Talker CUDA IPC 分支；
- 本轮改动与本文一起提交；具体 commit 以父仓库和 sglang-omni 历史为准；
- Relax 和标准 sglang 中已有的注释/标记修改是此前工作区内容，本轮未修改；
- 父仓库 `e2e_*.log` 未删除、未暂存。
