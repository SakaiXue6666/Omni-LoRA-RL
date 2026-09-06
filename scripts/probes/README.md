# 探针脚本

v1 → v2 迁移期间（2026-08-10 ~ 08-12）用来逐条验证假设的脚本。**跑训练不需要它们**，
它们的作用是给 `docs/migration-v2.md` 里的每条结论提供可复现的出处。

## 配对方式

大部分探针是一对文件：

- `mig_NN_*.py` —— 探针本体，在容器里执行的逻辑，不依赖 Modal
- `modal_probe_*.py` —— Modal 驱动，负责选机型、装环境、把本体送上去跑

想在自己的机器上复现，直接跑 `mig_NN_*.py` 即可；想复现当时的完整环境，跑对应的
`modal_probe_*.py`。

## 索引

| 探针 | 本体 | Modal 驱动 | 验了什么 | 结论 |
|---|---|---|---|---|
| 环境闸门 | `mig_00_env.py` | `modal_migrate.py` / `modal_env_gate_v2.py` | Megatron-Bridge 0.5.0 与 `megatron.core` 能否共存、`export_adapter_weights` 是否存在 | 通过，2026-07 的卡点已被上游解除 |
| 环境闸门 b | `mig_01_omni_bridge.py` | `modal_gate_omni.py` | `AutoBridge` 认不认 Qwen3-Omni | 认 |
| 一 | `mig_02_lora_scope.py` | `modal_probe_lora_scope.py` | 上游 `wrap_model_provider_with_lora` 会不会把 LoRA 挂到 audio/vision 塔 | 不会，塔是 HF 命名 |
| 二 | `mig_03_export_names.py` | `modal_probe_export_names.py` | `export_adapter_weights` 的命名与 v1 手写导出是否 parity | 一致 |
| 三 | `mig_04_peft_prefix.py` | `modal_probe_peft_prefix.py` | 导出的 adapter 目录能否被标准 PEFT 读回 | **不能** → Relax PR #262 |
| 四 | `mig_05_sglang_lora_scope.py` | `modal_probe_sglang_scope.py` | sglang 的 `_lora_pattern` 命中哪些真实模块名 | 只命中 thinker 文本主干 |
| 五 | `mig_06_serve_lora.py` | `modal_probe_serve_lora.py` | 1×A100 上把 adapter 热推给 sglang：生效、可逆、稳定 | 五项全过 |
| 六 | `mig_07_cpu_reduce.py` | `modal_probe_serve_lora.py --stage reduce` | CPU 张量过 `monkey_patch_torch_reductions` 的越界 | 复现 IndexError → sglang PR |
| 七 | `mig_08_train_side.py` | `modal_probe_train_side.py` | 训练侧冒烟：Bridge 建 Omni + LoRA 注入 + adapter 导出 | 一次通过 |
| 八 | `mig_09_tp_export.py` | `modal_probe_tp_export.py` | TP=2 下 adapter 导出 parity | 三种切分全部正确还原 |
| 九 | —（自带） | `modal_probe_transport.py` | adapter 传输三路对照，复现真机 ENOENT | 内联字节全过 → Relax PR |

## 上游 PR 的验证脚本

| 脚本 | 用途 |
|---|---|
| `modal_verify_pr1.py` | sgl-project 第一个 PR（`should_apply_lora` 门控）的分支验证 |
| `modal_verify_sglang_pr2.py` | sgl-project 第二个 PR（`patch_torch` CPU 张量守卫） |
| `modal_verify_pr3.py` | Relax 第三个 PR（adapter 传输改内联字节），pytest + 仓库自带 pre-commit |
| `modal_verify_relax_prs.py` | Relax 前两个 PR 分支的双向验证（打补丁全过 / 换回上游全挂） |
| `modal_run_lora_tests.py` | 在 v2 的 sglang fork 上跑 LoRA 单测 |
| `modal_lint_relax_prs.py` | 在容器里跑 `ruff format --check` + `ruff check`（本地连不上 pip 源） |
| `modal_precommit_relax_prs.py` | 在容器里跑 Relax 仓库自带的 pre-commit —— 专门用来复现 `docformatter` 那个把 CI 搞挂的 hook |
