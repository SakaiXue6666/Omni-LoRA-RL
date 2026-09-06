# Probe scripts

Scripts used during the v1 → v2 migration (2026-08-10 ~ 08-12) to verify assumptions one at a
time. **You do not need them to run training.** Their purpose is to give every conclusion in
`docs/migration-v2.md` a reproducible source.

## How they pair up

Most probes are a pair of files:

- `mig_NN_*.py` — the probe itself, the logic that runs inside the container, no Modal dependency
- `modal_probe_*.py` — the Modal driver: picks the machine type, sets up the environment, ships
  the probe up there

To reproduce on your own machine, run `mig_NN_*.py` directly; to reproduce the exact environment
of the time, run the matching `modal_probe_*.py`.

## Index

| Probe | Body | Modal driver | What it checks | Conclusion |
|---|---|---|---|---|
| Env gate | `mig_00_env.py` | `modal_migrate.py` / `modal_env_gate_v2.py` | Whether Megatron-Bridge 0.5.0 can coexist with `megatron.core`, and whether `export_adapter_weights` exists | Passed; the 2026-07 blocker is gone upstream |
| Env gate b | `mig_01_omni_bridge.py` | `modal_gate_omni.py` | Whether `AutoBridge` recognizes Qwen3-Omni | It does |
| 1 | `mig_02_lora_scope.py` | `modal_probe_lora_scope.py` | Whether upstream `wrap_model_provider_with_lora` attaches LoRA to the audio/vision towers | It does not; the towers use HF naming |
| 2 | `mig_03_export_names.py` | `modal_probe_export_names.py` | Whether `export_adapter_weights` naming has parity with v1's hand-written export | Identical |
| 3 | `mig_04_peft_prefix.py` | `modal_probe_peft_prefix.py` | Whether the exported adapter directory can be loaded back by stock PEFT | **No** → Relax PR #262 |
| 4 | `mig_05_sglang_lora_scope.py` | `modal_probe_sglang_scope.py` | Which real module names sglang's `_lora_pattern` matches | Only the thinker text body |
| 5 | `mig_06_serve_lora.py` | `modal_probe_serve_lora.py` | Hot-pushing an adapter to sglang on 1×A100: takes effect, reversible, stable | All five checks pass |
| 6 | `mig_07_cpu_reduce.py` | `modal_probe_serve_lora.py --stage reduce` | Out-of-range access when CPU tensors go through `monkey_patch_torch_reductions` | Reproduced the IndexError → sglang PR |
| 7 | `mig_08_train_side.py` | `modal_probe_train_side.py` | Training-side smoke test: Bridge builds Omni + LoRA injection + adapter export | Passed first try |
| 8 | `mig_09_tp_export.py` | `modal_probe_tp_export.py` | Adapter export parity under TP=2 | All three sharding modes reconstruct correctly |
| 9 | — (self-contained) | `modal_probe_transport.py` | Three-way comparison of adapter transport; reproduces the ENOENT seen on real hardware | Inlined bytes pass → Relax PR |

## Verification scripts for the upstream PRs

| Script | Purpose |
|---|---|
| `modal_verify_pr1.py` | Branch verification for the first sgl-project PR (the `should_apply_lora` gate) |
| `modal_verify_sglang_pr2.py` | Second sgl-project PR (the CPU-tensor guard in `patch_torch`) |
| `modal_verify_pr3.py` | Third Relax PR (adapter transport switched to inlined bytes); pytest + the repo's own pre-commit |
| `modal_verify_relax_prs.py` | Two-way verification of the first two Relax PR branches (all pass with the patch / all fail reverted to upstream) |
| `modal_run_lora_tests.py` | Runs the LoRA unit tests against the v2 sglang fork |
| `modal_lint_relax_prs.py` | Runs `ruff format --check` + `ruff check` inside a container (the local machine cannot reach the pip index) |
| `modal_precommit_relax_prs.py` | Runs Relax's own pre-commit inside a container — specifically to reproduce the `docformatter` hook that broke CI |
