# Omni-LoRA-RL

Reinforcement-learning training for Qwen3-Omni Thinker + LoRA. The task is **S2TT** (English
speech → Chinese text), the algorithm is GRPO, the reward is sentence-level BLEU, and inference
goes through sglang's LoRA adapter hot-loading — every training step pushes the updated adapter
straight from GPU memory into the inference engine, without touching disk.

**Best result: 100 full steps on 4×A100-80GB, BLEU from 0.287 to 0.487.**
The current code path has reproduced the first 40 steps (0.294 → 0.391). Complete per-step data
for all four experiments is in [`docs/results/experiments.md`](docs/results/experiments.md).

This repository is the **entry point / hub**; the actual code is pulled in as submodules pointing
at two forks.

## What is in here

| Directory | Origin | Branch | Role |
|---|---|---|---|
| `Relax/` | fork of `redai-infra/Relax` | `lora-omni-v2` | Training side (Megatron-Bridge + LoRA + rollout) |
| `sglang/` | fork of `sgl-project/sglang` `v0.5.12.post1` | `lora-omni-v2` | Inference side (LoRA serving for Omni) |
| `omni_s2tt/` | this repo | — | Training scripts, BLEU reward, data preparation, curve statistics |
| `scripts/`, `docs/` | this repo | — | Modal entry point and migration-era probes; documentation and experiment data |

The forks exist because five changes have not landed upstream yet (all filed as PRs, see "Current
status" below). Once they merge, the forks can collapse back to "upstream plus the two
Omni-specific bits".

**Hardware**: four 80GB cards (A100/H100). The model is a 30B MoE (3B active), configured
colocated — training and inference share the same cards, TP4 / EP4 / PP1. A single node is
enough; no multi-node setup needed. If memory is tight, start by adjusting
`SGLANG_MEM_FRACTION` (default 0.7), which decides how much the inference engine takes.

---

# What was tried, and what came out

The dataset throughout is **FLEURS** (`google/fleurs`, `en_us` audio paired with `cmn_hans_cn`
text by `id`), and the reward is sacreBLEU (Chinese tokenizer) / 100. The metric is
`rollout/raw_reward` per step, i.e. the mean BLEU over that batch of samples. **The criterion is
always a ten-step window mean, never a single step** — single-step noise is large; we have seen a
drop from 0.539 to 0.397.

| Experiment | Data | Steps | Result (first window → last window) |
|---|---|---|---|
| zh→en translation + BLEU (validating the reward design) | 256 hand-made long sentences | 10 | 0.463 → 0.523 |
| **en→zh S2TT (offline)** | FLEURS, 128 items | **100** | **0.287 → 0.487** |
| Simultaneous (960 ms fixed chunks, multi-turn) | FLEURS, 97 full clips | 20 | 0.155 → 0.265 |
| en→zh S2TT (current code path) | FLEURS, 128 items | 40 | 0.294 → 0.391 |

There was also one useful failure: math multiple-choice with a 0/1 reward gave 32/32 correct,
zero in-group variance, advantage = 0, and nothing to learn from. That is why every experiment
afterwards uses a continuous reward — the reasoning is in
[`docs/design/reward-design.md`](docs/design/reward-design.md).

## The main line: S2TT, 100 steps

Ten-step window means, rising monotonically:

| Range | BLEU | | Range | BLEU |
|---|---|---|---|---|
| 1–10 | 0.2868 | | 51–60 | 0.4137 |
| 11–20 | 0.3442 | | 61–70 | 0.4400 |
| 21–30 | 0.3448 | | 71–80 | 0.4822 |
| 31–40 | 0.3886 | | 81–90 | 0.4890 |
| 41–50 | 0.4020 | | 91–100 | 0.4873 |

Range [0.234 @step3, 0.610 @step95]. Supporting evidence: response length dropped from 28 tokens
to 20 (tighter translations) and log_probs rose. Per-step raw data in
`docs/results/s2tt-100step-curve.json`.

## Limits: what has not been verified

- **The current code path has only been run to 40 steps.** That 100-step curve was produced on
  the older implementation; whether it keeps climbing to 0.487 past step 40 has not been verified
  on the current code. The first 40 steps match segment by segment (all deltas within ±0.013).
- **Resuming has never been verified.** The most recent attempt meant to continue from
  `iter_0000004` but actually restarted from 0 — the previous run was killed early and
  `latest_checkpointed_iteration.txt` was never written. The LoRA resume path has never been
  investigated on its own.
- **The simultaneous-interpretation code has been ported but never run.** It is in
  `Relax/examples/simul_s2tt/` with the four necessary changes applied; known risks and the errors
  to expect are in [`docs/design/simul-port-notes.md`](docs/design/simul-port-notes.md) — read it
  before spending GPU time.
- **Convention**: `omni_s2tt/curve.py` computes from the per-sample rewards in `rollout_result`,
  while the numbers above come from `rollout/raw_reward` in the training log. The two should be
  equal, but they have never been cross-checked.

---

# Current status

> As of 2026-09-05

**Closed loop**: the S2TT training path runs end to end on the current code (2026-08-12, 40
steps). sglang's `should_apply_lora` gate, Omni's `_lora_pattern`, Relax's wildcard expansion and
PEFT prefix, and adapter transport were all implicitly validated by that run.

**In flight**: five upstream PRs, all open, no movement since mid-August.

| PR | Repo | Status |
|---|---|---|
| [#34428](https://github.com/sgl-project/sglang/pull/34428) Honor `should_apply_lora` | sglang | Open. **CI is blocked on the missing `run-ci` label; the tests have never actually executed.** Five code owners have not responded |
| [#34595](https://github.com/sgl-project/sglang/pull/34595) CPU-tensor reduce guard | sglang | Open, blocked on the same `run-ci` label |
| [#261](https://github.com/redai-infra/Relax/pull/261) Wildcard target-module expansion | Relax | Open, CI green, awaiting code-owner approval |
| [#262](https://github.com/redai-infra/Relax/pull/262) Export adapters in PEFT's key layout | Relax | Open, CI green, awaiting code-owner approval |
| [#265](https://github.com/redai-infra/Relax/pull/265) Adapter transport switched to inlined bytes | Relax | Open, awaiting code-owner approval |

The PR bodies are archived in `docs/upstream-prs/`. What blocks the two sglang PRs is not a
technical problem — a maintainer just needs to add a label, which makes it the easiest thing to
push on.

**Next steps** (in priority order):

1. Get the `run-ci` label onto the two sglang PRs
2. Investigate why LoRA resume does not take effect
3. Get simultaneous interpretation running (the code is in place; see the risk list in
   `docs/design/simul-port-notes.md`)
4. Run the current code path for a full 100 steps and confirm 0.487 reproduces

---

# How to run it

Two paths, with **different verification status — be clear which one you are on**:

| | Status |
|---|---|
| **A. Modal** | ✅ **Verified.** Every experiment above was produced on this path |
| **B. Your own server / Docker** | ⚠️ **Derived from the source, never actually run.** It corresponds one-to-one with A, but nobody has walked it through |

For path B: whoever gets it working first, please come back and fix this document. Filing a PR
with the discrepancies is far more useful than leaving an unverified manual in place.

## A. Running on Modal (verified)

```bash
modal run scripts/modal_train_s2tt.py::check                      # check data and weights first (CPU, tens of seconds)
modal run scripts/modal_train_s2tt.py --num-rollout 40 --detach   # start training, spawned so it detaches from the local process
modal run scripts/modal_train_s2tt.py::result --call-id <ID>      # fetch the result
```

**Do not skip `--detach`.** Using `remote()` binds the training run's lifetime to the local modal
process, so the app gets reaped the moment the local side drops — that is exactly how a run got
killed at step 5, leaving nothing but an `iter_0000004`.

The image, PYTHONPATH and environment variables in `scripts/modal_train_s2tt.py` correspond
one-to-one with every section of B below, so you can read them side by side.

## B. Running on your own server / Docker (never actually run)

### 1. Get the code

```bash
git clone --recursive https://github.com/SakaiXue6666/Omni-LoRA-RL.git
cd Omni-LoRA-RL

# Already cloned but forgot --recursive:
git submodule update --init --recursive
```

Check that both submodules landed on the expected commits (a mismatch means init was not clean):

```bash
git submodule status
# +9e94202... Relax  (heads/lora-omni-v2)
#  02044692... sglang (v0.5.12.post1-5-g02044692cc)
```

### 2. Start the container

Use the official Relax image, pinned by digest — `:latest` drifts, and that is exactly what made
the older implementation irreproducible. The image already contains Megatron-LM,
Megatron-Bridge, flashinfer and transformer-engine; you do not need to install them.

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

Do not omit `--shm-size`. The default 64MB makes the shared-memory paths in NCCL and Ray fail at
random.

One package to add inside the container (the BLEU reward needs its Chinese tokenizer):

```bash
pip install --no-cache-dir sacrebleu
```

#### PYTHONPATH

This is the step most likely to go wrong. All three parts are required:

```bash
cd /workspace/Omni-LoRA-RL
export PYTHONPATH=$PWD:$PWD/sglang/python:$PYTHONPATH
```

- `$PWD`: so that `--custom-rm-path omni_s2tt.bleu_rm.compute_bleu_reward` can import the reward
  module
- `$PWD/sglang/python`: **shadows the sglang preinstalled in the image**, otherwise you run the
  copy without Omni LoRA support
- the trailing `$PYTHONPATH`: the image's original `/root/Megatron-LM:/pkg:/root` must be
  preserved — `megatron` and `megatron.bridge` live there. The first run dropped it, and once the
  Ray job started it died on `from megatron.core import mpu` with ModuleNotFoundError; all ten
  retries burned on the same line

Before burning GPU time, self-check. It takes seconds:

```bash
python3 -c "import megatron.core, relax, sglang, omni_s2tt.bleu_rm as b; \
print('megatron', megatron.core.__file__); print('relax', relax.__file__); \
print('sglang ', sglang.__file__); print('reward ', b.__file__)"
```

The `sglang` line must point at `/workspace/Omni-LoRA-RL/sglang/python/...`. Anywhere else means
the image's copy shadowed it, and LoRA will get attached to the audio/vision towers.

### 3. Prepare the weights

```bash
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /models/qwen3-omni
```

Two files must exist. Missing either does not fail immediately — it fails in a strange way:

```bash
# 1. tokenizer.json: sgl-router is written in Rust and only accepts the fast format when
#    registering a tokenizer
python3 -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('/models/qwen3-omni', trust_remote_code=True, use_fast=True).save_pretrained('/models/qwen3-omni')"

# 2. chat_template.json: Qwen3-Omni keeps its template inside the processor, and a bare
#    AutoTokenizer cannot read it under transformers 5.x — you get an empty template, so the
#    model emits <|im_end|> straight away and every rollout comes back empty
test -f /models/qwen3-omni/chat_template.json && echo OK
```

### 4. Prepare the data

One JSON object per line, four fields:

```json
{
  "prompt": "<audio>\nPlease translate the English speech into Chinese. Only output the Chinese translation.",
  "audios": ["/data/s2tt/audio/fleurs_00000123_en.wav"],
  "label": {"ground_truth": "reference translation"},
  "metadata": {"src_lang": "en", "tgt_lang": "zh", "rm_type": "bleu", "src_text": "source text"}
}
```

The `<audio>` in `prompt` is a placeholder; `--multimodal-keys '{"audio": "audios"}'` fills it
from the paths in `audios`. Use absolute paths, mono 16 kHz wav.

The data behind that curve (FLEURS en→zh, first 128 items of validation):

```bash
pip install "datasets>=2.19,<3" "numpy<2" soundfile librosa "huggingface_hub<0.26"
python3 omni_s2tt/prep_fleurs_s2tt.py --out-dir /data/s2tt --limit 128
```

128 items at `rollout-batch 8` is one epoch every 16 steps, so 40 steps is about 2.5 epochs.

### 5. Run training

```bash
cd /workspace/Omni-LoRA-RL

export HF_CKPT=/models/qwen3-omni
export DATA=/data/s2tt/train_s2tt.jsonl
export NUM_ROLLOUT=40
export NUM_GPUS=4
export SAVE_DIR=/data/s2tt/ckpt/s2tt_run1     # leave empty to skip checkpointing
export SAVE_INTERVAL=5

# feed the chat template in explicitly (see step 3)
export CHAT_TEMPLATE_KWARGS=$(python3 -c "
import json; t = json.load(open('$HF_CKPT/chat_template.json'))['chat_template']
print(json.dumps({'chat_template': t}, ensure_ascii=False))")

bash omni_s2tt/run-qwen3-omni-lora-s2tt-4gpu.sh 2>&1 | tee train.log
```

You do not need to start Ray yourself. The script sources `Relax/scripts/entrypoint/local.sh`,
which cleans up leftover processes, brings up a single-node Ray head, probes NVLink and sets
`RUNTIME_ENV_JSON`. **If you already have a Ray cluster**, just set `RAY_ADDRESS` — once detected,
it hands off to `ray-job.sh` instead of starting another head.

> **On a shared server, be careful**: the cleanup section of `local.sh` is `pkill -9 python` plus
> `pkill -9 ray`, and it does not care whose processes they are. Inside a container this is fine
> (PID namespace isolation); running it directly on the host will take your colleagues' jobs down
> with it. On bare metal, start Ray yourself and skip that section — this is the path
> `scripts/modal_train_s2tt.py` takes:
>
> ```bash
> export RELAX_ENTRYPOINT_MODE=local        # tell the script not to source local.sh
> export RAY_ADDRESS=http://127.0.0.1:8265
> export RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$PYTHONPATH\", \
>   \"PYTHONUNBUFFERED\": \"1\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \
>   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\"}}"
> ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 \
>   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
> ```
>
> On this path you must supply `RUNTIME_ENV_JSON` yourself: the Ray job is a separate process, and
> without passing PYTHONPATH through you are back to `ModuleNotFoundError: megatron`.

To run in the background, wrap it in `nohup ... &` or tmux. Mind `set -o pipefail` on that pipe
(the script already has it): without it the exit code is `tee`'s, so a crashed training run
reports success.

### 6. Records and results

Running on a server means there is no Modal dashboard to fall back on, so first, where everything
lands. **As long as `SAVE_DIR` is set, the first three are automatic** — no extra flags needed.

#### Per-sample rewards (the one most worth collecting)

`$SAVE_DIR/rollout_result/train/<step>.jsonl`, one file per step, one sample per line:

```
rollout_id / sample_index / group_index / prompt / response / reward / label
prompt_length / response_length / total_length / status
```

The BLEU curve is just the mean of this data; what you need for debugging is the raw responses —
an empty translation, one carrying `<|im_end|>`, or all eight in a group being identical (which
leaves GRPO with no gradient) all have to be found here. If `SAVE_DIR` is unset, passing
`--rollout-result-dir <dir>` explicitly works too.

Computing the curve:

```bash
python3 omni_s2tt/curve.py $SAVE_DIR/rollout_result/train --csv curve.csv
```

It prints per-step means with a histogram bar and computes the first/last window means. Reference
values are in the "What was tried" section above.

#### TensorBoard

On by default (`--use-tensorboard` defaults to true). The output directory follows a priority
order: the `TENSORBOARD_DIR` environment variable > `$SAVE_DIR/tensorboard_log` >
`tensorboard_log/<project>/<experiment>` (a relative path, which follows the Ray job's working
directory and is hard to find). Pin an absolute path explicitly before you start:

```bash
export TENSORBOARD_DIR=/data/s2tt/tb/run1        # must be exported before ray start
tensorboard --logdir /data/s2tt/tb --host 0.0.0.0 --port 6006
```

The curves are under `rollout/raw_reward` (that step's mean BLEU), `rollout/rewards`,
`response_len/*` and `perf/*`.

#### Text logs

Two places, with different purposes:

- `log/qwen3-omni-lora-s2tt-<timestamp>.log` — what the script itself tees, i.e. the Ray driver's
  output. The training main thread and the per-step metrics lines are here
- `/tmp/ray/session_latest/logs/` — the Ray workers' logs. **This is where to look when training
  crashes**: the driver side usually has nothing but "actor died", while the real traceback is in
  the workers' `python-core-worker-*.log` and `worker-*.err`

On a server, remember to copy both somewhere with long-term storage: `/tmp` is gone after a
reboot.

#### wandb (optional)

Lab machines often have no outbound network, so use offline mode and `wandb sync` later:

```bash
export WANDB_API_KEY=...          # only needed when online
# add to WANDB_ARGS in the training script: --use-wandb --wandb-mode offline --wandb-dir /data/s2tt/wandb
```

#### Checkpoints

`$SAVE_DIR/iter_XXXXXXX`, including optimizer state; point `--load` at the same directory to
resume. The default `--max-actor-ckpt-to-keep 1` keeps only the newest one; raise `MAX_CKPT_KEEP`
to keep more. Note that the resume path has never been verified — see "Limits" above.

## What to tune, and where

Every hyperparameter goes through an environment variable; you do not need to edit the script:

| Variable | Default | Notes |
|---|---|---|
| `LORA_RANK` / `LORA_ALPHA` | 16 / 32 | Every recorded curve used this configuration; change it and you can no longer compare against them |
| `LR` | 1e-4 | Constant decay |
| `ROLLOUT_TEMPERATURE` | 1.1 | Slightly raised so translations within a group vary, which is what gives GRPO a gradient. **Read `docs/design/reward-design.md` before lowering it** |
| `ROLLOUT_BATCH` / `N_SAMPLES` | 8 / 8 | 64 sequences per step |
| `GLOBAL_BATCH` | 64 | |
| `SGLANG_MEM_FRACTION` | 0.7 | Fraction of GPU memory the inference engine takes |
| `PROJECT_NAME` | Relax/v2/omni-lora-s2tt | TensorBoard project name |

Changing dataset means just changing `DATA`, with fields matching section 4. Changing the reward
means pointing `--custom-rm-path` at your own function (signature in `omni_s2tt/bleu_rm.py`).
Neither requires touching Relax's code.

## Finding what we changed: the `[YULIN-MOD]` markers

`Relax/` and `sglang/` are forks of large upstream projects. Our own changes are a tiny fraction
of those trees, and every one of them is bracketed by a pair of markers:

```python
# ===== [YULIN-MOD] START: keep audio_seqlens on CPU so multi-audio sequences work =====
...our code...
# ===== [YULIN-MOD] END =====
```

The `START` line always says *what the change is for*, in one line. Some older blocks carry a 🚨
before the `=====`; that is decoration only, so match on the bracketed name:

```bash
grep -rn "\[YULIN-MOD\] START" Relax sglang
```

**That grep is the complete index of our delta against upstream.** If you want to know "what did
we actually change, and why", this is the answer — not the git history, which also contains the
44-file vendor patch Relax applies to sglang and everything upstream did on its own.

There are currently 12 marked blocks in 11 files:

| Area | What is changed |
|---|---|
| `Relax/examples/simul_s2tt/` (5 files) | The whole simultaneous-interpretation rollout: fixed-chunk env, multi-turn generate, config, self-test |
| `Relax/.../weight_update/update_weight_from_tensor.py` | Inline the adapter bytes instead of passing a shared-memory reference (filed as Relax #265) |
| `Relax/.../modeling_qwen3_omni/utils.py` | Keep `audio_seqlens` on CPU so sequences with 2+ audio segments do not hit a device mismatch |
| `sglang/.../lora/lora_manager.py` | Honor the per-model `should_apply_lora` gate (filed as sglang #34428) |
| `sglang/.../managers/tp_worker.py` | Install the torch reducers before deserializing |
| `sglang/.../models/qwen3_omni_moe.py` (2 blocks) | Declare LoRA support on the thinker text body; pad mel frames before batching |
| `sglang/.../utils/patch_torch.py` | Leave CPU tensors alone in the reducer patch (filed as sglang #34595) |

Several of these are upstream bugs rather than our adaptation, which is why they are also open
PRs — see "Current status". As those merge, the corresponding blocks get deleted and the fork
shrinks. The ones that will stay indefinitely are the Omni-specific pieces: upstream sglang has
no LoRA support for Qwen3-Omni at all.

**When you change something in either fork, bracket it the same way.** An unmarked change is
invisible to that grep, which quietly makes the index wrong. It is easy to forget when the change
is small: the `audio_seqlens` fix above is two lines and shipped without markers at first, which
had to be corrected afterwards.

## Changing Relax / sglang code

`Relax/` and `sglang/` are submodules pointing at two forks. **Changing them is not like changing
a file in this repo — it takes three steps.** Skipping the third is the most common accident:
other people pull and still get the old code, with no error.

```bash
# 1. Edit and commit inside the submodule
cd Relax                      # or sglang
# ...make your changes...
git add -A && git commit -m "fix(lora): ..."

# 2. Push to your fork
git push mine lora-omni-v2    # same for sglang; the branch name is also lora-omni-v2

# 3. Back in this repo, move the pointer to the new commit (the step people forget)
cd ..
git add Relax                 # or sglang
git commit -m "chore: bump Relax to <short sha>"
git push
```

Whether you did step 3 is easy to check:

```bash
git submodule status
# +265723e... Relax   ← a leading + means the pointer and the actual checkout disagree, i.e. step 3 was skipped
#  0204469... sglang  ← a leading space is what you want
```

Pulling other people's updates:

```bash
git pull && git submodule update --init --recursive
```

**Never rebase or force-push a submodule branch this repo already points at** — the pointer then
refers to a commit that no longer exists, and other people's `clone --recursive` fails outright
with a cryptic message.

If a change is a bug in upstream itself (rather than in our adaptation), file a PR upstream while
you are at it; once merged, the fork carries one fewer thing. The ones already filed are listed
under "Current status" above, and `docs/upstream-prs/` shows how the bodies were written.

## Pitfalls

Ordered by how often they come up. All of them actually happened on this pipeline:

1. **`ModuleNotFoundError: No module named 'megatron'`** — `PYTHONPATH` overwrote the image's own,
   see B step 2. The error appears inside the Ray job, not the main process.
2. **Rollouts come back empty / full of `<|im_end|>`** — the chat template was not fed in.
3. **`RuntimeError: unable to open shared memory object </torch_...>`, and only one TP rank dies**
   — adapter transport went through a shared-memory reference. The `Relax` submodule must be on
   `lora-omni-v2` (including `19aea461`), the commit that switched the adapter to inlined bytes.
   The reasoning is in probe 9 of `docs/migration-v2.md`.
4. **sgl-router will not start and complains about the tokenizer** — `tokenizer.json` is missing,
   see B step 3.
5. **`[bleu_rm] special token found in response` in the log** — the text sglang returns carries
   things like `<|im_end|>`, and concatenating them into the response pushes BLEU down to roughly
   40% of its true value. The reward module **only reports, it does not change the score**,
   because every recorded curve was produced under the same non-stripping conditions and changing
   it would make them incomparable. A few stray occurrences are fine; in bulk it means something
   is wrong on the rollout side — go look at the rollout, do not tune the reward.
6. **NCCL hangs at initialization** — `--shm-size` too small, or `NCCL_SOCKET_IFNAME` needs to be
   specified on a multi-NIC machine.

---

# Going deeper

| Document | Contents |
|---|---|
| [`docs/results/experiments.md`](docs/results/experiments.md) | Full record of all four experiments: configuration, per-step curves, conclusions, and why the failed one failed |
| [`docs/design/reward-design.md`](docs/design/reward-design.md) | Why the reward is BLEU, why the temperature is 1.1, how in-group variance is created |
| [`docs/design/simul-port-notes.md`](docs/design/simul-port-notes.md) | Porting simultaneous interpretation: the four changes, six known risks and what each one looks like when it fails |
| [`docs/migration-v2.md`](docs/migration-v2.md) | How the current implementation came about: conclusions from nine probes, and the motivation and evidence behind five upstream PRs |
| [`scripts/probes/`](scripts/probes/) | The reproducible script behind every one of those conclusions, with an index |
| `docs/design/my_plan.md` | Early planning and parameter snapshots |
| `docs/results/*.json` | Per-step raw data for all four curves, with `_meta` describing the conventions |

The earlier implementation is frozen on this repository's `v1` branch (tag `v1-frozen`),
read-only, together with the image digest of the time and three archived patches.
