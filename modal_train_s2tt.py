"""v2 的端到端训练：4×A100 上跑 Qwen3-Omni Thinker LoRA 的 S2TT（英语音频→中文）。

对照的是 v1 那次 100 步实验（`omni-lora-rl/modal_relax_smoke.py::learn` task="s2tt"）。
超参逐项照抄，因为这次跑 40 步的唯一目的是回答"迁到 v2 之后还学不学得动"，
reward 口径或超参动一点，曲线就没法比。v1 的参照值：前 10 步 BLEU 均值约 0.29，
31-40 步约 0.41。判据用窗口均值而不是单步——v1 自己的曲线单步能从 0.539 掉到 0.397。

相对 v1 runner 的差异，都是因为 v2 的环境更正规了：

  * 镜像：v1 是 slime 镜像 + 手工补 Relax 依赖 + 强装 redai fork 的 megatron-bridge；
    v2 直接用 Relax 官方镜像（按 digest 钉死），依赖都是现成的。
  * sglang：v1 往旧镜像里注入新 sglang，导致内核版本对不上，要
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK + triton backend + 禁 cuda graph 一路绕。
    v2 的 fork 就是镜像里那个版本（0.5.12.post1 + Relax 的 vendor patch），不用绕。
  * reward：v1 改了 Relax 源码加 rm_type=bleu；v2 走 --custom-rm-path，不动框架。

用法（Windows 控制台先切 UTF-8）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_train_s2tt.py::check          # 先查数据卷（CPU，几十秒）
    modal run modal_train_s2tt.py --num-rollout 40 --detach
"""

from __future__ import annotations

import os
import pathlib

import modal


HERE = pathlib.Path(__file__).resolve().parent
LOCAL_PKG = HERE / "omni_s2tt"
REMOTE_PKG = "/root/omni_s2tt"
TRAIN_SCRIPT = f"{REMOTE_PKG}/run-qwen3-omni-lora-s2tt-4gpu.sh"

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

# Relax 与 sglang 都钉死在 v2 submodule 的同一个提交上，跟本地工作区保持一致。
# 用我们的 fork：分支在上游 9a5674af 之上多一个提交，把 adapter 推送从共享内存
# 改成内联字节（见 modal_probe_transport.py 的三路对照）。
RELAX_REPO = os.environ.get("RELAX_V2_REPO", "https://github.com/SakaiXue6666/Relax.git")
RELAX_REF = os.environ.get("RELAX_V2_REF", "lora-omni-v2")
RELAX_SRC = "/root/RelaxSrc"

SGLANG_FORK = "https://github.com/SakaiXue6666/sglang.git"
SGLANG_REF = os.environ.get("SGLANG_V2_REF", "lora-omni-v2")
SGLANG_SRC = "/root/sglang_fork"

MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_MOUNT = "/models"
OMNI_CKPT = "/models/qwen3-omni"

S2TT_DIR = "/s2tt"
S2TT_JSONL = f"{S2TT_DIR}/train_s2tt.jsonl"

# fork 的 sglang 顶在最前面覆盖镜像预装的那份，Relax 源码紧随其后（镜像里没装 relax）。
# 后半段是镜像原本的 PYTHONPATH（'/root/Megatron-LM/:/pkg/:/root/'），必须原样带上：
# megatron 与 megatron.bridge 都在 /root/Megatron-LM。第一次跑漏了它，Ray job 起来后
# 在 `from megatron.core import mpu` 直接 ModuleNotFoundError，十次重试全废在这上面。
MEGATRON_SRC = "/root/Megatron-LM"
PYTHONPATH = f"{SGLANG_SRC}/python:{RELAX_SRC}:{MEGATRON_SRC}:/pkg:/root:/sgl-workspace/sglang/python"

image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .run_commands(
        f"git clone --filter=blob:none --branch {RELAX_REF} {RELAX_REPO} {RELAX_SRC}",
        f"cd {RELAX_SRC} && git log -1 --format='Relax %H %s'",
        f"git clone --filter=blob:none --branch {SGLANG_REF} {SGLANG_FORK} {SGLANG_SRC}",
        f"cd {SGLANG_SRC} && git log -1 --format='sglang %H %s'",
        # BLEU reward 用 sacrebleu 的 zh tokenizer，与 v1 一致
        "pip install --no-cache-dir sacrebleu",
    )
    .env({"PYTHONPATH": PYTHONPATH, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(LOCAL_PKG.as_posix(), REMOTE_PKG, copy=True, ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App("v2-omni-lora-s2tt")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
s2tt_volume = modal.Volume.from_name("s2tt-data", create_if_missing=True)


def _chat_template_kwargs() -> str:
    """Qwen3-Omni 的 chat template 在 processor 的 chat_template.json 里，裸
    AutoTokenizer 在 transformers 5.x 下读不到，会拿到空模板。显式喂进去。

    v1 踩过的坑：不套模板的话模型直接吐 <|im_end|>，rollout 全是空输出。
    """
    import json

    ct_path = os.path.join(OMNI_CKPT, "chat_template.json")
    template = None
    if os.path.exists(ct_path):
        try:
            with open(ct_path, encoding="utf-8") as f:
                obj = json.load(f)
            template = obj.get("chat_template") if isinstance(obj, dict) else None
        except Exception as e:  # noqa: BLE001
            print(f"[chat_template] 读 {ct_path} 失败：{e}", flush=True)
    if not template:
        raise RuntimeError(
            f"{ct_path} 里没有 chat_template。v1 在这里退回过内置 ChatML 模板，"
            "但那会让 v2 的 prompt 与 v1 不完全一致，BLEU 就不可比了 —— 先查权重卷。"
        )
    return json.dumps({"chat_template": template}, ensure_ascii=False)


def _parse_reward_curve(log_path: str) -> list:
    """从训练日志抽每步 reward。

    形如 ``rollout 7: {'rollout/raw_reward': 0.41, ..., 'rollout/rewards': 0.41, ...}``。
    raw_reward 是这一步 batch 的 BLEU 均值，也就是要和 v1 比的那条曲线。
    """
    import re

    step_pat = re.compile(r"rollout (\d+):")
    raw_pat = re.compile(r"'rollout/raw_reward':\s*([-\d.eE]+)")
    rew_pat = re.compile(r"'rollout/rewards':\s*([-\d.eE]+)")

    seen: dict[int, tuple] = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            raw_m = raw_pat.search(line)
            if not raw_m:
                continue
            step_m = step_pat.search(line)
            if not step_m:
                continue
            rew_m = rew_pat.search(line)
            seen[int(step_m.group(1))] = (
                float(raw_m.group(1)),
                float(rew_m.group(1)) if rew_m else float("nan"),
            )
    return [(s, seen[s][0], seen[s][1]) for s in sorted(seen)]


@app.function(image=image, cpu=2.0, timeout=15 * 60, volumes={S2TT_DIR: s2tt_volume, MODEL_MOUNT: model_volume})
def check() -> None:
    """烧 4 张 A100 之前先确认数据和权重都在（CPU，几十秒）。"""
    import json
    import random

    print("========== 数据卷 ==========", flush=True)
    assert os.path.exists(S2TT_JSONL), f"没有 {S2TT_JSONL}，v1 的 prep_s2tt 应该已经把它写进 s2tt-data 卷"

    records = []
    with open(S2TT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  样本数: {len(records)}", flush=True)

    missing = [r for r in records if not os.path.isfile((r.get("audios") or [""])[0])]
    print(f"  音频缺失: {len(missing)} 条", flush=True)
    assert not missing, f"有 {len(missing)} 条样本的 wav 找不到，先补数据"

    r = random.choice(records)
    print(f"  样例 prompt : {r['prompt'][:80]!r}", flush=True)
    print(f"  样例 audios : {r['audios']}", flush=True)
    print(f"  样例 label  : {r['label']}", flush=True)
    print(f"  样例 meta   : {r.get('metadata')}", flush=True)

    print("\n========== reward 自测 ==========", flush=True)
    from omni_s2tt.bleu_rm import get_bleu_reward

    ref = r["label"]["ground_truth"]
    print(f"  完全一致 -> {get_bleu_reward(ref, r['label'], {'tgt_lang': 'zh'}):.4f}", flush=True)
    print(f"  截断一半 -> {get_bleu_reward(ref[: len(ref) // 2], r['label'], {'tgt_lang': 'zh'}):.4f}", flush=True)
    print(f"  完全无关 -> {get_bleu_reward('今天天气不错', r['label'], {'tgt_lang': 'zh'}):.4f}", flush=True)

    print("\n========== 权重卷 ==========", flush=True)
    for name in ("config.json", "chat_template.json", "tokenizer.json"):
        p = os.path.join(OMNI_CKPT, name)
        print(f"  {'有' if os.path.exists(p) else '缺'}  {name}", flush=True)

    print("\n========== 源码 ==========", flush=True)
    import subprocess

    for label, path in (("Relax", RELAX_SRC), ("sglang", SGLANG_SRC)):
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H %s"], cwd=path, capture_output=True, text=True, check=False
        )
        print(f"  {label}: {out.stdout.strip()}", flush=True)

    print("\n========== import 自检 ==========", flush=True)
    import megatron.core
    import relax
    import sglang

    print(f"  megatron {megatron.core.__file__}", flush=True)
    print(f"  relax    {relax.__file__}", flush=True)
    print(f"  sglang   {sglang.__version__} 来自 {os.path.dirname(sglang.__file__)}", flush=True)

    print("\n[check] 通过", flush=True)


@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={MODEL_MOUNT: model_volume, S2TT_DIR: s2tt_volume},
    timeout=240 * 60,
    # Modal 的 GPU 函数都可能被抢占；靠卷上的 ckpt + 自动重试续跑。
    retries=modal.Retries(max_retries=10, initial_delay=5.0),
)
def train(num_rollout: int = 40, tag: str = "v2", save: bool = True) -> None:
    """4 卡 colocate 跑 GRPO，判据是后 10 步的 BLEU 均值高于前 10 步。"""
    import json
    import subprocess

    assert os.path.exists(os.path.join(OMNI_CKPT, "config.json")), f"权重卷里没有 {OMNI_CKPT}/config.json"
    assert os.path.exists(S2TT_JSONL), f"没有 {S2TT_JSONL}，先跑 check"

    # 坑14：sgl-router(Rust) 注册 tokenizer 时要 fast 格式的 tokenizer.json。
    tok_json = os.path.join(OMNI_CKPT, "tokenizer.json")
    if not os.path.exists(tok_json):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(OMNI_CKPT, trust_remote_code=True, use_fast=True)
        tok.save_pretrained(OMNI_CKPT)
        try:
            model_volume.commit()
        except Exception:  # noqa: BLE001
            pass
        print(f"[train] 已生成 {tok_json}", flush=True)

    save_dir = f"{S2TT_DIR}/ckpt/v2_s2tt_{tag}" if save else ""

    env = os.environ.copy()
    env.update(
        {
            "RELAX": RELAX_SRC,
            "MODEL_CONFIG_DIR": f"{RELAX_SRC}/scripts/models",
            "PYTHONPATH": PYTHONPATH,
            "RELAX_ENTRYPOINT_MODE": "local",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONUNBUFFERED": "1",
            "NUM_GPUS": "4",
            "RAY_ADDRESS": "http://127.0.0.1:8265",
            "HF_CKPT": OMNI_CKPT,
            "DATA": S2TT_JSONL,
            "NUM_ROLLOUT": str(num_rollout),
            "SAVE_DIR": save_dir,
            "SAVE_INTERVAL": "5",
            "CHAT_TEMPLATE_KWARGS": _chat_template_kwargs(),
            "RUNTIME_ENV_JSON": json.dumps(
                {
                    "env_vars": {
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": PYTHONPATH,
                        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
                    }
                }
            ),
        }
    )

    # 配置错误要在秒级暴露，而不是每次占着 4 张 A100 跑五分钟再死、还照着 retries 重试十次。
    print("\n========== import 自检 ==========", flush=True)
    preflight = subprocess.run(
        [
            "python3",
            "-c",
            "import megatron.core, relax, sglang, omni_s2tt.bleu_rm as b; "
            "print('megatron', megatron.core.__file__); print('relax', relax.__file__); "
            "print('sglang', sglang.__file__); print('bleu_rm', b.__file__)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    print(preflight.stdout.strip(), flush=True)
    if preflight.returncode != 0:
        print(preflight.stderr.strip(), flush=True)
        raise RuntimeError("import 自检没过，PYTHONPATH 或依赖有问题，不烧卡了")

    print("\n========== 起 ray head ==========", flush=True)
    # 被抢占后 Modal 可能复用同一容器重试，上一轮的 ray head 还占着 6379，
    # 直接 start 会端口冲突。先 stop 让启动幂等（没跑时是 no-op）。
    subprocess.run("ray stop --force", shell=True, env=env, check=False)
    subprocess.run(
        "ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 "
        "--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265",
        shell=True,
        env=env,
        check=True,
    )

    log_path = "/root/train_s2tt.log"
    print(f"\n========== 训练 {num_rollout} 步（ckpt={save_dir or '关'}）==========", flush=True)
    # 不套 pipefail 的话管道退出码取的是 tee 的，训练崩了也报 0 —— 第一次跑就被这个骗过。
    proc = subprocess.run(
        f"set -o pipefail; bash {TRAIN_SCRIPT} 2>&1 | tee {log_path}",
        shell=True,
        executable="/bin/bash",
        cwd=RELAX_SRC,
        env=env,
    )
    print(f"\n[exit] 训练 returncode={proc.returncode}", flush=True)

    if save_dir:
        try:
            s2tt_volume.commit()
            print(f"[train] 已 commit s2tt 卷（ckpt 在 {save_dir}）", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[train] commit 卷失败（忽略）：{e}", flush=True)

    print("\n========== BLEU 曲线 ==========", flush=True)
    curve = _parse_reward_curve(log_path)
    if not curve:
        print("[FAIL] 日志里没解析到 reward 行", flush=True)
        raise SystemExit(proc.returncode or 1)

    for step, raw, _ in curve:
        print(f"  step {step:3d}: BLEU={raw:.3f} |{'#' * int(round(raw * 40))}", flush=True)

    raws = [r for _, r, _ in curve]
    k = min(10, max(1, len(raws) // 3))
    first, last = sum(raws[:k]) / k, sum(raws[-k:]) / k
    print(f"\n  前 {k} 步均值 = {first:.3f}", flush=True)
    print(f"  后 {k} 步均值 = {last:.3f}", flush=True)
    print(f"  区间 [{min(raws):.3f}, {max(raws):.3f}]，共 {len(raws)} 步", flush=True)
    print("\n  v1 参照：前 10 步约 0.29，31-40 步约 0.41", flush=True)
    print(f"  判定：{'上升' if last > first else '没升'}", flush=True)

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


@app.local_entrypoint()
def main(num_rollout: int = 40, tag: str = "v2") -> None:
    """起训练并立刻返回。

    这里用 spawn 而不是 remote，是 v1 的做法（learn_audio 里那句 `learn.spawn(...)`，
    注释写着"与本地连接解耦"）。remote 会阻塞，训练的生命周期就绑在本地那个 modal
    进程上：第一次跑 40 步时本地一断，app 跟着在第 5 步被收掉，只留下 iter_0000004。
    spawn 把调用交给服务端就走人，本地关机也不影响。

    拿结果：modal run modal_train_s2tt.py::result --call-id <上面打印的 ID>
    """
    call = train.spawn(num_rollout=num_rollout, tag=tag)
    print(f"\n训练已 spawn，与本地连接无关了。call id: {call.object_id}")
    print(f"取结果: modal run modal_train_s2tt.py::result --call-id {call.object_id}")
    print("看日志: modal app logs <app-id>（上面那个链接里也能看）")


@app.local_entrypoint()
def result(call_id: str, timeout: int = 0) -> None:
    """按 call id 取回 spawn 出去的那次训练的返回值。

    timeout=0 表示只看一眼：还没跑完就直接说还在跑，不阻塞。
    """
    import modal

    call = modal.FunctionCall.from_id(call_id)
    try:
        print(call.get(timeout=timeout))
    except TimeoutError:
        print("还在跑。日志去 Modal 面板看，或者过会儿再取。")
