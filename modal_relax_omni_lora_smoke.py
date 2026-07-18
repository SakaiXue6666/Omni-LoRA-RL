"""Real Relax -> SGLang-Omni tensor-LoRA training smoke test on Modal.

This is intentionally separate from:

* ``modal_relax_smoke.py`` (the standard SGLang training baseline), and
* ``modal_omni_serve_lora_tensor.py`` (the simulated tensor-update E2E).

The test runs Megatron TP4 and an external SGLang-Omni thinker TP4 colocated
on the same four A100-80GB GPUs. One rollout
step proves initial adapter loading, rollout, training, and the post-train
unload/reload. Use two steps to additionally consume the trained adapter in a
subsequent rollout.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import modal

from modal_relax_smoke import (
    MEGATRON_REMOTE,
    MODEL_DIR,
    PYTHONPATH as RELAX_PYTHONPATH,
    RELAX_REMOTE,
    S2TT_DIR,
    S2TT_JSONL,
    image as relax_image,
    model_volume,
    s2tt_volume,
)


APP_NAME = "relax-sglang-omni-lora-smoke"
HERE = Path(__file__).resolve().parent
OMNI_LOCAL = HERE / "sglang-omni"
OMNI_REMOTE = "/root/sglang-omni"
SGLANG_REMOTE = "/root/sglang_src/python"
OMNI_VENV = "/opt/sglang-omni-venv"
OMNI_PYTHON = f"{OMNI_VENV}/bin/python"
OMNI_PORT = 30000
TRAIN_GPU_COUNT = 4
OMNI_TP_SIZE = 4
TOTAL_GPU_COUNT = 4

OMNI_RUNTIME_DEPS = (
    "typer pyzmq msgpack pydantic pyyaml xxhash httpx fastapi uvicorn "
    "pybase64 requests pillow accelerate safetensors soundfile librosa av "
    "qwen-vl-utils==0.0.11 transformers==5.6.0"
)

image = (
    relax_image.run_commands(
        f"python -m venv --system-site-packages {OMNI_VENV}",
        f"{OMNI_VENV}/bin/pip install --no-cache-dir {OMNI_RUNTIME_DEPS}",
    )
    .add_local_file(
        (HERE / "modal_relax_smoke.py").as_posix(),
        "/root/modal_relax_smoke.py",
        copy=True,
    )
    .add_local_dir(
        OMNI_LOCAL.as_posix(),
        OMNI_REMOTE,
        copy=True,
        ignore=["**/.git", "**/__pycache__", "**/*.pyc", "**/node_modules"],
    )
)

app = modal.App(APP_NAME)


_OMNI_SERVER_CODE = f"""
from sglang_omni.cli.serve import (
    _apply_stage_server_args_override,
    apply_parallelism_cli_overrides,
    apply_thinker_server_args_cli_overrides,
)
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.serve import launch_server

config = Qwen3OmniPipelineConfig(model_path={MODEL_DIR!r})
apply_parallelism_cli_overrides(
    config,
    thinker_tp_size={OMNI_TP_SIZE},
    thinker_gpus="0,1,2,3",
    talker_gpu=None,
    code2wav_gpu=None,
)
apply_thinker_server_args_cli_overrides(
    config,
    cpu_offload_gb=None,
    quantization=None,
    enable_lora=True,
    max_lora_rank=16,
    lora_target_modules="qkv_proj,o_proj",
    max_loras_per_batch=1,
)
_apply_stage_server_args_override(
    config,
    stage_name="thinker",
    updates={{
        "attention_backend": "triton",
        "disable_cuda_graph": True,
        "disable_custom_all_reduce": True,
        "mem_fraction_static": 0.55,
        "disable_overlap_schedule": True,
        "max_running_requests": 4,
    }},
    reason="real Relax tensor-LoRA smoke settings",
)
launch_server(
    config,
    host="127.0.0.1",
    port={OMNI_PORT},
    model_name="qwen3-omni",
)
"""


def _assigned_gpus() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
    else:
        devices = [str(index) for index in range(TOTAL_GPU_COUNT)]
    if len(devices) < TOTAL_GPU_COUNT:
        raise RuntimeError(
            f"Expected {TOTAL_GPU_COUNT} assigned GPUs, got {devices}"
        )
    return devices


def _write_bleu_smoke_dataset(source_path: str, destination_path: str) -> None:
    import wave

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Missing FLEURS dataset {source_path}; run modal_relax_smoke.py::prep"
        )

    record = None
    with open(source_path, encoding="utf-8") as source:
        for line in source:
            candidate = json.loads(line)
            audio_paths = candidate.get("audios")
            label = candidate.get("label")
            ground_truth = (
                label.get("ground_truth") if isinstance(label, dict) else label
            )
            if (
                isinstance(audio_paths, list)
                and len(audio_paths) == 1
                and os.path.exists(audio_paths[0])
                and ground_truth
            ):
                record = candidate
                break

    if record is None:
        raise RuntimeError(f"No usable FLEURS record found in {source_path}")

    audio_path = record.pop("audios")[0]
    record["audio"] = [audio_path]
    record.setdefault("metadata", {})["id"] = "omni-real-bleu-smoke-0"
    with wave.open(audio_path, "rb") as audio:
        duration_s = audio.getnframes() / audio.getframerate()
    if duration_s > 8.0:
        raise RuntimeError(
            f"Cost-controlled BLEU smoke expected <=8s audio, got {duration_s:.2f}s"
        )

    with open(destination_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        "[data] FLEURS BLEU smoke "
        f"duration={duration_s:.2f}s "
        f"source={record['metadata'].get('src_text', '')!r} "
        f"reference={record['label']!r}",
        flush=True,
    )


def _write_bleu_smoke_config(path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("max_turns: 1\nsimul_chunk_ms: 30000\n")


_GRAD_NORM_RE = re.compile(r"'train/grad_norm': ([0-9.eE+-]+)")


def _validate_training_evidence(lines: list[str], num_rollout: int) -> None:
    output = "".join(lines)
    required_markers = [
        f"Actor training completed step {num_rollout - 1}/{num_rollout}",
        "All training steps finished",
        f"rollout {num_rollout - 1}:",
    ]
    missing = [marker for marker in required_markers if marker not in output]
    if missing:
        raise RuntimeError(f"Missing training evidence markers: {missing}")

    grad_norms = [
        float(match.group(1))
        for match in _GRAD_NORM_RE.finditer(output)
    ]
    if not grad_norms or not any(value > 0.0 for value in grad_norms):
        raise RuntimeError(
            "BLEU smoke completed without a non-zero gradient; "
            f"observed grad_norms={grad_norms}"
        )
    print(f"[evidence] non-zero training grad_norms={grad_norms}", flush=True)


def _wait_for_omni(process, *, required_process=None) -> None:
    import time

    import requests

    base_url = f"http://127.0.0.1:{OMNI_PORT}"
    for attempt in range(180):
        if required_process is not None and required_process.poll() is not None:
            raise RuntimeError(
                "Relax exited while waiting for SGLang-Omni readiness "
                f"with code {required_process.returncode}"
            )
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang-Omni exited before readiness with code {process.returncode}"
            )
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                model_info = requests.get(
                    f"{base_url}/model_info",
                    timeout=10,
                )
                model_info.raise_for_status()
                print(
                    f"[omni] ready after {attempt * 5}s: {model_info.json()}",
                    flush=True,
                )
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("SGLang-Omni did not become healthy within 15 minutes")


def _stop_process_group(process) -> None:
    import signal
    import time

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except Exception:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    time.sleep(1)


@app.function(
    image=image,
    cpu=2.0,
    volumes={"/models": model_volume},
    timeout=20 * 60,
)
def probe() -> None:
    """Cheap image/import/config probe; does not allocate a GPU."""
    import subprocess

    subprocess.run(
        [
            OMNI_PYTHON,
            "-c",
            (
                "import transformers, sglang_omni; "
                "assert transformers.__version__ == '5.6.0'; "
                "print('omni transformers', transformers.__version__)"
            ),
        ],
        env={
            **os.environ,
            "PYTHONPATH": f"{SGLANG_REMOTE}:{OMNI_REMOTE}",
        },
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-c",
            (
                "import transformers, relax; "
                "assert transformers.__version__ == '5.3.0'; "
                "print('relax transformers', transformers.__version__)"
            ),
        ],
        env={**os.environ, "PYTHONPATH": RELAX_PYTHONPATH},
        check=True,
    )
    subprocess.run(
        [
            "bash",
            "-n",
            (
                f"{RELAX_REMOTE}/scripts/training/multimodal/"
                "run-qwen3-30B-A3B-omni-lora-omni-simul.sh"
            ),
        ],
        check=True,
    )
    print("PROBE PASS", flush=True)


@app.function(
    image=image,
    cpu=4.0,
    timeout=20 * 60,
)
def unit() -> None:
    """Run the CPU regressions that cover the independent Omni backend."""
    import subprocess

    subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "tests/engine/rollout/test_sglang_omni_rollout.py",
            "tests/distributed/ray/test_rollout_engine_selection.py",
            "tests/backends/sglang_omni/test_omni_engine.py",
        ],
        cwd=RELAX_REMOTE,
        env={**os.environ, "PYTHONPATH": RELAX_PYTHONPATH},
        check=True,
    )


@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={"/models": model_volume, S2TT_DIR: s2tt_volume},
    timeout=3 * 60 * 60,
)
def smoke(num_rollout: int = 1) -> None:
    """Run the real external-Omni Relax training closure."""
    import subprocess
    import threading
    import time

    if num_rollout not in (1, 2):
        raise ValueError("Cost-controlled smoke only permits one or two rollouts")
    if not os.path.exists(f"{MODEL_DIR}/config.json"):
        raise FileNotFoundError(f"Missing model volume file {MODEL_DIR}/config.json")

    devices = _assigned_gpus()
    train_devices = devices[:TRAIN_GPU_COUNT]
    omni_devices = devices[:OMNI_TP_SIZE]
    print(
        f"[gpu split] Relax={train_devices}, SGLang-Omni={omni_devices}",
        flush=True,
    )

    train_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(train_devices),
        "MEGATRON": MEGATRON_REMOTE,
        "RELAX": RELAX_REMOTE,
        "MODEL_CONFIG_DIR": f"{RELAX_REMOTE}/scripts/models",
        "PYTHONPATH": RELAX_PYTHONPATH,
        "RELAX_ENTRYPOINT_MODE": "local",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTHONUNBUFFERED": "1",
        "NUM_GPUS": "4",
        "RAY_ADDRESS": "http://127.0.0.1:8265",
        "HF_CKPT": MODEL_DIR,
        "MODEL_DIR": "/models",
        "DATA": "/tmp/omni-real-bleu-smoke.jsonl",
        "NUM_ROLLOUT": str(num_rollout),
        "RM_TYPE": "bleu",
        "N_SAMPLES": "4",
        "ROLLOUT_BATCH": "1",
        "GLOBAL_BATCH": "4",
        "ROLLOUT_TEMPERATURE": "1.3",
        "ROLLOUT_MAX_RESPONSE_LEN": "64",
        "ROLLOUT_MAX_PROMPT_LEN": "4096",
        "CUSTOM_CONFIG_PATH": "/tmp/omni-real-bleu-smoke.yaml",
        "MAX_GLOBAL_RESTART": "0",
        "OMNI_ROUTER_HOST": "127.0.0.1",
        "OMNI_ROUTER_PORT": str(OMNI_PORT),
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "RUNTIME_ENV_JSON": json.dumps(
            {
                "env_vars": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": RELAX_PYTHONPATH,
                    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
                    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
                }
            }
        ),
    }
    script = (
        f"{RELAX_REMOTE}/scripts/training/multimodal/"
        "run-qwen3-30B-A3B-omni-lora-omni-simul.sh"
    )

    omni_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(omni_devices),
        "PYTHONPATH": f"{SGLANG_REMOTE}:{OMNI_REMOTE}",
        "PYTHONUNBUFFERED": "1",
        "SGLANG_OMNI_STARTUP_TIMEOUT": "1800",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    }
    train_process = None
    train_output_thread = None
    omni_process = None
    train_output_lines: list[str] = []

    try:
        _write_bleu_smoke_dataset(
            S2TT_JSONL,
            train_env["DATA"],
        )
        _write_bleu_smoke_config(train_env["CUSTOM_CONFIG_PATH"])
        subprocess.run(
            ["ray", "stop", "--force"],
            env=train_env,
            check=False,
        )
        subprocess.run(
            [
                "ray",
                "start",
                "--head",
                "--node-ip-address",
                "127.0.0.1",
                "--num-gpus",
                "4",
                "--disable-usage-stats",
                "--dashboard-host=0.0.0.0",
                "--dashboard-port=8265",
            ],
            env=train_env,
            check=True,
        )

        # Match Relax's proven colocated lifecycle: finish the Megatron actor
        # initialization before allocating the rollout model and its KV cache.
        # Starting Omni first makes Megatron hit its temporary model-construction
        # peak while roughly 43.5 GiB/rank is already occupied.
        actor_ready = threading.Event()

        train_process = subprocess.Popen(
            ["bash", script],
            cwd=RELAX_REMOTE,
            env=train_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def _forward_train_output() -> None:
            assert train_process is not None
            assert train_process.stdout is not None
            for line in train_process.stdout:
                train_output_lines.append(line)
                print(line, end="", flush=True)
                if "[actor] Service deployed successfully" in line:
                    actor_ready.set()

        train_output_thread = threading.Thread(
            target=_forward_train_output,
            name="relax-train-output",
            daemon=True,
        )
        train_output_thread.start()

        actor_deadline = time.monotonic() + 20 * 60
        while not actor_ready.wait(timeout=1):
            returncode = train_process.poll()
            if returncode is not None:
                train_output_thread.join(timeout=10)
                raise subprocess.CalledProcessError(
                    returncode,
                    ["bash", script],
                )
            if time.monotonic() >= actor_deadline:
                raise TimeoutError(
                    "Relax Megatron actor did not become ready within 20 minutes"
                )

        print(
            "[startup] Megatron actor ready; starting colocated SGLang-Omni TP4",
            flush=True,
        )
        omni_process = subprocess.Popen(
            [OMNI_PYTHON, "-c", _OMNI_SERVER_CODE],
            env=omni_env,
            start_new_session=True,
        )
        _wait_for_omni(
            omni_process,
            required_process=train_process,
        )

        returncode = train_process.wait()
        train_output_thread.join(timeout=10)
        if returncode != 0:
            raise subprocess.CalledProcessError(
                returncode,
                ["bash", script],
            )
        _validate_training_evidence(train_output_lines, num_rollout)
        print(f"REAL RELAX OMNI SMOKE PASS (num_rollout={num_rollout})", flush=True)
    finally:
        if train_process is not None:
            _stop_process_group(train_process)
        if train_output_thread is not None:
            train_output_thread.join(timeout=10)
        subprocess.run(
            ["ray", "stop", "--force"],
            env=train_env,
            check=False,
        )
        if omni_process is not None:
            _stop_process_group(omni_process)
        print("[cleanup] Ray and SGLang-Omni stopped", flush=True)


@app.local_entrypoint()
def main(num_rollout: int = 1) -> None:
    smoke.remote(num_rollout=num_rollout)
