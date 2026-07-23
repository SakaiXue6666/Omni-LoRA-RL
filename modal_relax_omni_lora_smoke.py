"""Real Relax -> SGLang-Omni tensor-LoRA training smoke test on Modal.

This is intentionally separate from:

* ``modal_relax_smoke.py`` (the standard SGLang training baseline), and
* ``modal_omni_serve_lora_tensor.py`` (the simulated tensor-update E2E).

The test runs Megatron TP4 and an external SGLang-Omni thinker TP4 colocated
on the same four A100-80GB GPUs. One rollout
step proves initial adapter loading, rollout, training, and the post-train
unload/reload. Use two steps to additionally consume the trained adapter in a
subsequent rollout. The opt-in ``speech`` mode keeps Talker/Code2Wav in the
same service and synthesizes one WAV with the final in-memory Thinker adapter.
"""

from __future__ import annotations

import json
import math
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
STREAMING_CHUNK_MS = 960
POST_TRAIN_SPEECH_REQUEST_ID = "relax-post-train-speech"

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


_OMNI_TEXT_SERVER_CODE = f"""
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


_OMNI_SPEECH_CONFIG_CODE = f"""
from sglang_omni.cli.serve import (
    _apply_stage_server_args_override,
    apply_parallelism_cli_overrides,
    apply_thinker_server_args_cli_overrides,
)
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniSpeechPipelineConfig,
)

config = Qwen3OmniSpeechPipelineConfig(model_path={MODEL_DIR!r})

# Use the supported hybrid speech topology: Thinker is TP4 while Talker,
# Code2Wav, and the encoders share the Thinker leader's GPU 0. Keep GPU 0's
# complete speech stack close to the proven 0.55 Thinker-only
# budget: 0.45 Thinker + 0.10 Talker + 3 * 0.02 auxiliary stages = 0.61.
# The Thinker runtime subtracts its existing 0.05 encoder reserve, leaving a
# 0.40 SGLang/KV budget on each TP rank, which is ample for four short samples.
stage_fractions = {{
    "image_encoder": 0.02,
    "audio_encoder": 0.02,
    "thinker": 0.45,
    "talker_ar": 0.10,
    "code2wav": 0.02,
}}
stage_by_name = {{stage.name: stage for stage in config.stages}}
for stage_name, fraction in stage_fractions.items():
    stage_by_name[stage_name].runtime.resources.total_gpu_memory_fraction = fraction

apply_parallelism_cli_overrides(
    config,
    thinker_tp_size={OMNI_TP_SIZE},
    thinker_gpus="0,1,2,3",
    talker_gpu=0,
    code2wav_gpu=0,
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

for stage_name in ("thinker", "talker_ar"):
    _apply_stage_server_args_override(
        config,
        stage_name=stage_name,
        updates={{
            "attention_backend": "triton",
            "disable_cuda_graph": True,
            "disable_custom_all_reduce": True,
            "disable_overlap_schedule": True,
            **({{"max_running_requests": 4}} if stage_name == "thinker" else {{}}),
        }},
        reason="real Relax post-train speech smoke settings",
    )
"""


_OMNI_SPEECH_SERVER_CODE = _OMNI_SPEECH_CONFIG_CODE + f"""
from sglang_omni.serve import launch_server

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


def _write_bleu_smoke_dataset(source_path: str, destination_path: str) -> float:
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
    return duration_s


def _write_bleu_smoke_config(path: str, *, streaming: bool) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        if streaming:
            handle.write(
                f"max_turns: 8\nsimul_chunk_ms: {STREAMING_CHUNK_MS}\n"
            )
        else:
            handle.write("max_turns: 1\nsimul_chunk_ms: 30000\n")


_GRAD_NORM_RE = re.compile(r"'train/grad_norm': ([0-9.eE+-]+)")
_TRAIN_LOSS_RE = re.compile(r"'train/loss': ([^,}]+)")
_PPO_KL_RE = re.compile(r"'train/ppo_kl': ([^,}]+)")
_PG_CLIPFRAC_RE = re.compile(r"'train/pg_clipfrac': ([^,}]+)")
_ROLLOUT_RAW_REWARD_RE = re.compile(r"'rollout/raw_reward': ([^,}]+)")
_STREAMING_ROLLOUT_RE = re.compile(
    r"\[omni-simul-evidence\] "
    r"sample_index=(?P<sample_index>\S+) "
    r"rollout_turns=(?P<rollout_turns>\d+) "
    r"num_chunks=(?P<num_chunks>\d+) "
    r"stop_reason=(?P<stop_reason>\S+) "
    r"status=(?P<status>\S+)"
)
_STREAMING_OUTPUT_RE = re.compile(
    r"\[omni-simul-output\] "
    r"sample_index=(?P<sample_index>\S+) "
    r"response=(?P<response>\"(?:\\.|[^\"\\])*\")"
)


def _validate_training_evidence(
    lines: list[str],
    num_rollout: int,
    *,
    require_nonzero_gradient: bool = True,
) -> None:
    output = "".join(lines)
    required_markers = [
        f"Actor training completed step {num_rollout - 1}/{num_rollout}",
        "All training steps finished",
        f"rollout {num_rollout - 1}:",
    ]
    missing = [marker for marker in required_markers if marker not in output]
    if missing:
        raise RuntimeError(f"Missing training evidence markers: {missing}")

    raw_rewards = [
        float(match.group(1))
        for match in _ROLLOUT_RAW_REWARD_RE.finditer(output)
    ]
    if len(raw_rewards) < num_rollout or not all(
        math.isfinite(value) for value in raw_rewards
    ):
        raise RuntimeError(
            "BLEU smoke did not report finite rollout rewards for every step; "
            f"observed raw_rewards={raw_rewards}"
        )
    train_losses = [
        float(match.group(1))
        for match in _TRAIN_LOSS_RE.finditer(output)
    ]
    if len(train_losses) < num_rollout or not all(
        math.isfinite(value) for value in train_losses
    ):
        raise RuntimeError(
            "BLEU smoke did not report finite training losses for every step; "
            f"observed train_losses={train_losses}"
        )
    ppo_kls = [float(match.group(1)) for match in _PPO_KL_RE.finditer(output)]
    clipfracs = [
        float(match.group(1))
        for match in _PG_CLIPFRAC_RE.finditer(output)
    ]
    if (
        len(ppo_kls) < num_rollout
        or len(clipfracs) < num_rollout
        or not all(math.isfinite(value) for value in ppo_kls + clipfracs)
    ):
        raise RuntimeError(
            "BLEU smoke did not report finite PPO alignment metrics for every step; "
            f"observed ppo_kls={ppo_kls}, clipfracs={clipfracs}"
        )
    grad_norms = [
        float(match.group(1))
        for match in _GRAD_NORM_RE.finditer(output)
    ]
    if (
        require_nonzero_gradient
        and (not grad_norms or not any(value > 0.0 for value in grad_norms))
    ):
        raise RuntimeError(
            "BLEU smoke completed without a non-zero gradient; "
            f"observed grad_norms={grad_norms}"
        )
    print(
        "[evidence] training "
        f"raw_rewards={raw_rewards} losses={train_losses} grad_norms={grad_norms} "
        f"ppo_kls={ppo_kls} clipfracs={clipfracs}",
        flush=True,
    )


def _validate_streaming_evidence(
    train_lines: list[str],
    omni_lines: list[str],
    *,
    expected_samples: int,
    expected_chunks: int,
    expected_steps: int,
) -> None:
    train_output = "".join(train_lines)
    omni_output = "".join(omni_lines)
    summaries = [
        {
            "sample_index": match.group("sample_index"),
            "rollout_turns": int(match.group("rollout_turns")),
            "num_chunks": int(match.group("num_chunks")),
            "stop_reason": match.group("stop_reason"),
            "status": match.group("status"),
        }
        for match in _STREAMING_ROLLOUT_RE.finditer(train_output)
    ]
    if len(summaries) != expected_samples:
        raise RuntimeError(
            "Streaming smoke did not report one rollout summary per sample; "
            f"expected={expected_samples}, observed={summaries}"
        )
    if any(summary["num_chunks"] != expected_chunks for summary in summaries):
        raise RuntimeError(
            "Streaming smoke used an unexpected audio chunk count; "
            f"expected={expected_chunks}, observed={summaries}"
        )
    if any(summary["rollout_turns"] < 2 for summary in summaries):
        raise RuntimeError(
            "Streaming smoke did not exercise a multi-turn trajectory for every "
            f"sample: {summaries}"
        )
    invalid_stops = [
        summary
        for summary in summaries
        if summary["stop_reason"] not in {"chunks_exhausted", "length"}
        or summary["status"] not in {"completed", "truncated"}
    ]
    if invalid_stops:
        raise RuntimeError(
            "Streaming smoke observed an abort or invalid terminal state: "
            f"{invalid_stops}"
        )
    full_audio_coverage = sum(
        summary["rollout_turns"] >= expected_chunks for summary in summaries
    )
    if full_audio_coverage < 1:
        raise RuntimeError(
            "Streaming smoke never submitted every audio chunk for any trajectory; "
            f"expected_chunks={expected_chunks}, observed={summaries}"
        )
    if full_audio_coverage < expected_steps:
        raise RuntimeError(
            "Streaming smoke did not provide at least one full-audio trajectory "
            f"per training step; expected={expected_steps}, "
            f"observed={full_audio_coverage}"
        )
    output_previews = [
        {
            "sample_index": match.group("sample_index"),
            "response": json.loads(match.group("response")),
        }
        for match in _STREAMING_OUTPUT_RE.finditer(train_output)
    ]
    if len(output_previews) != expected_samples:
        raise RuntimeError(
            "Streaming smoke did not report one output preview per sample; "
            f"expected={expected_samples}, observed={output_previews}"
        )
    invalid_outputs = [
        preview
        for preview in output_previews
        if not preview["response"].strip()
        or "\ufffd" in preview["response"]
        or any(
            ord(character) < 32
            and character not in {"\n", "\r", "\t"}
            for character in preview["response"]
        )
    ]
    if invalid_outputs:
        raise RuntimeError(
            "Streaming smoke observed an empty or malformed Unicode output: "
            f"{invalid_outputs}"
        )
    successful_requests = omni_output.count(
        'POST /generate HTTP/1.1" 200 OK'
    )
    expected_requests = sum(
        summary["rollout_turns"] for summary in summaries
    )
    if successful_requests != expected_requests:
        raise RuntimeError(
            "Streaming rollout summaries do not match successful Omni requests; "
            f"expected={expected_requests}, observed={successful_requests}"
        )
    if "has_active_lora=True" not in omni_output:
        raise RuntimeError("Streaming smoke never observed an active Thinker policy LoRA")
    control_events = [
        match.group(1)
        for match in re.finditer(
            r'POST /(load_lora_adapter_from_tensors|'
            r'generate|unload_lora_adapter) HTTP/1\.1" 200 OK',
            omni_output,
        )
    ]
    load_positions = [
        index
        for index, event in enumerate(control_events)
        if event == "load_lora_adapter_from_tensors"
    ]
    expected_loads = expected_steps + 1
    if len(load_positions) != expected_loads:
        raise RuntimeError(
            "Streaming smoke did not complete the initial and post-train tensor "
            f"loads; expected={expected_loads}, observed={len(load_positions)}"
        )
    unloads = control_events.count("unload_lora_adapter")
    if unloads != expected_steps:
        raise RuntimeError(
            "Streaming smoke did not unload the previous adapter after every "
            f"training step; expected={expected_steps}, observed={unloads}"
        )
    missing_post_reload_rollouts = [
        step
        for step in range(expected_steps)
        if "generate"
        not in control_events[load_positions[step] + 1 : load_positions[step + 1]]
    ]
    if missing_post_reload_rollouts:
        raise RuntimeError(
            "Streaming smoke did not consume the currently loaded adapter before "
            f"the next training update; missing_steps={missing_post_reload_rollouts}"
        )
    chunks_exhausted = sum(
        summary["stop_reason"] == "chunks_exhausted" for summary in summaries
    )
    truncated = sum(
        summary["status"] == "truncated" for summary in summaries
    )
    print(
        "[evidence] streaming "
        f"requests={successful_requests} "
        f"full_audio_coverage={full_audio_coverage}/{expected_samples} "
        f"chunks_exhausted={chunks_exhausted}/{expected_samples} "
        f"truncated={truncated}/{expected_samples}",
        flush=True,
    )
    print(
        "[evidence] streaming output previews="
        f"{json.dumps(output_previews, ensure_ascii=True)}",
        flush=True,
    )


def _run_post_train_speech(base_url: str) -> dict[str, object]:
    """Use the final in-memory policy adapter for one Talker/Code2Wav request."""
    import base64
    import io

    import numpy as np
    import requests
    import soundfile as sf

    payload = {
        "model": "qwen3-omni",
        "request_id": POST_TRAIN_SPEECH_REQUEST_ID,
        "messages": [
            {
                "role": "user",
                "content": "Say this short sentence: Hello after one Relax update.",
            }
        ],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
        "stage_params": {"thinker": {"lora_name": "policy"}},
    }
    print(
        "[speech] requesting post-train text+audio with Thinker LoRA policy",
        flush=True,
    )
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=15 * 60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Post-train speech request failed: "
            f"HTTP {response.status_code} {response.text[:1200]}"
        )

    message = response.json()["choices"][0]["message"]
    text = message.get("content") or ""
    audio = message.get("audio") or {}
    if not text.strip():
        raise RuntimeError("Post-train speech request returned empty Thinker text")
    if not audio.get("data"):
        raise RuntimeError("Post-train speech request returned no audio.data")

    wav_bytes = base64.b64decode(audio["data"], validate=True)
    if wav_bytes[:4] not in {b"RIFF", b"RF64"}:
        raise RuntimeError("Post-train speech response is not a WAV container")
    waveform, sample_rate = sf.read(
        io.BytesIO(wav_bytes), dtype="float32", always_2d=False
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    frames = int(waveform.shape[0])
    duration_s = frames / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if sample_rate <= 0 or frames <= 0 or duration_s < 0.05:
        raise RuntimeError(
            "Post-train speech WAV has invalid geometry: "
            f"sample_rate={sample_rate}, frames={frames}, duration={duration_s}"
        )
    if not math.isfinite(rms) or rms <= 1e-6:
        raise RuntimeError(f"Post-train speech WAV is silent or invalid: rms={rms}")

    print(f"[speech] Thinker text={text!r}", flush=True)
    print(
        "[speech] Talker WAV "
        f"bytes={len(wav_bytes)} sample_rate={sample_rate} frames={frames} "
        f"duration={duration_s:.3f}s rms={rms:.6f}",
        flush=True,
    )
    return {
        "wav_bytes": wav_bytes,
        "text": text,
        "sample_rate": int(sample_rate),
        "frames": frames,
        "duration_s": duration_s,
        "rms": rms,
    }


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
    speech_probe_code = _OMNI_SPEECH_CONFIG_CODE + """
from sglang_omni.config import build_stage_placement_plan

placement = build_stage_placement_plan(config)
assert stage_by_name["thinker"].tp_size == 4
assert stage_by_name["thinker"].gpu == [0, 1, 2, 3]
assert stage_by_name["talker_ar"].gpu == 0
assert stage_by_name["code2wav"].gpu == 0
assert abs(sum(stage_fractions.values()) - 0.61) < 1e-9
assert abs(placement.gpus[0].total_gpu_memory_fraction - 0.61) < 1e-9
for gpu_id in (1, 2, 3):
    assert abs(placement.gpus[gpu_id].total_gpu_memory_fraction - 0.45) < 1e-9
assert placement.same_gpu_stream_targets["thinker"] == frozenset({"talker_ar"})
print("speech topology", {
    name: (stage_by_name[name].gpu, fraction)
    for name, fraction in stage_fractions.items()
})
"""
    subprocess.run(
        [OMNI_PYTHON, "-c", speech_probe_code],
        env={
            **os.environ,
            "PYTHONPATH": f"{SGLANG_REMOTE}:{OMNI_REMOTE}",
        },
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
    """Run CPU regressions for the Omni backend and its stage IPC bridge."""
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
    subprocess.run(
        [
            OMNI_PYTHON,
            "-m",
            "pytest",
            "-q",
            "tests/unit_test/pipeline/test_stage_streaming.py",
            "tests/unit_test/scheduling/test_lora_admin.py",
        ],
        cwd=OMNI_REMOTE,
        env={
            **os.environ,
            "PYTHONPATH": f"{SGLANG_REMOTE}:{OMNI_REMOTE}",
        },
        check=True,
    )


@app.function(
    image=image,
    cpu=8.0,
    gpu="A100-80GB:4",
    volumes={"/models": model_volume, S2TT_DIR: s2tt_volume},
    timeout=3 * 60 * 60,
)
def smoke(
    num_rollout: int = 1,
    streaming: bool = False,
    speech: bool = False,
) -> dict[str, object] | None:
    """Run the real external-Omni Relax training closure."""
    import subprocess
    import threading
    import time

    if speech and (streaming or num_rollout != 1):
        raise ValueError(
            "Cost-controlled post-train speech smoke requires "
            "num_rollout=1 and streaming=False"
        )
    if streaming:
        if num_rollout not in (1, 2, 3):
            raise ValueError(
                "Cost-controlled streaming smoke permits at most three rollouts"
            )
    elif num_rollout not in (1, 2):
        raise ValueError(
            "Cost-controlled full-audio smoke permits one or two rollouts"
        )
    if not os.path.exists(f"{MODEL_DIR}/config.json"):
        raise FileNotFoundError(f"Missing model volume file {MODEL_DIR}/config.json")

    samples_per_prompt = 4
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
        "N_SAMPLES": str(samples_per_prompt),
        "ROLLOUT_BATCH": "1",
        "GLOBAL_BATCH": str(samples_per_prompt),
        "ROLLOUT_TEMPERATURE": "0.8" if streaming else "1.3",
        "ROLLOUT_MAX_RESPONSE_LEN": "128" if streaming else "64",
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
    omni_output_thread = None
    train_output_lines: list[str] = []
    omni_output_lines: list[str] = []
    speech_result: dict[str, object] | None = None

    try:
        duration_s = _write_bleu_smoke_dataset(
            S2TT_JSONL,
            train_env["DATA"],
        )
        _write_bleu_smoke_config(
            train_env["CUSTOM_CONFIG_PATH"],
            streaming=streaming,
        )
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
            "[startup] Megatron actor ready; starting colocated SGLang-Omni TP4 "
            f"(speech={speech})",
            flush=True,
        )
        omni_server_code = (
            _OMNI_SPEECH_SERVER_CODE if speech else _OMNI_TEXT_SERVER_CODE
        )
        omni_process = subprocess.Popen(
            [OMNI_PYTHON, "-c", omni_server_code],
            env=omni_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def _forward_omni_output() -> None:
            assert omni_process is not None
            assert omni_process.stdout is not None
            for line in omni_process.stdout:
                omni_output_lines.append(line)
                print(line, end="", flush=True)

        omni_output_thread = threading.Thread(
            target=_forward_omni_output,
            name="omni-output",
            daemon=True,
        )
        omni_output_thread.start()
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
        _validate_training_evidence(
            train_output_lines,
            num_rollout,
            require_nonzero_gradient=not streaming,
        )
        if streaming:
            _validate_streaming_evidence(
                train_output_lines,
                omni_output_lines,
                expected_samples=samples_per_prompt * num_rollout,
                expected_chunks=math.ceil(
                    duration_s * 1000 / STREAMING_CHUNK_MS
                ),
                expected_steps=num_rollout,
            )
        if speech:
            speech_result = _run_post_train_speech(
                f"http://127.0.0.1:{OMNI_PORT}"
            )
            expected_routing = (
                f"request_ids=['{POST_TRAIN_SPEECH_REQUEST_ID}'] "
                "lora_ids=['policy'] has_active_lora=True"
            )
            routing_deadline = time.monotonic() + 10
            while expected_routing not in "".join(omni_output_lines):
                if time.monotonic() >= routing_deadline:
                    raise RuntimeError(
                        "Post-train speech did not prove active Thinker policy routing; "
                        f"expected marker={expected_routing!r}"
                    )
                time.sleep(0.1)
            print(
                "[evidence] post-train speech used active Thinker policy LoRA",
                flush=True,
            )
        print(
            "REAL RELAX OMNI SMOKE PASS "
            f"(num_rollout={num_rollout}, streaming={streaming}, speech={speech})",
            flush=True,
        )
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
        if omni_output_thread is not None:
            omni_output_thread.join(timeout=10)
        print("[cleanup] Ray and SGLang-Omni stopped", flush=True)
    return speech_result


@app.local_entrypoint()
def main(
    num_rollout: int = 1,
    streaming: bool = False,
    speech: bool = False,
) -> None:
    result = smoke.remote(
        num_rollout=num_rollout,
        streaming=streaming,
        speech=speech,
    )
    if speech:
        if result is None:
            raise RuntimeError("Post-train speech smoke returned no WAV result")
        output_path = HERE / "relax_omni_post_train_speech.wav"
        output_path.write_bytes(result["wav_bytes"])
        print(f"[local] Thinker text: {result['text']!r}")
        print(
            "[local] Post-train Talker WAV saved: "
            f"{output_path} ({result['duration_s']:.3f}s, "
            f"{result['sample_rate']} Hz, RMS={result['rms']:.6f})"
        )
