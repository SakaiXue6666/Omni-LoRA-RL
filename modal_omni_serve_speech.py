"""Qwen3-Omni Talker/Code2Wav 的最小 Modal A100 语音冒烟测试。

保持 ``modal_omni_serve_lora.py`` 的磁盘 LoRA 基线不变。本脚本只验证：

1. 一张 A100-80GB 能按官方 colocated BF16 配置启动完整 speech pipeline；
2. ``/v1/chat/completions`` 同时返回 Thinker 文本和 Talker 音频；
3. Code2Wav 返回的 base64 WAV 可解码，且时长、能量都不是零。
4. 完整 WAV 会下载到本脚本旁边的 ``talker_smoke_output.wav``，供人耳试听。

运行方式：

    modal run modal_omni_serve_speech.py::probe
    modal run modal_omni_serve_speech.py
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
SGLANG_LOCAL = HERE / "sglang" / "python"
SGLANG_OMNI_LOCAL = HERE / "sglang-omni"

SGLANG_REMOTE = "/root/sglang_src/python"
SGLANG_OMNI_REMOTE = "/root/sglang-omni"
PYTHONPATH = f"{SGLANG_REMOTE}:{SGLANG_OMNI_REMOTE}"

MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_DIR = "/models/qwen3-omni"
COLOCATED_CONFIG = (
    f"{SGLANG_OMNI_REMOTE}/examples/configs/qwen3_omni_colocated_h100_bf16.yaml"
)

BASE_IMAGE = os.environ.get("RELAX_BASE_IMAGE", "slimerl/slime:latest")
OMNI_DEPS = (
    "pip install --no-cache-dir "
    "typer pyzmq msgpack pydantic pyyaml xxhash httpx fastapi uvicorn pybase64 "
    "requests pillow accelerate safetensors soundfile librosa av qwen-vl-utils "
    "|| true"
)
TRANSFORMERS_DEP = "pip install --no-cache-dir 'transformers==5.6.0' || true"

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python=None)
    .run_commands(OMNI_DEPS)
    .run_commands(TRANSFORMERS_DEP)
    .add_local_dir(
        SGLANG_LOCAL.as_posix(),
        SGLANG_REMOTE,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        SGLANG_OMNI_LOCAL.as_posix(),
        SGLANG_OMNI_REMOTE,
        copy=True,
        ignore=["**/.git", "**/__pycache__", "**/*.pyc", "**/node_modules"],
    )
    .env({"PYTHONPATH": PYTHONPATH, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)
test_image = image.run_commands("pip install --no-cache-dir pytest || true")

app = modal.App("sglang-omni-speech-smoke")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


def _build_speech_config():
    """加载官方 colocated 显存合同，仅覆盖本地模型路径和稳定性参数。"""
    from sglang_omni.cli.serve import _apply_stage_server_args_override
    from sglang_omni.config.manager import ConfigManager

    manager = ConfigManager.from_file(COLOCATED_CONFIG)
    config = manager.merge_config({"model_path": MODEL_DIR})

    # A100 冒烟优先稳定性，不需要 CUDA graph/overlap 的吞吐收益。
    for stage_name in ("thinker", "talker_ar"):
        _apply_stage_server_args_override(
            config,
            stage_name=stage_name,
            updates={
                "attention_backend": "triton",
                "disable_cuda_graph": True,
                "disable_custom_all_reduce": True,
                "disable_overlap_schedule": True,
            },
            reason="single-A100 speech smoke stability settings",
        )
    return config


def _serve_speech() -> None:
    """子进程入口：在 GPU 0 启动完整 Thinker + Talker + Code2Wav。"""
    from sglang_omni.serve import launch_server

    config = _build_speech_config()
    launch_server(config, host="127.0.0.1", port=8000, model_name="qwen3-omni")


@app.function(image=image, cpu=2.0, timeout=900)
def probe() -> None:
    """不申请 GPU：验证 speech 配置能构建且五个 GPU stage 都 colocate 到 GPU 0。"""
    config = _build_speech_config()
    expected = {"image_encoder", "audio_encoder", "thinker", "talker_ar", "code2wav"}
    gpu_stages = {stage.name: stage for stage in config.stages if stage.name in expected}
    assert set(gpu_stages) == expected
    assert {stage.gpu for stage in gpu_stages.values()} == {0}

    fractions = {
        name: stage.runtime.resources.total_gpu_memory_fraction
        for name, stage in gpu_stages.items()
    }
    assert abs(sum(float(value) for value in fractions.values()) - 0.94) < 1e-9
    print("[probe] stages:", [stage.name for stage in config.stages])
    print("[probe] colocated GPU:", {name: stage.gpu for name, stage in gpu_stages.items()})
    print("[probe] memory fractions:", fractions)
    print("[probe] PASS —— 单 A100 speech 配置构建成功")


@app.function(image=test_image, cpu=2.0, timeout=900)
def unit() -> None:
    """不申请 GPU：只跑 Talker forward-context 回归测试。"""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/unit_test/qwen3_omni/test_talker_forward_context.py",
            "tests/unit_test/qwen3_omni/test_fp8_backend_config.py",
            "-p",
            "no:cacheprovider",
        ],
        cwd=SGLANG_OMNI_REMOTE,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"[unit] pytest failed with code {result.returncode}")
    print("[unit] PASS —— Talker custom prefill publishes ForwardContext")


@app.function(
    image=image,
    gpu="A100-80GB:1",
    volumes={"/models": model_volume},
    timeout=60 * 60,
)
def e2e() -> dict[str, object]:
    """启动完整 speech pipeline，发一条短请求并验证 WAV。"""
    import base64
    import glob
    import io
    import multiprocessing
    import subprocess
    import time

    import numpy as np
    import requests
    import soundfile as sf

    os.environ.setdefault("SGLANG_OMNI_STARTUP_TIMEOUT", "1800")
    os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"
    stage_err_dir = "/tmp/sglang_omni_speech_stage_errors"
    os.environ["SGLANG_OMNI_STAGE_ERROR_DIR"] = stage_err_dir

    def dump_stage_errors() -> None:
        files = sorted(glob.glob(f"{stage_err_dir}/*.log"))
        if not files:
            print("[e2e] 没有 stage traceback 文件")
            return
        for path in files:
            print("=" * 70)
            print(f"[e2e] STAGE ERROR {path}")
            try:
                with open(path, encoding="utf-8") as handle:
                    print(handle.read())
            except Exception as exc:  # noqa: BLE001
                print(f"[e2e] traceback 读取失败: {exc}")

    def print_gpu_memory(label: str) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"[e2e] GPU memory ({label}): {result.stdout.strip() or result.stderr.strip()}")

    assert os.path.exists(f"{MODEL_DIR}/config.json"), (
        f"Volume 里没找到模型权重 {MODEL_DIR}/config.json"
    )

    multiprocessing.set_start_method("spawn", force=True)
    process = multiprocessing.Process(target=_serve_speech, daemon=False)
    process.start()
    base_url = "http://127.0.0.1:8000"

    try:
        print("[e2e] 等待完整 Thinker + Talker + Code2Wav 服务就绪……")
        for attempt in range(360):
            try:
                response = requests.get(f"{base_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"[e2e] 服务就绪，用时约 {attempt * 5}s")
                    break
            except requests.RequestException:
                pass
            if not process.is_alive():
                dump_stage_errors()
                raise SystemExit("[e2e] 服务进程提前退出")
            time.sleep(5)
        else:
            dump_stage_errors()
            raise SystemExit("[e2e] 等待健康检查超时")

        print_gpu_memory("ready")
        payload = {
            "model": "qwen3-omni",
            "messages": [
                {
                    "role": "user",
                    "content": "Say this short sentence: Hello from Qwen Omni.",
                }
            ],
            "modalities": ["text", "audio"],
            "audio": {"format": "wav"},
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": False,
        }
        print("[e2e] 发送唯一一条 text+audio 请求（max_tokens=16）……")
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=900,
        )
        if response.status_code != 200:
            print(f"[e2e] HTTP {response.status_code}: {response.text[:1200]}")
            time.sleep(3)
            dump_stage_errors()
            raise SystemExit(f"[e2e] HTTP request failed: {response.status_code}")

        message = response.json()["choices"][0]["message"]
        text = message.get("content") or ""
        audio = message.get("audio") or {}
        assert text.strip(), "Thinker 返回了空文本"
        assert audio.get("data"), "Talker/Code2Wav 没有返回 audio.data"

        wav_bytes = base64.b64decode(audio["data"], validate=True)
        assert wav_bytes[:4] in {b"RIFF", b"RF64"}, "返回内容不是 WAV 容器"
        waveform, sample_rate = sf.read(
            io.BytesIO(wav_bytes), dtype="float32", always_2d=False
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        frames = int(waveform.shape[0])
        duration_s = frames / float(sample_rate)
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))

        assert sample_rate > 0 and frames > 0, "WAV 没有有效采样"
        assert duration_s >= 0.05, f"WAV 太短: {duration_s:.4f}s"
        assert np.isfinite(rms) and rms > 1e-6, f"WAV 静音或数值异常: RMS={rms}"

        output_path = "/tmp/qwen3_omni_talker_smoke.wav"
        with open(output_path, "wb") as handle:
            handle.write(wav_bytes)
        print(f"[e2e] Thinker text: {text!r}")
        print(
            "[e2e] Talker WAV: "
            f"bytes={len(wav_bytes)}, sample_rate={sample_rate}, frames={frames}, "
            f"duration={duration_s:.3f}s, rms={rms:.6f}"
        )
        print(f"[e2e] 临时音频: {output_path}")
        print_gpu_memory("after request")
        print("[e2e] PASS —— Talker 生成语音，Code2Wav 输出有效 WAV")
        return {
            "wav_bytes": wav_bytes,
            "text": text,
            "sample_rate": int(sample_rate),
            "frames": frames,
            "duration_s": duration_s,
            "rms": rms,
        }
    finally:
        if process.is_alive():
            print("[e2e] 终止 speech 服务进程……")
            process.terminate()
            process.join(timeout=30)
        if process.is_alive():
            print("[e2e] 服务未及时退出，强制结束……")
            process.kill()
            process.join(timeout=10)


@app.local_entrypoint()
def main() -> None:
    result = e2e.remote()
    output_path = HERE / "talker_smoke_output.wav"
    output_path.write_bytes(result["wav_bytes"])
    print(f"[local] Thinker text: {result['text']!r}")
    print(
        "[local] Talker WAV 已保存: "
        f"{output_path} ({result['duration_s']:.3f}s, "
        f"{result['sample_rate']} Hz, RMS={result['rms']:.6f})"
    )
