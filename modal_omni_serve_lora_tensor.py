"""Cost-controlled Qwen3-Omni tensor LoRA hot-update E2E on Modal.

This intentionally lives beside, rather than replacing,
``modal_omni_serve_lora.py``. The existing script remains the disk-adapter
regression baseline; this script exercises the Relax-compatible
``flattened_bucket`` HTTP data path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

APP_NAME = "sglang-omni-lora-tensor-e2e"
# v1 冻结镜像 = slime nightly-dev-20260428a（:latest 会漂移，认定过程见 README_v1.md）
BASE_IMAGE = os.environ.get(
    "RELAX_BASE_IMAGE",
    "slimerl/slime@sha256:bd219aba21be6e404ff09e385f34f40993b60773b928e13f341e8d77590da6aa",
)
MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_DIR = "/models/qwen3-omni"
MODEL_VOLUME_PATH = "/models"
PORT = 8000

HERE = Path(__file__).resolve().parent
SGLANG_DIR = HERE / "sglang" / "python"
OMNI_DIR = HERE / "sglang-omni"
SGLANG_REMOTE = "/root/sglang_src/python"
OMNI_REMOTE = "/root/sglang-omni"
PYTHONPATH = f"{SGLANG_REMOTE}:{OMNI_REMOTE}"

LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

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
        SGLANG_DIR.as_posix(),
        SGLANG_REMOTE,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        OMNI_DIR.as_posix(),
        OMNI_REMOTE,
        copy=True,
        ignore=["**/.git", "**/__pycache__", "**/*.pyc", "**/node_modules"],
    )
    .env({"PYTHONPATH": PYTHONPATH, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)
app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)


def _build_pipeline_config(
    thinker_tp_size: int = 1, *, speech: bool = False
) -> Any:
    from sglang_omni.cli.serve import (
        _apply_stage_server_args_override,
        apply_parallelism_cli_overrides,
        apply_thinker_server_args_cli_overrides,
    )

    if speech:
        if thinker_tp_size != 1:
            raise ValueError("the single-GPU colocated speech smoke requires TP=1")
        from sglang_omni.config.manager import ConfigManager

        config = ConfigManager.from_file(
            f"{OMNI_REMOTE}/examples/configs/qwen3_omni_colocated_h100_bf16.yaml"
        ).merge_config({"model_path": MODEL_DIR})
    else:
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig

        config = Qwen3OmniPipelineConfig(model_path=MODEL_DIR)
    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=thinker_tp_size,
        thinker_gpus=",".join(str(rank) for rank in range(thinker_tp_size)),
        talker_gpu=0 if speech else None,
        code2wav_gpu=0 if speech else None,
    )
    apply_thinker_server_args_cli_overrides(
        config,
        cpu_offload_gb=None,
        quantization=None,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        lora_target_modules="qkv_proj,o_proj",
        max_loras_per_batch=1,
    )
    updates = {
        "attention_backend": "triton",
        "disable_cuda_graph": True,
        "disable_custom_all_reduce": True,
        "disable_overlap_schedule": True,
    }
    if not speech:
        updates["mem_fraction_static"] = 0.85
    for stage_name in (("thinker", "talker_ar") if speech else ("thinker",)):
        _apply_stage_server_args_override(
            config,
            stage_name=stage_name,
            updates=updates,
            reason="tensor LoRA e2e Qwen3-Omni settings",
        )
    return config


def _serve_thinker(thinker_tp_size: int, speech: bool = False) -> None:
    from sglang_omni.serve import launch_server

    config = _build_pipeline_config(thinker_tp_size, speech=speech)
    launch_server(
        config,
        host="127.0.0.1",
        port=PORT,
        model_name="qwen3-omni",
    )


def _build_flattened_bucket_payload(seed: int) -> tuple[dict, int, int]:
    """Build the same pickle + base64 envelope emitted by Relax."""
    import pickle

    import pybase64
    import torch
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    thinker_config = getattr(config, "thinker_config", config)
    text_config = getattr(thinker_config, "text_config", thinker_config)

    num_layers = int(text_config.num_hidden_layers)
    hidden_size = int(text_config.hidden_size)
    num_heads = int(text_config.num_attention_heads)
    num_kv_heads = int(text_config.num_key_value_heads)
    head_dim = int(
        getattr(text_config, "head_dim", hidden_size // max(num_heads, 1))
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim

    tensors: dict[str, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        prefix = (
            "base_model.model.thinker.model.layers."
            f"{layer_idx}.self_attn"
        )
        tensors[f"{prefix}.qkv_proj.lora_A.weight"] = (
            torch.randn(
                LORA_RANK,
                hidden_size,
                generator=generator,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        tensors[f"{prefix}.qkv_proj.lora_B.weight"] = (
            torch.randn(
                qkv_dim,
                LORA_RANK,
                generator=generator,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        tensors[f"{prefix}.o_proj.lora_A.weight"] = (
            torch.randn(
                LORA_RANK,
                q_dim,
                generator=generator,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        tensors[f"{prefix}.o_proj.lora_B.weight"] = (
            torch.randn(
                hidden_size,
                LORA_RANK,
                generator=generator,
                dtype=torch.bfloat16,
            )
            * 0.02
        )

    bucket = FlattenedTensorBucket(named_tensors=list(tensors.items()))
    flattened_tensor_data = {
        "flattened_tensor": bucket.get_flattened_tensor(),
        "metadata": bucket.get_metadata(),
    }
    serialized_tensors = pybase64.b64encode(
        pickle.dumps(flattened_tensor_data)
    ).decode("utf-8")

    payload = {
        "lora_name": "policy",
        "serialized_tensors": serialized_tensors,
        "config_dict": {
            "peft_type": "LORA",
            "r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": 0.0,
            "target_modules": list(LORA_TARGET_MODULES),
            "bias": "none",
        },
        "load_format": "flattened_bucket",
        "pinned": False,
    }
    return payload, len(tensors), len(serialized_tensors)


@app.function(
    image=image,
    cpu=2.0,
    volumes={MODEL_VOLUME_PATH: model_volume},
    timeout=15 * 60,
)
def probe() -> None:
    """Validate the exact GPU image/config/payload path without allocating a GPU."""
    if not os.path.exists(f"{MODEL_DIR}/config.json"):
        raise FileNotFoundError(f"model volume is missing {MODEL_DIR}/config.json")
    config = _build_pipeline_config()
    speech_config = _build_pipeline_config(speech=True)
    payload, tensor_count, serialized_size = _build_flattened_bucket_payload(
        seed=20260717
    )
    assert payload["load_format"] == "flattened_bucket"
    assert payload["lora_name"] == "policy"
    assert tensor_count > 0 and serialized_size > 0
    speech_stages = {stage.name: stage for stage in speech_config.stages}
    assert speech_stages["thinker"].gpu == 0
    assert speech_stages["talker_ar"].gpu == 0
    assert speech_stages["code2wav"].gpu == 0
    print(f"pipeline={type(config).__name__}", flush=True)
    print(
        f"payload tensors={tensor_count}, base64_bytes={serialized_size}",
        flush=True,
    )
    print("PROBE PASS", flush=True)


@app.function(
    image=image,
    gpu="A100-80GB:1",
    volumes={MODEL_VOLUME_PATH: model_volume},
    timeout=30 * 60,
)
def e2e() -> None:
    _run_e2e(thinker_tp_size=1)


@app.function(
    image=image,
    gpu="A100-80GB:2",
    volumes={MODEL_VOLUME_PATH: model_volume},
    timeout=30 * 60,
)
def e2e_tp2() -> None:
    _run_e2e(thinker_tp_size=2)


@app.function(
    image=image,
    gpu="A100-80GB:1",
    volumes={MODEL_VOLUME_PATH: model_volume},
    timeout=30 * 60,
)
def e2e_speech() -> None:
    """Verify tensor hot-updates followed by a colocated Talker WAV request."""
    _run_e2e(thinker_tp_size=1, speech=True)


def _run_e2e(thinker_tp_size: int, *, speech: bool = False) -> None:
    import base64
    import glob
    import io
    import multiprocessing as mp
    import time

    import numpy as np
    import requests
    import soundfile as sf

    os.environ.setdefault("SGLANG_OMNI_STARTUP_TIMEOUT", "1800")
    os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"
    stage_error_dir = "/tmp/sglang_stage_errors"
    os.environ["SGLANG_OMNI_STAGE_ERROR_DIR"] = stage_error_dir

    print(
        "=== Tensor LoRA E2E: "
        f"{thinker_tp_size} x A100-80GB / TP={thinker_tp_size} / "
        "flattened_bucket ===",
        flush=True,
    )
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    if not os.path.exists(f"{MODEL_DIR}/config.json"):
        raise FileNotFoundError(f"model volume is missing {MODEL_DIR}/config.json")

    ctx = mp.get_context("spawn")
    server_process = ctx.Process(
        target=_serve_thinker,
        args=(thinker_tp_size, speech),
        daemon=False,
    )
    server_process.start()
    base_url = f"http://127.0.0.1:{PORT}"

    def _dump_stage_errors() -> None:
        for path in sorted(glob.glob(f"{stage_error_dir}/*.log")):
            print(f"stage error file: {path}", flush=True)
            try:
                with open(path, encoding="utf-8") as handle:
                    print(handle.read()[:12000], flush=True)
            except Exception as exc:
                print(f"failed to read stage error file: {exc}", flush=True)
        try:
            response = requests.get(
                f"{base_url}/debug/stage_errors",
                timeout=10,
            )
            print(
                "stage_errors:",
                response.status_code,
                response.text[:4000],
                flush=True,
            )
        except Exception as exc:
            print(f"stage_errors unavailable: {exc}", flush=True)

    def _generate(lora_name: str | None) -> tuple[str, list]:
        body = {
            "messages": [{"role": "user", "content": "只回答：今天天气很好。"}],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
            },
            "stream": False,
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 0,
        }
        if lora_name is not None:
            body["stage_params"] = {"thinker": {"lora_name": lora_name}}

        response = requests.post(f"{base_url}/generate", json=body, timeout=600)
        if response.status_code != 200:
            _dump_stage_errors()
            raise RuntimeError(
                f"generate({lora_name!r}) failed: "
                f"{response.status_code} {response.text[:4000]}"
            )
        result = response.json()
        logprobs = (result.get("meta_info") or {}).get("output_token_logprobs")
        if not logprobs:
            raise RuntimeError(f"generate({lora_name!r}) returned no logprobs")
        return str(result.get("text", "")), logprobs

    def _load_update(seed: int) -> dict:
        payload, tensor_count, serialized_size = _build_flattened_bucket_payload(
            seed
        )
        print(
            f"update seed={seed}: tensors={tensor_count}, "
            f"base64_bytes={serialized_size}",
            flush=True,
        )
        response = requests.post(
            f"{base_url}/load_lora_adapter_from_tensors",
            json=payload,
            timeout=300,
        )
        del payload
        if response.status_code != 200:
            _dump_stage_errors()
            raise RuntimeError(
                "tensor load failed: "
                f"{response.status_code} {response.text[:4000]}"
            )
        result = response.json()
        if not result.get("success"):
            raise RuntimeError(f"tensor load reported failure: {result}")
        return result

    try:
        for attempt in range(180):
            if not server_process.is_alive():
                raise RuntimeError(
                    f"server exited before readiness, exitcode={server_process.exitcode}"
                )
            try:
                if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                    print(f"server ready after {attempt * 5}s", flush=True)
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("server did not become healthy within 15 minutes")

        base_text, base_logprobs = _generate(None)
        print(f"base text={base_text!r}", flush=True)
        print(f"base logprobs={base_logprobs}", flush=True)

        first_load = _load_update(seed=20260717)
        print(f"first tensor load={first_load}", flush=True)
        first_text, first_logprobs = _generate("policy")
        print(f"update-1 text={first_text!r}", flush=True)
        print(f"update-1 logprobs={first_logprobs}", flush=True)

        unload_response = requests.post(
            f"{base_url}/unload_lora_adapter",
            json={"lora_name": "policy"},
            timeout=180,
        )
        if unload_response.status_code != 200:
            raise RuntimeError(
                "unload failed: "
                f"{unload_response.status_code} {unload_response.text[:4000]}"
            )
        unload_result = unload_response.json()
        if not unload_result.get("success"):
            raise RuntimeError(f"unload reported failure: {unload_result}")
        print(f"unload={unload_result}", flush=True)

        second_load = _load_update(seed=20260718)
        print(f"second tensor load={second_load}", flush=True)
        second_text, second_logprobs = _generate("policy")
        print(f"update-2 text={second_text!r}", flush=True)
        print(f"update-2 logprobs={second_logprobs}", flush=True)

        base_changed = base_logprobs != first_logprobs
        update_changed = first_logprobs != second_logprobs
        print(f"base_vs_update_1_changed={base_changed}", flush=True)
        print(f"update_1_vs_update_2_changed={update_changed}", flush=True)
        if not base_changed:
            raise RuntimeError("first tensor LoRA update did not change token logprobs")
        if not update_changed:
            raise RuntimeError("second tensor LoRA update did not change token logprobs")

        if speech:
            request_id = "tensor-lora-speech-smoke"
            speech_response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "request_id": request_id,
                    "model": "qwen3-omni",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Say this short sentence: Hello after tensor update.",
                        }
                    ],
                    "modalities": ["text", "audio"],
                    "audio": {"format": "wav"},
                    "max_tokens": 16,
                    "temperature": 0.0,
                    "stream": False,
                    "stage_params": {"thinker": {"lora_name": "policy"}},
                },
                timeout=900,
            )
            if speech_response.status_code != 200:
                _dump_stage_errors()
                raise RuntimeError(
                    "post-update speech failed: "
                    f"{speech_response.status_code} {speech_response.text[:4000]}"
                )
            message = speech_response.json()["choices"][0]["message"]
            speech_text = str(message.get("content") or "")
            audio = message.get("audio") or {}
            wav_bytes = base64.b64decode(audio.get("data") or "", validate=True)
            if not speech_text.strip() or wav_bytes[:4] not in {b"RIFF", b"RF64"}:
                raise RuntimeError("post-update speech returned empty text or invalid WAV")
            waveform, sample_rate = sf.read(
                io.BytesIO(wav_bytes), dtype="float32", always_2d=False
            )
            waveform = np.asarray(waveform, dtype=np.float32)
            frames = int(waveform.shape[0])
            duration_s = frames / float(sample_rate)
            rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
            if (
                sample_rate <= 0
                or frames <= 0
                or duration_s < 0.05
                or not np.isfinite(rms)
                or rms <= 1e-6
            ):
                raise RuntimeError(
                    "post-update WAV is empty or invalid: "
                    f"rate={sample_rate} frames={frames} duration={duration_s} rms={rms}"
                )
            print(f"speech text={speech_text!r}", flush=True)
            print(
                "speech WAV: "
                f"bytes={len(wav_bytes)} rate={sample_rate} frames={frames} "
                f"duration={duration_s:.3f}s rms={rms:.6f}",
                flush=True,
            )

        print("PASS", flush=True)
    finally:
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=30)
        if server_process.is_alive():
            server_process.kill()
            server_process.join(timeout=5)
        print(
            f"server cleanup complete, exitcode={server_process.exitcode}",
            flush=True,
        )


@app.local_entrypoint()
def main() -> None:
    e2e.remote()
