"""探针九 —— 复现 adapter 跨进程传输的失败，并验证 v1 的绕法（纯 CPU，几分钟）。

背景：4×A100 的真机训练跑到第一次权重同步时，基础权重（CUDA 张量，走 CUDA IPC）
同步成功，adapter（CPU 张量）推给 sglang 时四个 TP rank 里成了三个，TP0 报：

    RuntimeError: unable to open shared memory object </torch_4103_...>
    in read-write mode: No such file or directory

上游 Relax 用 `MultiprocessingSerializer.serialize(tensors, output_str=True)`。CPU 张量
经它走的是 torch 的 file_system 共享内存策略：payload 里只装文件名，消费者自己去
/dev/shm 映射，因此依赖生产者进程在那一刻仍持有存储。

v1 当年明确绕开了这条路，`update_lora_from_tensor.py` 里写着：

    # 不使用 MultiprocessingSerializer：它可能通过 fd sharing 传 CPU Tensor，
    # 跨 Ray actor 进程树时可能因 authkey 不同触发 AuthenticationError。
    serialized = pybase64.b64encode(pickle.dumps(flattened_tensor_data)).decode("utf-8")

也就是把字节直接内联进 RPC payload，完全不碰共享内存。

这个探针把两种序列化摆在一起对照，形状照真机：一个生产者 actor 序列化一次，四个
消费者 actor（对应 TP0-3）各自反序列化同一份 payload，生产者全程持有张量引用。
判据是 A/B —— MultiprocessingSerializer 复现失败、pickle 内联全过，才算定位准确。

    modal run modal_probe_transport.py
"""

from __future__ import annotations

import os

import modal


RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

SGLANG_FORK = "https://github.com/SakaiXue6666/sglang.git"
SGLANG_REF = os.environ.get("SGLANG_V2_REF", "lora-omni-v2")
SGLANG_SRC = "/root/sglang_fork"

PYTHONPATH = f"{SGLANG_SRC}/python:/root/Megatron-LM:/pkg:/root:/sgl-workspace/sglang/python"

image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .run_commands(
        f"git clone --filter=blob:none --branch {SGLANG_REF} {SGLANG_FORK} {SGLANG_SRC}",
        "pip install --no-cache-dir pybase64",
    )
    .env({"PYTHONPATH": PYTHONPATH})
)

app = modal.App("v2-adapter-transport-probe")


@app.function(image=image, cpu=8.0, timeout=20 * 60)
def probe() -> int:
    import pickle
    import traceback

    import pybase64
    import ray
    import torch

    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    print(f"torch 共享策略: {torch.multiprocessing.get_sharing_strategy()}", flush=True)

    ray.init(num_cpus=8, include_dashboard=False, log_to_driver=True)

    @ray.remote(num_cpus=1)
    class Consumer:
        """对应 sglang 的一个 TP worker：装 reducer，然后反序列化。"""

        def __init__(self, rank: int):
            self.rank = rank

        def load(self, payload: str, how: str, delay: float = 0.0) -> dict:
            # tp_worker.load_lora_adapter_from_tensors 里就是这么做的
            monkey_patch_torch_reductions()
            # 真机上 TP0 的 scheduler 还在忙别的，比其他 rank 晚到。这里显式模拟它迟到。
            if delay:
                import time

                time.sleep(delay)
            try:
                if how in ("mp", "mp_fs"):
                    tensors = MultiprocessingSerializer.deserialize(payload)
                else:
                    tensors = pickle.loads(pybase64.b64decode(payload))
                checksum = float(sum(t.float().sum().item() for t in tensors.values()))
                return {"rank": self.rank, "ok": True, "n": len(tensors), "checksum": checksum}
            except Exception as e:  # noqa: BLE001
                return {
                    "rank": self.rank,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc().splitlines()[-3:],
                }

    @ray.remote(num_cpus=1)
    class Producer:
        """对应 Relax 的训练 rank 0：造 adapter 张量、序列化、同步调用四个消费者。"""

        def __init__(self, delay: float = 0.0):
            self.delay = delay
            # 形状照 Qwen3-Omni rank 16 的 adapter：48 层 × q/k/v/o × lora_A/B
            self.tensors = {}
            for layer in range(48):
                for proj, out in (("q_proj", 4096), ("k_proj", 512), ("v_proj", 512), ("o_proj", 2048)):
                    base = f"base_model.model.thinker.model.layers.{layer}.self_attn.{proj}"
                    self.tensors[f"{base}.lora_A.weight"] = torch.randn(16, 2048, dtype=torch.bfloat16)
                    self.tensors[f"{base}.lora_B.weight"] = torch.randn(out, 16, dtype=torch.bfloat16)
            self.expected = float(sum(t.float().sum().item() for t in self.tensors.values()))
            nbytes = sum(t.numel() * t.element_size() for t in self.tensors.values())
            print(
                f"[producer] {len(self.tensors)} 个张量，共 {nbytes / 1e6:.1f} MB，"
                f"校验和 {self.expected:.3f}",
                flush=True,
            )

        def run(self, consumers, how: str) -> list:
            from torch.multiprocessing import get_sharing_strategy, set_sharing_strategy

            prev = get_sharing_strategy()
            # mp_fs 完全照抄上游 update_weight_from_tensor.py 的写法：切 file_system，
            # 序列化和整个同步加载都在这个策略下进行，结束才还原。
            if how == "mp_fs":
                set_sharing_strategy("file_system")
            try:
                if how in ("mp", "mp_fs"):
                    payload = MultiprocessingSerializer.serialize(self.tensors, output_str=True)
                else:
                    payload = pybase64.b64encode(pickle.dumps(self.tensors)).decode("utf-8")
                print(
                    f"[producer] {how}（策略 {get_sharing_strategy()}）: payload {len(payload) / 1e6:.1f} MB",
                    flush=True,
                )

                # 生产者全程持有 self.tensors，与 Relax 里那句"keep tensors alive"一致
                # 只让 TP0 迟到，其余立刻进来并在返回时释放各自的张量
                results = ray.get(
                    [c.load.remote(payload, how, self.delay if i == 0 else 0.0) for i, c in enumerate(consumers)]
                )
            finally:
                set_sharing_strategy(prev)
            for r in results:
                r["expected"] = self.expected
            return results

    delay = float(os.environ.get("TP0_DELAY", "8"))
    print(f"TP0 迟到 {delay} 秒（其余 rank 立刻进来并释放）", flush=True)
    producer = Producer.remote(delay)
    consumers = [Consumer.remote(i) for i in range(4)]

    failures = 0
    modes = (
        ("mp", "MultiprocessingSerializer + 默认 file_descriptor"),
        ("mp_fs", "MultiprocessingSerializer + file_system（上游实际用的）"),
        ("pickle", "pickle+base64（v1 用的）"),
    )
    for how, label in modes:
        print(f"\n========== {label} ==========", flush=True)
        results = ray.get(producer.run.remote(consumers, how))
        for r in results:
            if r["ok"]:
                drift = abs(r["checksum"] - r["expected"])
                verdict = "值对得上" if drift < 1.0 else f"值对不上，差 {drift:.3f}"
                print(f"  TP{r['rank']}: 成功，{r['n']} 个张量，{verdict}", flush=True)
            else:
                print(f"  TP{r['rank']}: 失败 —— {r['error']}", flush=True)
                for line in r["trace"]:
                    print(f"           {line.strip()}", flush=True)

        n_bad = sum(1 for r in results if not r["ok"])
        if how in ("mp", "mp_fs"):
            if n_bad:
                print(f"\n  复现成功：4 个 rank 里 {n_bad} 个失败，与真机现象一致", flush=True)
            else:
                print("\n  没复现出来 —— 说明触发条件还不止跨进程这一条", flush=True)
                failures += 1
        else:
            if n_bad:
                print(f"\n  绕法也失败了 {n_bad} 个，v1 的做法在这里不成立", flush=True)
                failures += 1
            else:
                print("\n  绕法全过：内联字节不碰共享内存，四个 rank 都拿到了正确的值", flush=True)

    ray.shutdown()
    return failures


@app.local_entrypoint()
def main() -> None:
    rc = probe.remote()
    print(f"\n[probe] 判定失败项 = {rc}")
