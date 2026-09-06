## What

The adapter push in LoRA adapter mode now carries the tensor bytes in the payload (`serialize_adapter_tensors`) instead of a reference to shared host memory. The call site loses its sharing-strategy dance and its keep-alive requirement along with it.

## Why

A colocate run of Qwen3-Omni-30B-A3B thinker LoRA (rank 16, TP4, 4×A100) gets through model build, LoRA injection, engine startup and the full base-weight sync (19743 parameters, 7.7s), then dies on the first adapter push — but only on one rank. TP1–TP3 log `loading from tensors completes`; TP0 raises:

```
RuntimeError: unable to open shared memory object </torch_4103_2908601601_198>
in read-write mode: No such file or directory
```

`4103` is the pid of training rank 0, the producer.

The payload built under `file_system` does not contain the adapter; it contains a `/dev/shm` file name. That storage is reference counted, and the count is what the *consumers* hold: the ranks that map it first release their reference when the call returns, the count reaches zero, the file is unlinked, and a rank that arrives late opens a path that no longer exists. Nothing is wrong with the adapter, the LoRA scope, or the engine — the bytes simply expire in flight.

Being timing dependent, it presents as one rank dying while its peers load the very same adapter, which reads like a rank-local fault and is not one.

The current comment in this code records that the default `file_descriptor` strategy was already found unable to cross the Ray → HTTP hops, and `file_system` was adopted in response. That diagnosis is right; `file_system` is simply the next pothole on the same road, because both strategies share the property that matters here — the payload is a reference, and a reference is only as good as the producer's grip on the storage.

A CPU-only probe reproduces all three transports end to end (one Ray actor serializes, four consumer actors deserialize, rank 0 arriving late by 8s):

| Transport | Result |
| --- | --- |
| `file_descriptor` (torch default) | all four ranks fail with `AuthenticationError` |
| `file_system` (current) | rank 0 fails with the ENOENT above, ranks 1–3 succeed |
| Inlined pickle (this PR) | all four ranks succeed, checksums match |

Worth stating plainly: with all four consumers arriving simultaneously, `file_system` passes. The race only materialises when one rank is late — which is precisely why it survived into production and why it should not be left to scheduling luck.

## How

`serialize_adapter_tensors` pickles the tensors and base64-encodes them, which is the exact wire format SGLang already expects: `MultiprocessingSerializer.deserialize` base64-decodes and unpickles, and a plain pickle simply has no reference to resolve. No SGLang-side change is needed.

The base-weight path is deliberately untouched. Those tensors stay on device and serialize to CUDA IPC handles, which are self-contained across these hops — which is also why base sync never failed while the adapter push did.

The cost is the payload: 0.1MB of handles becomes 31.6MB of base64 for a rank-16 adapter on a 30B model, once per weight update, on a path that already does a cross-process gather. Adapters are bounded by rank rather than model size, so this stays small where it matters.

## Testing

Run inside the Relax training image (`sglang 0.5.12.post1`, `torch 2.11.0+cu129`):

```
pytest tests/utils/test_megatron_peft_utils.py::TestSerializeAdapterTensors
3 passed
```

The three cases pin the property that broke, not the implementation: the payload round-trips through plain base64 + unpickle, it is at least as large as the tensors it carries, and it contains no `/torch_` handle. Substituting the shared-memory serialization back into the helper fails the latter two — `assert 530 >= 524288`, and the handle found verbatim in the payload — so the tests would have caught this before it reached a GPU.

End to end, a 40-step GRPO run (Qwen3-Omni-30B-A3B thinker LoRA, 4×A100 colocate, TP4/EP4, adapter pushed every step) completed 40/40 pushes with reward rising from 0.294 over the first ten steps to 0.392 over the last ten.

`pre-commit run --files` is clean on all three touched files.

- [x] `pre-commit run --all-files` passes
- [x] Tests pass (`pytest tests/`)
- [x] New tests added (if applicable)
- [ ] Documentation updated (if applicable)

## Type of Change

- [x] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement
- [ ] CI/CD or build changes

## Screenshots / Logs

Production failure, four TP workers of one engine:

```
[TP1] loading from tensors completes
[TP2] loading from tensors completes
[TP3] loading from tensors completes
[TP0] RuntimeError: unable to open shared memory object </torch_4103_2908601601_198>
      in read-write mode: No such file or directory
```

Same chain reproduced on CPU, with and without the fix:

```
[file_system]      rank0 FAIL: unable to open shared memory object </torch_367_...>
                   rank1 ok  rank2 ok  rank3 ok
[inlined pickle]   rank0 ok  rank1 ok  rank2 ok  rank3 ok   (checksums match)
```
