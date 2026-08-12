## Motivation

`monkey_patch_torch_reductions()` installs `_reduce_tensor_modified` as `reductions.reduce_tensor` and calls `init_reductions()`, which rebinds the `ForkingPickler` dispatch for `torch.Tensor` — every tensor in the process, not only the CUDA ones. The replacement rewrites argument 6 unconditionally:

```python
output_args = _modify_tuple(
    output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid
)
```

Only the CUDA reduction carries a device index there. A CPU tensor reduces to a shorter tuple with no device slot at all, so `_modify_tuple` indexes past the end:

```
IndexError: tuple index out of range
  File "sglang/srt/utils/patch_torch.py", line 103, in _modify_tuple
    return *t[:index], modifier(t[index]), *t[index + 1 :]
```

It surfaces from inside `ForkingPickler`, which makes it read as a serialization bug rather than anything device related.

Anything that sends a CPU tensor through torch multiprocessing in a patched process hits this. The path that found it is the RL weight-update surface this patch exists for: `update_weights_from_tensor` and `load_lora_adapter_from_tensors` accept whatever the trainer serializes, and LoRA adapters in particular are naturally staged on the host — they are small, they are gathered across TP ranks before the push, and they do not need to occupy device memory in the meantime.

This is not specific to one trainer. [verl#4065](https://github.com/volcengine/verl/issues/4065), open since November 2025 with several independent "same bug" reports, is the identical traceback through `_reduce_tensor_modified` -> `_modify_tuple`, and the thread converges on the same diagnosis ("LoRA weights being kept on the CPU") and circulates this same arity guard as a local patch. Users are editing `patch_torch.py` in site-packages today, or steering to a merge-the-adapter path to avoid pushing host tensors at all.

The constant already documents its own fragility:

```python
# The signature has not been changed for years, and we will not need this when the next version is released,
# so it looks safe to use a constant.
_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6
```

The assumption that holds is about the *position* of the device index. What does not hold is that a device index is present at all.

## Modifications

Guard the rewrite on the argument count:

```python
if len(output_args) > _REDUCE_TENSOR_ARG_DEVICE_INDEX:
    output_args = _modify_tuple(...)
```

CUDA tensors are unaffected — their reduced form is long enough and the device slot is still rewritten to a UUID, which is the entire point of the patch. CPU tensors pass through to the original reducer's output untouched.

Arity was chosen over an `is_cuda` check on the input tensor because it tests the actual precondition (`output_args` has a slot at index 6) rather than a proxy for it, and it therefore also covers any other non-CUDA reduced form reaching this function.

Added `test/registered/unit/utils/test_patch_torch_cpu_tensor.py` with two cases: a CPU tensor round-trips through `MultiprocessingSerializer` after patching, and — so the guard cannot silently disarm the patch — a CUDA-shaped argument tuple still gets its device index rewritten, verified with a mocked original reducer so the test stays on CPU CI. The first fails with the `IndexError` above without this change; the second passes either way.

## Accuracy Tests

Not applicable: no kernel or model forward code is touched. Behavior for CUDA tensors is unchanged by construction, and the second test pins that.

## Speed Tests and Profiling

Not applicable. The change adds one length comparison per tensor reduction.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
