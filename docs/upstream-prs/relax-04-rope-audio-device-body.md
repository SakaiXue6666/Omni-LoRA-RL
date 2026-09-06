# PR 4 — redai-infra/Relax

Branch: `SakaiXue6666:fix/rope-index-audio-seqlens-device` (based on main @ `f361a16`)

Link to open the PR:
https://github.com/redai-infra/Relax/compare/main...SakaiXue6666:Relax:fix/rope-index-audio-seqlens-device?expand=1

---

## Title

```
fix(qwen3-omni): keep audio_seqlens on CPU in get_rope_index
```

---

## Body (paste as-is)

## What

`get_rope_index` normalizes `audio_seqlens` onto the CPU before use, the same way the video
branches already do with `second_per_grids[i].cpu()`. Two lines, no behavior change for
single-audio sequences.

## Why

`get_rope_index` builds position ids on CPU with `torch.arange(...)`, and the running counters
`st` / `st_idx` accumulate each segment's length into that CPU chain. `audio_seqlens` arrives as a
CUDA tensor — callers derive it from `feature_attention_mask.sum(1)`, e.g.
`modeling_qwen3_omni/model.py`:

```python
audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)   # CUDA
...
position_ids, _ = get_rope_index(
    ...
    audio_seqlens=audio_feature_lengths,                           # no .cpu()
)
```

`_get_feat_extract_output_lengths` is pure arithmetic and preserves the device, so `audio_len`
stays on CUDA and contaminates the counter:

```python
audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_index])   # CUDA scalar
llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
st += text_len + bos_len + audio_len + eos_len                            # st becomes CUDA
```

On the **next** segment, `st_idx` comes from `llm_pos_ids_list[-1].max() + 1` (CPU) while
`text_len = min_ed - st` is now CUDA, and `st_idx += text_len` raises:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

**The first audio segment is always safe** — at that point `st` is still a plain int, so
`text_len` is an int too. This only fires once a sequence holds **two or more audio segments**,
which is why single-audio workloads never see it.

The asymmetry is visible in the function itself: both video branches guard with
`second_per_grids[video_index].cpu()`, and the audio branches do not.

## How it is reachable

`scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh` passes
`--multimodal-keys '{"image":"image","audio":"audio"}'`, where `audio` is a list-valued field —
nothing constrains it to a single clip. Any sample carrying two or more audio segments takes this
path, as does multi-turn audio interaction generally.

## Verification

Being upfront about what was and was not checked:

- **The mechanism is verified.** `_get_feat_extract_output_lengths` returns a tensor (not an int)
  when handed a tensor element, and `st` therefore becomes a tensor from the second iteration
  onward, inheriting whatever device `audio_seqlens` is on. This is reproducible on CPU.
- **The chain is traced link by link** through `model.py` → `get_rope_index` →
  `_get_feat_extract_output_lengths` → the counter update, on main @ `f361a16`.
- **I do not have a CUDA reproduction to show.** I have no GPU box available, and the failure is
  by definition a device mismatch, so I cannot demonstrate the `RuntimeError` itself here.

What I can say from experience is that we hit this exact failure — same message, same
multi-audio trigger, same first-segment-is-fine pattern — in the equivalent function in
`megatron.bridge`'s Qwen3-Omni model, under a multi-turn audio workload (simultaneous
translation, roughly ten audio chunks per sample). That is a different module from the vendored
copy here, so it is corroboration rather than proof; we patched this copy pre-emptively rather
than waiting to hit it again.

Happy to add a CUDA-gated regression test if you would like one — I left it out rather than ship
a test I cannot run.

## Risk

Minimal. `audio_seqlens` is read in exactly two places, both feeding
`_get_feat_extract_output_lengths` for a length that is then used to size `torch.arange(...)` and
to advance a CPU counter — both want CPU. Only the compute device changes: the lengths are
identical and `position_ids` is still returned on `input_ids.device`. Single-audio sequences,
which is the common path today, are bit-for-bit unaffected.

## Checklist

- [x] `ruff` line-length (119) respected; the addition is `#` comments, so the `docformatter`
      hook does not apply to it
- [ ] Regression test — see Verification above; happy to add one on request
