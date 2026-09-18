# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""torch stand-ins for the triton kernels of upstream's V2 model runner.

Every `@triton.jit` kernel under `vllm/v1/worker/gpu/` is a module global
that its python wrapper launches as `kernel[grid](*args, **constexpr)`. The
patches below replace those globals with `_TorchKernel` objects whose
`__getitem__` returns the torch implementation, so every wrapper -- and every
module that imported a wrapper by name -- runs the torch code. Each torch
function takes `(grid, *args)` in the kernel's own argument order, receives
the tensor where the kernel received a pointer, and vectorizes over the grid.
"""

import functools
import types
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput, AsyncPoolingOutput
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import (
    FusedStagedWriter,
    StagedWriteTensor,
    UvaBufferPool,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import (
    PendingRecv,
    PPHandler,
    compute_need_sampled_mask,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

from vllm_rbln.patches.registry import register_patch

_NO_TRITON_REASON = (
    "Upstream's V2 model runner prepares inputs, updates request state and "
    "samples with eager triton kernels, and has no hook to substitute them. "
    "Triton has no RBLN backend, so those kernels cannot run on an RBLN "
    "tensor whether or not triton itself imports."
)

_NO_CUDA_REASON = (
    "Upstream's V2 model runner moves results host-side on a CUDA copy stream "
    "and reads them behind a CUDA event, with no non-CUDA path. RBLN has no "
    "torch.cuda streams, so the copy runs synchronously instead."
)


def _placeholder_triton() -> bool:
    """The stand-in module vLLM installs when triton does not import."""
    return not HAS_TRITON


def _runner_device() -> torch.device:
    return torch.device(current_platform.device_type)


class _TorchKernel:
    """`kernel[grid](*args, **constexpr)` -> `fn(grid, *args, **constexpr)`."""

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, grid):
        if not isinstance(grid, tuple):
            grid = (grid,)
        return functools.partial(self._fn, grid)


def _patch_kernel(module: str, name: str, fn) -> None:
    register_patch(
        target=f"{module}.{name}",
        key=f"{__name__}.{name}",
        owner_module=__name__,
        reason=_NO_TRITON_REASON,
    )(_TorchKernel(fn))


def _expand(cu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per element of a `[cu[0], cu[-1])` range: its segment and offset in it."""
    lens = cu[1:] - cu[:-1]
    seg = torch.repeat_interleave(
        torch.arange(lens.shape[0], device=cu.device),
        lens,
        output_size=int(cu[-1] - cu[0]),
    )
    off = torch.arange(int(cu[-1] - cu[0]), device=cu.device) - (cu[:-1] - cu[0])[seg]
    return seg, off


def _unpack_bits(packed: torch.Tensor, width: int) -> torch.Tensor:
    """int32 `[rows, ceil(width/32)]` -> bool `[rows, width]`, LSB first."""
    shifts = torch.arange(32, device=packed.device, dtype=torch.int32)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :width].bool()


# --------------------------------------------------------------------------
# config: upstream ties the V2 runner to triton
# --------------------------------------------------------------------------


def _validate_v2_model_runner(self: VllmConfig) -> None:
    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        raise ValueError(
            f"Model Runner V2 does not yet support: {', '.join(unsupported)}"
        )


register_patch(
    target="vllm.config.vllm.VllmConfig._validate_v2_model_runner",
    reason=(
        "VllmConfig._validate_v2_model_runner refuses VLLM_USE_V2_MODEL_RUNNER=1 "
        "without triton because upstream's V2 kernels are triton. RBLN runs "
        "them as torch (this module), so only the feature checks apply."
    ),
)(_validate_v2_model_runner)


# --------------------------------------------------------------------------
# triton placeholder helpers
# --------------------------------------------------------------------------

register_patch(
    target="vllm.triton_utils.triton.next_power_of_2",
    key=f"{__name__}.next_power_of_2",
    owner_module=__name__,
    reason=(
        "Wrappers in vllm/v1/worker/gpu/ size their launch with "
        "triton.next_power_of_2 before the kernel object is touched; the "
        "TritonPlaceholder module has cdiv but not that helper."
    ),
    condition=_placeholder_triton,
)(next_power_of_2)


# --------------------------------------------------------------------------
# buffer_utils: UVA buffers and staged writes
# --------------------------------------------------------------------------


class _HostBuffer:
    """Host tensor plus a device mirror; UVA has no equivalent on RBLN."""

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        device = _runner_device()
        self.uva = (
            self.cpu
            if device.type == "cpu"
            else torch.zeros(size, dtype=dtype, device=device)
        )


register_patch(
    target="vllm.v1.worker.gpu.buffer_utils.UvaBuffer",
    reason=(
        "UvaBuffer pins host memory and maps it into the device address space "
        "(is_uva_available), which upstream exposes only for CUDA-alike and "
        "XPU. RBLN has no UVA, so the buffer is a host tensor with an "
        "explicit device mirror."
    ),
)(_HostBuffer)


def _copy_to_uva(
    self: UvaBufferPool, x: torch.Tensor | np.ndarray | list
) -> torch.Tensor:
    self._curr = (self._curr + 1) % self.max_concurrency
    buf = self._uva_bufs[self._curr]
    dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
    n = len(x)
    dst[:n] = x
    if buf.uva is not buf.cpu:
        buf.uva[:n].copy_(buf.cpu[:n])
    return buf.uva[:n]


register_patch(
    target="vllm.v1.worker.gpu.buffer_utils.UvaBufferPool.copy_to_uva",
    reason=(
        "UvaBufferPool.copy_to_uva writes the host half of a UVA buffer and "
        "returns the device view of the same memory. Without UVA the device "
        "mirror has to be written explicitly."
    ),
)(_copy_to_uva)


def _apply_write(self: StagedWriteTensor) -> None:
    n = len(self._staged_write_indices)
    if n == 0:
        return
    device = self.gpu.device
    idx = torch.tensor(self._staged_write_indices, dtype=torch.int64)
    starts = torch.tensor(self._staged_write_starts, dtype=torch.int64)
    cu = torch.tensor(self._staged_write_cu_lens, dtype=torch.int64)
    total = int(cu[-1])
    write, off = _expand(torch.cat([cu.new_zeros(1), cu]))
    flat_idx = idx[write] * self.gpu.stride(0) + starts[write] + off
    contents = torch.tensor(self._staged_write_contents, dtype=self.dtype)
    assert contents.shape[0] == total
    self.gpu.view(-1)[flat_idx.to(device)] = contents.to(device)
    self.clear_staged_writes()


register_patch(
    target="vllm.v1.worker.gpu.buffer_utils.StagedWriteTensor.apply_write",
    reason=(
        "apply_write flushes staged row writes with _apply_write_kernel, "
        "handing the kernel the destination as a raw pointer table; a torch "
        "scatter needs the tensor itself, so the method is replaced rather "
        "than the kernel."
    ),
)(_apply_write)


def _fused_apply(
    self: FusedStagedWriter,
    tensors: Sequence[StagedWriteTensor],
    output_ptrs: torch.Tensor,
    output_strides: torch.Tensor,
) -> None:
    for t in tensors:
        t.apply_write()


register_patch(
    target="vllm.v1.worker.gpu.buffer_utils.FusedStagedWriter.apply",
    reason=(
        "FusedStagedWriter.apply addresses each destination through a uint64 "
        "data_ptr table (_load_ptr) inside one kernel launch; torch cannot "
        "dereference raw pointers, so each tensor flushes its own writes."
    ),
)(_fused_apply)


# --------------------------------------------------------------------------
# block_table: gather and slot mapping address block tables by data_ptr
# --------------------------------------------------------------------------


def _gather_block_tables(
    self: BlockTables,
    idx_mapping: torch.Tensor,
    num_reqs_padded: int,
    out: tuple[torch.Tensor, ...] | None = None,
    out_ptrs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if out is None:
        out = tuple(self.input_block_tables)
    num_reqs = idx_mapping.shape[0]
    src_rows = idx_mapping.long()
    for group_id in range(self.num_kv_cache_groups):
        dst = out[group_id]
        dst[:num_reqs] = self.block_tables[group_id].gpu[src_rows]
        dst[num_reqs:num_reqs_padded].zero_()
    return tuple(bt[:num_reqs_padded] for bt in out)


register_patch(
    target="vllm.v1.worker.gpu.block_table.BlockTables.gather_block_tables",
    reason=(
        "gather_block_tables launches _gather_block_tables_kernel with the "
        "block tables as a uint64 data_ptr table; torch cannot dereference "
        "raw pointers, so the method indexes the tensors directly."
    ),
)(_gather_block_tables)


def _compute_slot_mappings(
    self: BlockTables,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens_padded: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert self.cp_size == 1, "context parallelism is not supported on RBLN"
    slot_mappings = self.slot_mappings if out is None else out
    num_reqs = idx_mapping.shape[0]
    qsl = query_start_loc[: num_reqs + 1].long()
    num_tokens = int(qsl[-1])
    tok_req, _ = _expand(qsl)
    rows = idx_mapping.long()[tok_req]
    pos = positions[:num_tokens].long()
    for group_id in range(self.num_kv_cache_groups):
        block_size = self.kernel_block_sizes[group_id]
        block_table = self.block_tables[group_id].gpu
        block_numbers = block_table[rows, pos // block_size].long()
        slot_mappings[group_id, :num_tokens] = (
            block_numbers * block_size + pos % block_size
        )
        slot_mappings[group_id, num_tokens:] = PAD_SLOT_ID
    return slot_mappings[:, :num_tokens_padded]


register_patch(
    target="vllm.v1.worker.gpu.block_table.BlockTables.compute_slot_mappings",
    reason=(
        "compute_slot_mappings launches _compute_slot_mappings_kernel with the "
        "block tables as a uint64 data_ptr table; torch cannot dereference "
        "raw pointers, so the method indexes the tensors directly."
    ),
)(_compute_slot_mappings)


# --------------------------------------------------------------------------
# input_batch kernels
# --------------------------------------------------------------------------

_INPUT_BATCH = "vllm.v1.worker.gpu.input_batch"


def _prepare_prefill_inputs(
    grid,
    input_ids,
    next_prefill_tokens,
    idx_mapping,
    query_start_loc,
    all_token_ids,
    _all_token_ids_stride,
    prefill_lens,
    num_computed_tokens,
    **_,
) -> None:
    num_reqs = grid[0]
    req = idx_mapping[:num_reqs].long()
    qsl = query_start_loc[: num_reqs + 1].long()
    query_len = qsl[1:] - qsl[:-1]
    num_computed = num_computed_tokens[req].long()
    prefill_len = prefill_lens[req].long()
    is_prefill = num_computed < prefill_len
    if not bool(is_prefill.any()):
        return
    tok_req, tok_off = _expand(qsl)
    keep = is_prefill[tok_req]
    rows = req[tok_req][keep]
    cols = (num_computed[tok_req] + tok_off)[keep]
    dst = (torch.arange(tok_req.shape[0], device=qsl.device) + qsl[0])[keep]
    input_ids[dst] = all_token_ids[rows, cols].to(input_ids.dtype)

    next_pos = num_computed + query_len
    update = is_prefill & (next_pos < prefill_len)
    if bool(update.any()):
        rows = req[update]
        next_prefill_tokens[rows] = all_token_ids[rows, next_pos[update]].to(
            next_prefill_tokens.dtype
        )


_patch_kernel(_INPUT_BATCH, "_prepare_prefill_inputs_kernel", _prepare_prefill_inputs)


def _prepare_pos_seq_lens(
    grid,
    pos,
    seq_lens,
    idx_mapping,
    query_start_loc,
    num_computed_tokens,
    max_num_reqs,
    **_,
) -> None:
    num_reqs = grid[0] - 1
    seq_lens[num_reqs:max_num_reqs].zero_()
    req = idx_mapping[:num_reqs].long()
    qsl = query_start_loc[: num_reqs + 1].long()
    num_computed = num_computed_tokens[req].long()
    seq_lens[:num_reqs] = (num_computed + qsl[1:] - qsl[:-1]).to(seq_lens.dtype)
    tok_req, tok_off = _expand(qsl)
    pos[qsl[0] : qsl[-1]] = (num_computed[tok_req] + tok_off).to(pos.dtype)


_patch_kernel(_INPUT_BATCH, "_prepare_pos_seq_lens_kernel", _prepare_pos_seq_lens)


def _combine_sampled_and_draft_tokens(
    grid,
    input_ids,
    idx_mapping,
    last_sampled_tokens,
    query_start_loc,
    seq_lens,
    prefill_len,
    draft_tokens,
    _draft_tokens_stride,
    cu_num_logits,
    logits_indices,
    NUM_NEW_SAMPLED_TOKENS: int = 1,
    **_,
) -> None:
    num_reqs = grid[0]
    req = idx_mapping[:num_reqs].long()
    qsl = query_start_loc[: num_reqs + 1].long()
    cu = cu_num_logits[: num_reqs + 1].long()
    num_logits = cu[1:] - cu[:-1]
    num_draft = num_logits - NUM_NEW_SAMPLED_TOKENS
    query_end = qsl[1:]
    logits_start = query_end - num_logits

    logit_req, logit_off = _expand(cu)
    logits_indices[cu[0] : cu[-1]] = (logits_start[logit_req] + logit_off).to(
        logits_indices.dtype
    )

    seq_len = seq_lens[:num_reqs].long()
    prefill = prefill_len[req].long()
    generating = seq_len > prefill
    if NUM_NEW_SAMPLED_TOKENS > 0:
        write_last = generating & (seq_len - num_logits >= prefill)
        if bool(write_last.any()):
            input_ids[logits_start[write_last]] = last_sampled_tokens.view(-1)[
                req[write_last]
            ].to(input_ids.dtype)

    has_draft = generating & (num_draft > 0)
    if bool(has_draft.any()):
        draft_lens = torch.where(has_draft, num_draft, torch.zeros_like(num_draft))
        cu_draft = torch.cat([draft_lens.new_zeros(1), torch.cumsum(draft_lens, 0)])
        d_req, d_off = _expand(cu_draft)
        dst = query_end[d_req] - draft_lens[d_req] + d_off
        input_ids[dst] = draft_tokens[req[d_req], d_off].to(input_ids.dtype)


_patch_kernel(
    _INPUT_BATCH,
    "_combine_sampled_and_draft_tokens_kernel",
    _combine_sampled_and_draft_tokens,
)


def _get_num_sampled_and_rejected(
    grid,
    num_sampled,
    num_rejected,
    seq_lens,
    cu_num_logits,
    idx_mapping,
    prefill_len,
    **_,
) -> None:
    num_reqs = grid[0]
    req = idx_mapping[:num_reqs].long()
    chunked = seq_lens[:num_reqs].long() < prefill_len[req].long()
    cu = cu_num_logits[: num_reqs + 1].long()
    sampled = torch.where(
        chunked, torch.zeros_like(num_sampled[:num_reqs]), num_sampled[:num_reqs]
    )
    num_sampled[:num_reqs] = sampled
    rejected = (cu[1:] - cu[:-1]).to(num_rejected.dtype) - sampled.to(
        num_rejected.dtype
    )
    num_rejected[:num_reqs] = torch.where(chunked, torch.zeros_like(rejected), rejected)


_patch_kernel(
    _INPUT_BATCH, "_get_num_sampled_and_rejected_kernel", _get_num_sampled_and_rejected
)


def _post_update(
    grid,
    idx_mapping,
    num_computed_tokens,
    last_sampled_tokens,
    output_bin_counts,
    _output_bin_counts_stride,
    sampled_tokens,
    _sampled_tokens_stride,
    num_sampled,
    num_rejected,
    query_start_loc,
    all_token_ids,
    _all_token_ids_stride,
    total_len,
    **_,
) -> None:
    num_reqs = grid[0]
    req_raw = idx_mapping[:num_reqs].long()
    batch = torch.nonzero(req_raw >= 0).view(-1)
    if batch.numel() == 0:
        return
    req = req_raw[batch]
    sampled = num_sampled[batch].long()
    total = total_len[req].long()
    tokens = sampled_tokens[batch]
    for i in range(int(sampled.max())):
        mask = sampled > i
        rows = req[mask]
        token = tokens[mask, i].long()
        all_token_ids[rows, total[mask] + i] = token.to(all_token_ids.dtype)
        if output_bin_counts is not None:
            output_bin_counts.index_put_(
                (rows, token),
                torch.ones(
                    rows.shape[0], dtype=output_bin_counts.dtype, device=rows.device
                ),
                accumulate=True,
            )
    has_sampled = sampled > 0
    if bool(has_sampled.any()):
        last = tokens.gather(1, (sampled - 1).clamp(min=0).view(-1, 1)).view(-1)
        last_sampled_tokens.view(-1)[req[has_sampled]] = last[has_sampled].to(
            last_sampled_tokens.dtype
        )
        total_len[req[has_sampled]] = (total + sampled)[has_sampled].to(total_len.dtype)

    if query_start_loc is None:
        query_len = torch.zeros_like(sampled)
    else:
        qsl = query_start_loc[: num_reqs + 1].long()
        query_len = (qsl[1:] - qsl[:-1])[batch]
    delta = query_len - num_rejected[batch].long()
    num_computed_tokens[req] = (num_computed_tokens[req].long() + delta).to(
        num_computed_tokens.dtype
    )


_patch_kernel(_INPUT_BATCH, "_post_update_kernel", _post_update)


def _post_update_num_computed_tokens(
    grid, idx_mapping, num_computed_tokens, query_start_loc, **_
) -> None:
    num_reqs = grid[0]
    req = idx_mapping[:num_reqs].long()
    qsl = query_start_loc[: num_reqs + 1].long()
    num_computed_tokens[req] = (
        num_computed_tokens[req].long() + qsl[1:] - qsl[:-1]
    ).to(num_computed_tokens.dtype)


_patch_kernel(
    _INPUT_BATCH,
    "_post_update_num_computed_tokens_kernel",
    _post_update_num_computed_tokens,
)


def _expand_idx_mapping(
    grid, idx_mapping, expanded_idx_mapping, expanded_local_pos, cu_num_logits, **_
) -> None:
    num_reqs = grid[0]
    cu = cu_num_logits[: num_reqs + 1].long()
    seg, off = _expand(cu)
    expanded_idx_mapping[cu[0] : cu[-1]] = idx_mapping[:num_reqs][seg]
    expanded_local_pos[cu[0] : cu[-1]] = off.to(expanded_local_pos.dtype)


_patch_kernel(_INPUT_BATCH, "_expand_idx_mapping_kernel", _expand_idx_mapping)


# --------------------------------------------------------------------------
# sample kernels
# --------------------------------------------------------------------------


def _temperature(
    grid, logits, _stride, expanded_idx_mapping, temperature, vocab_size, **_
):
    num_tokens = grid[0]
    temp = temperature[expanded_idx_mapping[:num_tokens].long()].float()
    mask = (temp != 0.0) & (temp != 1.0)
    if bool(mask.any()):
        logits[mask] = (logits[mask].float() / temp[mask, None]).to(logits.dtype)


_patch_kernel("vllm.v1.worker.gpu.sample.gumbel", "_temperature_kernel", _temperature)


def _gumbel_seed(seed: int, pos: int) -> int:
    return hash((seed, pos)) & 0x7FFF_FFFF_FFFF_FFFF


def _gumbel_sample(
    grid,
    local_argmax,
    _local_argmax_stride,
    local_max,
    _local_max_stride,
    processed_logits,
    _processed_logits_stride,
    processed_logits_col,
    logits,
    _logits_stride,
    expanded_idx_mapping,
    seeds,
    pos,
    temp,
    vocab_size,
    APPLY_TEMPERATURE: bool = False,
    USE_FP64: bool = False,
    PER_TOKEN_COL: bool = False,
    **_,
) -> None:
    num_tokens = grid[0]
    req_raw = expanded_idx_mapping[:num_tokens].long()
    valid = req_raw >= 0
    req = req_raw.clamp(min=0)
    temperature = torch.where(
        valid, temp[req].float(), torch.zeros_like(req, dtype=torch.float32)
    )
    x = logits[:num_tokens].float()
    random = temperature != 0.0
    if APPLY_TEMPERATURE and bool(random.any()):
        x = torch.where(random[:, None], x / temperature.clamp(min=1e-30)[:, None], x)

    if processed_logits is not None:
        if processed_logits_col is None:
            col = torch.zeros(num_tokens, dtype=torch.long, device=x.device)
        elif PER_TOKEN_COL:
            col = processed_logits_col[:num_tokens].long()
        else:
            col = processed_logits_col.long().expand(num_tokens)
        rows = processed_logits.view(processed_logits.shape[0], -1, vocab_size)
        rows[req[valid], col[valid]] = x[valid].to(rows.dtype)

    y = x.double() if USE_FP64 else x
    if bool(random.any()):
        random_rows = torch.nonzero(random).view(-1)
        seed_list = seeds[req[random_rows]].tolist()
        pos_list = pos[random_rows].tolist()
        noise = torch.empty(random_rows.shape[0], vocab_size, dtype=y.dtype)
        for i, (seed, position) in enumerate(zip(seed_list, pos_list)):
            generator = torch.Generator().manual_seed(
                _gumbel_seed(int(seed), int(position))
            )
            u = torch.rand(vocab_size, generator=generator, dtype=y.dtype)
            if USE_FP64:
                u = u.clamp(min=2.2250738585072014e-308)
                noise[i] = -torch.log(-torch.log(u))
            else:
                u = u.clamp(min=4.6566127342e-10)
                noise[i] = -torch.log(-torch.log1p(-u))
        y[random_rows] = y[random_rows] + noise.to(y.device)

    value, index = y.max(dim=-1)
    local_max[:num_tokens].fill_(float("-inf"))
    local_max[:num_tokens, 0] = value.to(local_max.dtype)
    local_argmax[:num_tokens, 0] = index.to(local_argmax.dtype)


_patch_kernel(
    "vllm.v1.worker.gpu.sample.gumbel", "_gumbel_sample_kernel", _gumbel_sample
)


def _min_p(grid, logits, _stride, expanded_idx_mapping, min_p, vocab_size, **_):
    num_tokens = grid[0]
    p = min_p[expanded_idx_mapping[:num_tokens].long()].float()
    mask = p != 0.0
    if not bool(mask.any()):
        return
    rows = logits[mask].float()
    threshold = rows.max(dim=-1, keepdim=True).values + torch.log(p[mask])[:, None]
    logits[mask] = torch.where(rows < threshold, float("-inf"), rows).to(logits.dtype)


_patch_kernel("vllm.v1.worker.gpu.sample.min_p", "_min_p_kernel", _min_p)


def _penalties(
    grid,
    logits,
    _logits_stride,
    expanded_idx_mapping,
    token_ids,
    expanded_local_pos,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    prompt_bin_mask,
    _prompt_bin_mask_stride,
    output_bin_counts,
    _output_bin_counts_stride,
    vocab_size,
    **_,
) -> None:
    num_tokens = grid[0]
    req = expanded_idx_mapping[:num_tokens].long()
    rep = repetition_penalty[req].float()
    freq = frequency_penalty[req].float()
    pres = presence_penalty[req].float()
    use = (rep != 1.0) | (freq != 0.0) | (pres != 0.0)
    if not bool(use.any()):
        return
    rows = torch.nonzero(use).view(-1)
    req = req[rows]
    x = logits[rows].float()
    counts = output_bin_counts[req].float()

    local_pos = expanded_local_pos[:num_tokens].long()[rows]
    start = rows - local_pos
    for prev in range(int(local_pos.max())):
        mask = local_pos > prev
        prev_token = token_ids[start[mask] + prev + 1].long()
        counts.index_put_(
            (torch.nonzero(mask).view(-1), prev_token),
            torch.ones(prev_token.shape[0], dtype=counts.dtype, device=counts.device),
            accumulate=True,
        )
    output_mask = counts > 0

    prompt_mask = _unpack_bits(prompt_bin_mask[req], vocab_size)
    scale = torch.where(prompt_mask | output_mask, rep[rows, None], torch.ones_like(x))
    x = x * torch.where(x > 0, 1.0 / scale, scale)
    x = x - freq[rows, None] * counts - pres[rows, None] * output_mask.float()
    logits[rows] = x.to(logits.dtype)


_patch_kernel("vllm.v1.worker.gpu.sample.penalties", "_penalties_kernel", _penalties)


def _bincount(
    grid,
    expanded_idx_mapping,
    all_token_ids,
    _all_token_ids_stride,
    prompt_len,
    prefill_len,
    prompt_bin_mask,
    _prompt_bin_mask_stride,
    output_bin_counts,
    _output_bin_counts_stride,
    **_,
) -> None:
    num_words = prompt_bin_mask.shape[1]
    shifts = torch.arange(32, dtype=torch.int64, device=prompt_bin_mask.device)
    for req in expanded_idx_mapping[: grid[0]].tolist():
        prompt = int(prompt_len[req])
        prefill = int(prefill_len[req])
        tokens = all_token_ids[req, :prefill].long()
        seen = torch.zeros(num_words * 32, dtype=torch.bool, device=tokens.device)
        seen[tokens[:prompt]] = True
        packed = (seen.view(num_words, 32).long() << shifts).sum(-1)
        packed = torch.where(packed >= 2**31, packed - 2**32, packed)
        prompt_bin_mask[req] |= packed.to(prompt_bin_mask.dtype)
        output = tokens[prompt:]
        if output.numel():
            output_bin_counts[req].index_put_(
                (output,),
                torch.ones(
                    output.shape[0], dtype=output_bin_counts.dtype, device=output.device
                ),
                accumulate=True,
            )


_patch_kernel("vllm.v1.worker.gpu.sample.penalties", "_bincount_kernel", _bincount)


def _bias(
    grid,
    logits,
    _logits_stride,
    vocab_size,
    expanded_idx_mapping,
    num_allowed_token_ids,
    allowed_token_ids,
    _allowed_token_ids_stride,
    num_logit_bias,
    bias_token_ids,
    _bias_token_ids_stride,
    bias,
    _bias_stride,
    pos,
    min_lens,
    num_stop_token_ids,
    stop_token_ids,
    _stop_token_ids_stride,
    **_,
) -> None:
    num_tokens = grid[0]
    req = expanded_idx_mapping[:num_tokens].long()
    num_allowed = num_allowed_token_ids[req].tolist()
    num_bias = num_logit_bias[req].tolist()
    num_stop = num_stop_token_ids[req].tolist()
    min_len = min_lens[req].tolist()
    positions = pos[:num_tokens].tolist()
    for token_idx, req_idx in enumerate(req.tolist()):
        row = logits[token_idx]
        if num_allowed[token_idx] > 0:
            ids = allowed_token_ids[req_idx, : num_allowed[token_idx]].long()
            kept = row[ids].clone()
            row.fill_(float("-inf"))
            row[ids] = kept
        if num_bias[token_idx] > 0:
            n = num_bias[token_idx]
            ids = bias_token_ids[req_idx, :n].long()
            row[ids] = (row[ids].float() + bias[req_idx, :n].float()).to(row.dtype)
        if num_stop[token_idx] > 0 and positions[token_idx] + 1 < min_len[token_idx]:
            row[stop_token_ids[req_idx, : num_stop[token_idx]].long()] = float("-inf")


_patch_kernel("vllm.v1.worker.gpu.sample.logit_bias", "_bias_kernel", _bias)


def _bad_words(
    grid,
    logits,
    _logits_stride,
    expanded_idx_mapping,
    bad_word_token_ids,
    _bad_word_token_ids_stride,
    bad_word_offsets,
    _bad_word_offsets_stride,
    num_bad_words,
    all_token_ids,
    _all_token_ids_stride,
    prompt_len,
    total_len,
    input_ids,
    expanded_local_pos,
    **_,
) -> None:
    num_tokens = grid[0]
    reqs = expanded_idx_mapping[:num_tokens].tolist()
    local_pos = expanded_local_pos[:num_tokens].tolist()
    for token_idx, req in enumerate(reqs):
        count = int(num_bad_words[req])
        if count == 0:
            continue
        pos = local_pos[token_idx]
        prompt = int(prompt_len[req])
        output_len = int(total_len[req]) - prompt
        effective_len = output_len + pos
        offsets = bad_word_offsets[req, : count + 1].tolist()
        words = bad_word_token_ids[req, : offsets[-1]].tolist()
        output_tokens = all_token_ids[req, prompt : prompt + output_len].tolist()
        first = token_idx - pos
        spec_tokens = input_ids[first : first + pos + 1].tolist()
        for w in range(count):
            word = words[offsets[w] : offsets[w + 1]]
            prefix = word[:-1]
            if len(prefix) > effective_len:
                continue
            base = effective_len - len(prefix)
            matched = True
            for i, expected in enumerate(prefix):
                actual_pos = base + i
                actual = (
                    spec_tokens[actual_pos - output_len]
                    if actual_pos >= output_len
                    else output_tokens[actual_pos]
                )
                if actual != expected:
                    matched = False
                    break
            if matched:
                logits[token_idx, word[-1]] = float("-inf")


_patch_kernel("vllm.v1.worker.gpu.sample.bad_words", "_bad_words_kernel", _bad_words)


def _topk_log_softmax(grid, output, logits, _stride, topk_ids, topk, vocab_size, **_):
    batch_size = grid[0]
    x = logits[:batch_size].float()
    lse = torch.logsumexp(x, dim=-1, keepdim=True)
    ids = topk_ids.view(batch_size, topk).long()
    output.view(batch_size, topk)[:] = (x.gather(1, ids) - lse).to(output.dtype)


_patch_kernel(
    "vllm.v1.worker.gpu.sample.logprob", "_topk_log_softmax_kernel", _topk_log_softmax
)


def _ranks(grid, output, logits, _stride, token_ids, vocab_size, **_):
    batch_size = grid[0]
    x = logits[:batch_size]
    selected = x.gather(1, token_ids[:batch_size].view(batch_size, 1).long())
    output[:batch_size] = (x >= selected).sum(dim=-1).to(output.dtype)


_patch_kernel("vllm.v1.worker.gpu.sample.logprob", "_ranks_kernel", _ranks)


def _fill_logprob_token_ids(
    grid,
    out_token_ids,
    _out_token_ids_stride,
    out_valid_mask,
    _out_valid_mask_stride,
    sampled_token_ids,
    topk_indices,
    _topk_indices_stride,
    expanded_idx_mapping,
    num_per_req_token_ids,
    per_req_token_ids,
    _per_req_token_ids_stride,
    NUM_TOPK: int = 0,
    **_,
) -> None:
    batch_size = grid[0]
    out_token_ids[:batch_size, 0] = sampled_token_ids[:batch_size].to(
        out_token_ids.dtype
    )
    out_valid_mask[:batch_size, 0] = True
    req = expanded_idx_mapping[:batch_size].long()
    num_custom = num_per_req_token_ids[req].long()
    num_cols = out_token_ids.shape[1] - 1
    col = torch.arange(num_cols, device=req.device)
    custom = num_custom > 0
    valid_custom = custom[:, None] & (col[None, :] < num_custom[:, None])
    valid_topk = (~custom)[:, None] & (col[None, :] < NUM_TOPK)
    tokens = torch.zeros(
        batch_size, num_cols, dtype=out_token_ids.dtype, device=req.device
    )
    width = min(num_cols, per_req_token_ids.shape[1])
    tokens[:, :width] = torch.where(
        valid_custom[:, :width],
        per_req_token_ids[req][:, :width].to(tokens.dtype),
        tokens[:, :width],
    )
    if NUM_TOPK > 0:
        width = min(num_cols, NUM_TOPK)
        tokens[:, :width] = torch.where(
            valid_topk[:, :width],
            topk_indices[:batch_size, :width].to(tokens.dtype),
            tokens[:, :width],
        )
    valid = valid_custom | valid_topk
    out_token_ids[:batch_size, 1:] = torch.where(
        valid, tokens, out_token_ids[:batch_size, 1:]
    )
    out_valid_mask[:batch_size, 1:] |= valid


_patch_kernel(
    "vllm.v1.worker.gpu.sample.logprob",
    "_fill_logprob_token_ids_kernel",
    _fill_logprob_token_ids,
)


def _prompt_logprobs_token_ids(
    grid,
    prompt_logprobs_token_ids,
    query_start_loc,
    idx_mapping,
    num_computed_tokens,
    all_token_ids,
    _all_token_ids_stride,
    **_,
) -> None:
    num_reqs = grid[0]
    req = idx_mapping[:num_reqs].long()
    qsl = query_start_loc[: num_reqs + 1].long()
    tok_req, tok_off = _expand(qsl)
    rows = req[tok_req]
    cols = num_computed_tokens[req].long()[tok_req] + 1 + tok_off
    prompt_logprobs_token_ids[qsl[0] : qsl[-1]] = all_token_ids[rows, cols].to(
        prompt_logprobs_token_ids.dtype
    )


_patch_kernel(
    "vllm.v1.worker.gpu.sample.prompt_logprob",
    "_prompt_logprobs_token_ids_kernel",
    _prompt_logprobs_token_ids,
)


def _num_nans(grid, logits, _stride, num_nans, vocab_size, **_):
    num_nans[: grid[0]] = logits[: grid[0]].isnan().sum(dim=-1).to(num_nans.dtype)


_patch_kernel("vllm.v1.worker.gpu.metrics.logits", "_num_nans_kernel", _num_nans)


# --------------------------------------------------------------------------
# structured outputs: grammar bitmask on a CUDA copy stream
# --------------------------------------------------------------------------


def _apply_grammar_bitmask(
    grid,
    logits,
    _logits_stride,
    logits_indices,
    bitmask,
    _bitmask_stride,
    vocab_size,
    **_,
) -> None:
    num_masks = grid[0]
    rows = logits_indices[:num_masks].long()
    blocked = ~_unpack_bits(bitmask[:num_masks], vocab_size)
    logits[rows] = logits[rows].masked_fill(blocked, float("-inf"))


_patch_kernel(
    "vllm.v1.worker.gpu.structured_outputs",
    "_apply_grammar_bitmask_kernel",
    _apply_grammar_bitmask,
)


def _structured_outputs_init(
    self: StructuredOutputsWorker,
    max_num_logits: int,
    vocab_size: int,
    device: torch.device,
) -> None:
    self.logits_indices = torch.zeros(max_num_logits, dtype=torch.int32, device=device)
    self.grammar_bitmask = torch.zeros(
        (max_num_logits, (vocab_size + 31) // 32), dtype=torch.int32, device=device
    )
    self.device = device


def _structured_outputs_apply(
    self: StructuredOutputsWorker,
    logits: torch.Tensor,
    input_batch: InputBatch,
    grammar_req_ids: list[str],
    grammar_bitmask: np.ndarray,
) -> None:
    if not grammar_req_ids:
        return
    num_masks = grammar_bitmask.shape[0]
    bitmask = self.grammar_bitmask[:num_masks]
    bitmask.copy_(torch.from_numpy(grammar_bitmask))

    mapping: list[int] = []
    cu_num_logits = input_batch.cu_num_logits_np.tolist()
    req_id_to_idx = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        mapping.extend(range(cu_num_logits[req_idx], cu_num_logits[req_idx + 1]))
    assert num_masks == len(mapping)
    logits_indices = self.logits_indices[:num_masks]
    logits_indices.copy_(torch.tensor(mapping, dtype=torch.int32))
    _apply_grammar_bitmask(
        (num_masks,),
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        logits.shape[-1],
    )


register_patch(
    target="vllm.v1.worker.gpu.structured_outputs.StructuredOutputsWorker.__init__",
    reason=_NO_CUDA_REASON,
)(_structured_outputs_init)
register_patch(
    target="vllm.v1.worker.gpu.structured_outputs.StructuredOutputsWorker.apply_grammar_bitmask",
    reason=_NO_CUDA_REASON,
)(_structured_outputs_apply)


# --------------------------------------------------------------------------
# async_utils: the D2H copy of a step's sampler output
# --------------------------------------------------------------------------


def _to_cpu(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()


def _async_output_init(
    self: AsyncOutput,
    model_runner_output: ModelRunnerOutput,
    sampler_output: SamplerOutput,
    num_sampled_tokens: torch.Tensor,
    main_stream: Any,
    copy_stream: Any,
) -> None:
    self.model_runner_output = model_runner_output
    self.sampler_output = sampler_output
    self.num_sampled_tokens = num_sampled_tokens
    self.copy_event = types.SimpleNamespace(synchronize=lambda: None)

    self.sampled_token_ids = _to_cpu(sampler_output.sampled_token_ids)
    self.logprobs_tensors = None
    if sampler_output.logprobs_tensors is not None:
        self.logprobs_tensors = sampler_output.logprobs_tensors.to_cpu_nonblocking()
    self.num_nans = None
    if sampler_output.num_nans is not None:
        self.num_nans = _to_cpu(sampler_output.num_nans)
    self.num_sampled_tokens_np = _to_cpu(num_sampled_tokens)
    self.prompt_logprobs_dict = {
        k: v.to_cpu_nonblocking() if v is not None else None
        for k, v in model_runner_output.prompt_logprobs_dict.items()
    }
    torch.accelerator.synchronize()


register_patch(
    target="vllm.v1.worker.gpu.async_utils.AsyncOutput.__init__",
    reason=_NO_CUDA_REASON,
)(_async_output_init)


def _async_pooling_output_init(
    self: AsyncPoolingOutput,
    model_runner_output: ModelRunnerOutput,
    pooler_output: torch.Tensor,
    is_valid: torch.Tensor | None,
    main_stream: Any,
    copy_stream: Any,
) -> None:
    self.model_runner_output = model_runner_output
    self.pooler_output = pooler_output
    self.is_valid = is_valid
    self.copy_event = types.SimpleNamespace(synchronize=lambda: None)
    self.pooler_output_cpu = pooler_output.cpu()
    self.is_valid_cpu = is_valid.cpu() if is_valid is not None else None


register_patch(
    target="vllm.v1.worker.gpu.async_utils.AsyncPoolingOutput.__init__",
    reason=_NO_CUDA_REASON,
)(_async_pooling_output_init)


# --------------------------------------------------------------------------
# pp_utils: the sampled-token broadcast between PP stages
# --------------------------------------------------------------------------

_NO_CUDA_PP_REASON = (
    "PPHandler broadcasts the last stage's sampled tokens on a CUDA side stream "
    "over a sibling NCCL group and defers the read behind a CUDA event. RBLN "
    "has neither, so the broadcast runs synchronously on the PP group's CPU "
    "process group."
)


def _pp_handler_init(
    self: PPHandler, max_num_reqs: int, num_speculative_steps: int, device: torch.device
) -> None:
    pp_group = get_pp_group()
    self.is_last_rank = pp_group.is_last_rank
    self.last_rank = pp_group.last_rank
    self.max_sample_len = num_speculative_steps + 1
    self.device = device
    self.queue = deque() if self.is_last_rank else deque([None] * pp_group.world_size)
    self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)
    self.broadcast_group = pp_group.cpu_group


def _pp_handler_receive(self: PPHandler, input_batch: InputBatch) -> bool:
    assert not self.is_last_rank
    need_sampled_mask = compute_need_sampled_mask(input_batch)
    if need_sampled_mask is None:
        return False
    gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]
    num_reqs = input_batch.num_reqs
    sampled_tokens = torch.empty(num_reqs, self.max_sample_len, dtype=torch.int64)
    combined = torch.empty(2, num_reqs, dtype=torch.int32)
    dist.broadcast(sampled_tokens, src=self.last_rank, group=self.broadcast_group)
    dist.broadcast(combined, src=self.last_rank, group=self.broadcast_group)
    num_sampled, num_rejected = combined.to(self.device).unbind(dim=0)
    self.queue[-1] = PendingRecv(
        None,
        sampled_tokens.to(self.device),
        num_sampled,
        num_rejected,
        input_batch.idx_mapping,
        input_batch.idx_mapping_np,
        need_sampled_mask,
        gen_at_receive_np,
    )
    return bool(need_sampled_mask.all())


def _pp_handler_broadcast(
    self: PPHandler,
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    input_batch: InputBatch,
) -> None:
    assert self.is_last_rank
    if compute_need_sampled_mask(input_batch) is None:
        return
    assert sampled_token_ids.dtype == torch.int64
    dist.broadcast(
        sampled_token_ids.contiguous().cpu(),
        src=self.last_rank,
        group=self.broadcast_group,
    )
    combined = torch.stack((num_sampled, num_rejected), dim=0).cpu()
    dist.broadcast(combined, src=self.last_rank, group=self.broadcast_group)


def _pp_handler_get_prev_sampled_outputs(
    self: PPHandler,
) -> dict[str, torch.Tensor] | None:
    if not self.queue:
        return None
    slot = self.queue.popleft()
    self.queue.append(None)
    if slot is None:
        return None
    freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
    exclude_mask = freed | ~slot.need_sampled_mask
    idx_mapping = slot.idx_mapping
    if exclude_mask.any():
        if exclude_mask.all():
            return None
        idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
        idx_mapping = torch.from_numpy(idx_mapping_np).to(self.device)
    return dict(
        sampled_tokens=slot.sampled_tokens,
        num_sampled=slot.num_sampled,
        num_rejected=slot.num_rejected,
        idx_mapping=idx_mapping,
    )


for _name, _fn in (
    ("__init__", _pp_handler_init),
    ("receive", _pp_handler_receive),
    ("broadcast", _pp_handler_broadcast),
    ("get_prev_sampled_outputs", _pp_handler_get_prev_sampled_outputs),
):
    register_patch(
        target=f"vllm.v1.worker.gpu.pp_utils.PPHandler.{_name}",
        reason=_NO_CUDA_PP_REASON,
    )(_fn)
