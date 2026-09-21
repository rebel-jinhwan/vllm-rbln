# Copyright 2026 Rebellions Inc. All rights reserved.
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

"""torch implementations of the V2 model runner's Triton kernels.

Upstream launches each `@triton.jit` kernel under `vllm/v1/worker/gpu/` as
`kernel[grid](*args, **constexpr)`. Without Triton, `PlaceholderKernel` asks
`Platform.get_kernel_impl` for the implementation, which the platform answers
from `KERNELS`. Each function takes `(grid, *args, **constexpr)` in the
kernel's own argument order, receives the tensor where the kernel received a
pointer, and vectorizes over the grid.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch


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


def _gumbel_seed(seed: int, pos: int) -> int:
    return hash((seed, pos)) & 0x7FFF_FFFF_FFFF_FFFF


def _gumbel_noise(seed: int, pos: int, vocab_size: int, use_fp64: bool) -> torch.Tensor:
    """The per-(seed, position) Gumbel noise row that upstream's philox stream
    plays here: the same pair always yields the same row."""
    generator = torch.Generator().manual_seed(_gumbel_seed(seed, pos))
    dtype = torch.float64 if use_fp64 else torch.float32
    u = torch.rand(vocab_size, generator=generator, dtype=dtype)
    if use_fp64:
        return -torch.log(-torch.log(u.clamp(min=2.2250738585072014e-308)))
    return -torch.log(-torch.log1p(-u.clamp(min=4.6566127342e-10)))


def _uniform(seed: int, pos: int) -> float:
    """One draw in (0, 1) per (seed, position), independent of the row above."""
    generator = torch.Generator().manual_seed(_gumbel_seed(seed, pos) ^ 0x5BD1E995)
    return float(torch.rand(1, generator=generator).clamp(min=1e-30))


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


def _post_update_num_computed_tokens(
    grid, idx_mapping, num_computed_tokens, query_start_loc, **_
) -> None:
    num_reqs = grid[0]
    req = idx_mapping[:num_reqs].long()
    qsl = query_start_loc[: num_reqs + 1].long()
    num_computed_tokens[req] = (
        num_computed_tokens[req].long() + qsl[1:] - qsl[:-1]
    ).to(num_computed_tokens.dtype)


def _expand_idx_mapping(
    grid, idx_mapping, expanded_idx_mapping, expanded_local_pos, cu_num_logits, **_
) -> None:
    num_reqs = grid[0]
    cu = cu_num_logits[: num_reqs + 1].long()
    seg, off = _expand(cu)
    expanded_idx_mapping[cu[0] : cu[-1]] = idx_mapping[:num_reqs][seg]
    expanded_local_pos[cu[0] : cu[-1]] = off.to(expanded_local_pos.dtype)


def _temperature(
    grid, logits, _stride, expanded_idx_mapping, temperature, vocab_size, **_
):
    num_tokens = grid[0]
    temp = temperature[expanded_idx_mapping[:num_tokens].long()].float()
    mask = (temp != 0.0) & (temp != 1.0)
    if bool(mask.any()):
        logits[mask] = (logits[mask].float() / temp[mask, None]).to(logits.dtype)


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
        noise = torch.stack(
            [
                _gumbel_noise(int(seed), int(position), vocab_size, USE_FP64)
                for seed, position in zip(seed_list, pos_list)
            ]
        )
        y[random_rows] = y[random_rows] + noise.to(y.device)

    value, index = y.max(dim=-1)
    local_max[:num_tokens].fill_(float("-inf"))
    local_max[:num_tokens, 0] = value.to(local_max.dtype)
    local_argmax[:num_tokens, 0] = index.to(local_argmax.dtype)


def _min_p(grid, logits, _stride, expanded_idx_mapping, min_p, vocab_size, **_):
    num_tokens = grid[0]
    p = min_p[expanded_idx_mapping[:num_tokens].long()].float()
    mask = p != 0.0
    if not bool(mask.any()):
        return
    rows = logits[mask].float()
    threshold = rows.max(dim=-1, keepdim=True).values + torch.log(p[mask])[:, None]
    logits[mask] = torch.where(rows < threshold, float("-inf"), rows).to(logits.dtype)


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


def _topk_log_softmax(grid, output, logits, _stride, topk_ids, topk, vocab_size, **_):
    batch_size = grid[0]
    x = logits[:batch_size].float()
    lse = torch.logsumexp(x, dim=-1, keepdim=True)
    ids = topk_ids.view(batch_size, topk).long()
    output.view(batch_size, topk)[:] = (x.gather(1, ids) - lse).to(output.dtype)


def _ranks(grid, output, logits, _stride, token_ids, vocab_size, **_):
    batch_size = grid[0]
    x = logits[:batch_size]
    selected = x.gather(1, token_ids[:batch_size].view(batch_size, 1).long())
    output[:batch_size] = (x >= selected).sum(dim=-1).to(output.dtype)


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


def _num_nans(grid, logits, _stride, num_nans, vocab_size, **_):
    num_nans[: grid[0]] = logits[: grid[0]].isnan().sum(dim=-1).to(num_nans.dtype)


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


def _flatten_sampled(
    grid, flat_sampled, sampled, _stride, num_sampled, cu_num_logits, **_
) -> None:
    num_reqs = grid[0]
    counts = num_sampled[:num_reqs].tolist()
    starts = cu_num_logits[:num_reqs].tolist()
    rows = [r for r, n in enumerate(counts) for _ in range(n)]
    cols = [i for n in counts for i in range(n)]
    flat = [starts[r] + i for r, n in enumerate(counts) for i in range(n)]
    if flat:
        flat_sampled[torch.tensor(flat, device=flat_sampled.device)] = sampled[
            torch.tensor(rows, device=sampled.device),
            torch.tensor(cols, device=sampled.device),
        ]


def _apply_write(
    grid,
    out,
    out_stride,
    indices,
    starts,
    contents,
    cu_lens,
    group_ids,
    BLOCK_SIZE: int = 1024,
    MULTI_GROUP: bool = False,
    **_,
) -> None:
    assert not MULTI_GROUP, "the fused writer flushes per tensor without Triton"
    n = grid[0]
    # The index tables arrive on the platform device; the destination may be
    # a host tensor (tests), so index on the destination's device.
    idx = indices[:n].to(out.device).long()
    cu = torch.cat([cu_lens.new_zeros(1), cu_lens[:n]]).to(out.device).long()
    write, off = _expand(cu)
    flat = idx[write] * out_stride + starts[:n].to(out.device).long()[write] + off
    out.view(-1)[flat] = contents[: int(cu[-1])].to(out.device, out.dtype)


def _logsumexp(row: torch.Tensor) -> float:
    return float(torch.logsumexp(row.float(), dim=-1))


def _compute_local_logits_stats(
    grid,
    target_local_argmax,
    _s1,
    target_local_max,
    _s2,
    target_local_sumexp,
    _s3,
    draft_local_max,
    _s4,
    draft_local_sumexp,
    _s5,
    target_logits,
    _s6,
    draft_logits,
    _s7,
    _s8,
    expanded_idx_mapping,
    expanded_local_pos,
    temp,
    vocab_size,
    num_speculative_steps,
    HAS_DRAFT_LOGITS: bool = False,
    **_,
) -> None:
    """Per logit row, the statistics the rejection kernels read: the row's
    argmax and max for greedy requests, its max and sum-exp otherwise. One
    block per row holds the whole row; the others are the reduction identity."""
    num_logits = grid[0]
    rows = torch.arange(num_logits, device=target_logits.device)
    active = expanded_local_pos[:num_logits] < num_speculative_steps
    rows = rows[active]
    if rows.numel() == 0:
        return
    logits = target_logits[rows, :vocab_size].float()
    req = expanded_idx_mapping[rows].long()
    greedy = temp[req].float() == 0.0
    for t in (target_local_max, draft_local_max):
        t[rows] = float("-inf")
    for t in (target_local_sumexp, draft_local_sumexp):
        t[rows] = 0.0
    value, index = logits.max(dim=-1)
    target_local_max[rows, 0] = value
    target_local_argmax[rows, 0] = index
    sumexp = torch.exp(logits - value[:, None]).sum(dim=-1)
    target_local_sumexp[rows, 0] = torch.where(greedy, torch.zeros_like(sumexp), sumexp)
    if HAS_DRAFT_LOGITS:
        step = expanded_local_pos[rows].long()
        draft = draft_logits[req, step, :vocab_size].float()
        dvalue = draft.max(dim=-1).values
        draft_local_max[rows, 0] = dvalue
        draft_local_sumexp[rows, 0] = torch.exp(draft - dvalue[:, None]).sum(dim=-1)


def _rejection(
    grid,
    sampled,
    _s1,
    num_sampled,
    target_rejected_lse,
    draft_rejected_lse,
    target_logits,
    _s2,
    target_local_argmax,
    _s3,
    target_local_max,
    _s4,
    target_local_sumexp,
    _s5,
    draft_sampled,
    draft_logits,
    _s6,
    _s7,
    draft_local_max,
    _s8,
    draft_local_sumexp,
    _s9,
    cu_num_logits,
    idx_mapping,
    temp,
    seed,
    pos,
    synthetic_conditional_rates,
    cumulative_log_p,
    local_residual_mass,
    _s10,
    vocab_num_blocks,
    HAS_DRAFT_LOGITS: bool = False,
    SYNTHETIC_MODE: bool = False,
    USE_BLOCK_VERIFICATION: bool = False,
    **_,
) -> None:
    """Per request, accept draft tokens while the target agrees (greedy) or
    passes the acceptance test (random); the first rejected slot receives the
    target argmax (greedy) or is left for _resample."""
    if SYNTHETIC_MODE or USE_BLOCK_VERIFICATION:
        raise NotImplementedError(
            "RBLN rejection sampling supports rejection_sample_method='standard' only"
        )
    num_reqs = grid[0]
    cu = cu_num_logits[: num_reqs + 1].tolist()
    req_state = idx_mapping[:num_reqs].long()
    temps = temp[req_state].tolist()
    seeds = seed[req_state].tolist()
    req_state = req_state.tolist()
    positions = pos.tolist()
    drafts = draft_sampled.tolist()
    argmax = target_local_argmax[:, 0].tolist()
    vocab = draft_logits.shape[-1] if HAS_DRAFT_LOGITS else target_logits.shape[-1]
    vocab = min(vocab, target_logits.shape[-1])
    accepted_lens = []
    target_lses = []
    draft_lses = []
    for r in range(num_reqs):
        start, end = cu[r], cu[r + 1]
        greedy = temps[r] == 0.0
        accepted = 0
        for i in range(end - start - 1):
            li = start + i
            d = drafts[li + 1]
            if greedy:
                t = argmax[li]
                ok = t == d
                sampled[r, i] = d if ok else t
            else:
                lp = torch.log_softmax(target_logits[li, :vocab].float(), -1)[max(d, 0)]
                if HAS_DRAFT_LOGITS:
                    q = draft_logits[req_state[r], i, :vocab].float()
                    lp = lp - torch.log_softmax(q, -1)[max(d, 0)]
                ok = d >= 0 and float(lp) > np.log(_uniform(seeds[r], positions[li]))
                sampled[r, i] = max(d, 0)
            if not ok:
                break
            accepted += 1
        accepted_lens.append(accepted)
        idx = start + accepted
        rejected = not greedy and idx < end - 1
        target_lses.append(_logsumexp(target_logits[idx, :vocab]) if rejected else 0.0)
        draft_lses.append(
            _logsumexp(draft_logits[req_state[r], accepted, :vocab])
            if rejected and HAS_DRAFT_LOGITS
            else 0.0
        )
    num_sampled[:num_reqs] = torch.tensor(accepted_lens, dtype=num_sampled.dtype)
    target_rejected_lse[:num_reqs] = torch.tensor(target_lses)
    draft_rejected_lse[:num_reqs] = torch.tensor(draft_lses)


def _resample(
    grid,
    resampled_local_argmax,
    _s1,
    resampled_local_max,
    _s2,
    target_logits,
    _s3,
    target_rejected_lse,
    draft_logits,
    _s4,
    _s5,
    draft_rejected_lse,
    rejected_step,
    cu_num_logits,
    expanded_idx_mapping,
    draft_sampled,
    temp,
    seed,
    pos,
    cumulative_log_p,
    vocab_size,
    HAS_DRAFT_LOGITS: bool = False,
    USE_FP64: bool = False,
    USE_BLOCK_VERIFICATION: bool = False,
    **_,
) -> None:
    """One token per request from the residual distribution at the rejected
    slot, or from the target at the bonus slot; a greedy request's rejected
    slot already holds the target argmax and is skipped, as upstream does."""
    assert not USE_BLOCK_VERIFICATION
    num_reqs = grid[0]
    cu = cu_num_logits[: num_reqs + 1].tolist()
    accepted = rejected_step[:num_reqs].tolist()
    drafts = draft_sampled.tolist()
    positions = pos.tolist()
    for r in range(num_reqs):
        idx = cu[r] + accepted[r]
        is_bonus = idx == cu[r + 1] - 1
        req_state = int(expanded_idx_mapping[idx])
        t = float(temp[req_state])
        if t == 0.0 and not is_bonus:
            continue
        row = target_logits[idx, :vocab_size].float()
        if is_bonus:
            residual = row
        elif HAS_DRAFT_LOGITS:
            target_lp = row - float(target_rejected_lse[r])
            draft_lp = draft_logits[
                req_state, accepted[r], :vocab_size
            ].float() - float(draft_rejected_lse[r])
            ratio = torch.exp(draft_lp - target_lp)
            residual = torch.where(
                ratio < 1.0, target_lp + torch.log1p(-ratio), float("-inf")
            )
        else:
            residual = row.clone()
            residual[drafts[idx + 1]] = float("-inf")
        if t != 0.0:
            noise = _gumbel_noise(
                int(seed[req_state]), positions[idx], vocab_size, USE_FP64
            )
            residual = residual.cpu().to(noise.dtype) + noise
        value, index = residual.max(dim=-1)
        resampled_local_max[r] = float("-inf")
        resampled_local_max[r, 0] = value.to(resampled_local_max.dtype)
        resampled_local_argmax[r, 0] = int(index)


def _insert_resampled(
    grid,
    sampled,
    _s1,
    num_sampled,
    resampled_local_argmax,
    _s2,
    resampled_local_max,
    _s3,
    resample_num_blocks,
    cu_num_logits,
    expanded_idx_mapping,
    temp,
    **_,
) -> None:
    num_reqs = grid[0]
    cu = cu_num_logits[: num_reqs + 1].tolist()
    accepted = num_sampled[:num_reqs].tolist()
    for r in range(num_reqs):
        idx = cu[r] + accepted[r]
        num_sampled[r] = accepted[r] + 1
        is_bonus = idx == cu[r + 1] - 1
        if float(temp[int(expanded_idx_mapping[idx])]) == 0.0 and not is_bonus:
            continue
        block = int(resampled_local_max[r].argmax())
        sampled[r, accepted[r]] = resampled_local_argmax[r, block]


def _block_verification_unsupported(grid, *args, **kwargs) -> None:
    raise NotImplementedError(
        "RBLN rejection sampling supports rejection_sample_method='standard' only"
    )


def _draft_prepare_prefill_inputs(
    grid,
    last_token_indices,
    draft_current_step,
    draft_input_ids,
    draft_positions,
    draft_query_start_loc,
    draft_seq_lens,
    target_input_ids,
    target_positions,
    idx_mapping,
    last_sampled,
    next_prefill_tokens,
    num_sampled,
    num_rejected,
    query_start_loc,
    seq_lens,
    max_num_reqs,
    **_,
) -> None:
    """The draft's first-pass inputs: the target's tokens shifted one left
    within each request, the token the target sampled last (or the next
    prompt token) in the last kept slot, and the target's positions."""
    num_reqs = grid[0]
    qsl = query_start_loc[: num_reqs + 1].long()
    starts, ends = qsl[:-1], qsl[1:]
    query_len = ends - starts - num_rejected[:num_reqs].long()
    req_state = idx_mapping[:num_reqs].long()
    next_token = torch.where(
        num_sampled[:num_reqs] > 0,
        last_sampled[req_state].view(-1).to(draft_input_ids.dtype),
        next_prefill_tokens[req_state].to(draft_input_ids.dtype),
    )
    num_tokens = int(ends[-1])
    tok_req, off = _expand(qsl)
    kept = off < query_len[tok_req]
    src = torch.arange(num_tokens, device=qsl.device)
    shifted = src[kept & (off >= 1)]
    draft_input_ids[shifted - 1] = target_input_ids[shifted]
    last = starts + query_len - 1
    last_token_indices[:num_reqs] = last
    last_token_indices[num_reqs:max_num_reqs] = 0
    draft_input_ids[last] = next_token
    draft_positions[src[kept]] = target_positions[src[kept]]
    draft_query_start_loc[:num_reqs] = starts.to(draft_query_start_loc.dtype)
    draft_query_start_loc[num_reqs : max_num_reqs + 1] = ends[-1].to(
        draft_query_start_loc.dtype
    )
    draft_seq_lens[:num_reqs] = seq_lens[:num_reqs]
    draft_seq_lens[num_reqs:max_num_reqs] = 0
    draft_current_step.fill_(0)


def _draft_prepare_decode_inputs(
    grid,
    draft_tokens,
    _stride,
    target_seq_lens,
    num_rejected,
    input_ids,
    positions,
    query_start_loc,
    seq_lens,
    max_model_len,
    max_num_reqs,
    ADVANCE_DRAFT_POSITIONS: bool = True,
    **_,
) -> None:
    num_reqs = grid[0] - 1
    input_ids[:num_reqs] = draft_tokens[:num_reqs].to(input_ids.dtype)
    if ADVANCE_DRAFT_POSITIONS:
        positions[:num_reqs] = torch.clamp(
            positions[:num_reqs] + 1, max=max_model_len - 1
        )
        seq_lens[:num_reqs] = torch.clamp(
            target_seq_lens[:num_reqs] - num_rejected[:num_reqs] + 1, max=max_model_len
        )
    arange = torch.arange(max_num_reqs + 1, device=query_start_loc.device)
    query_start_loc[: max_num_reqs + 1] = torch.clamp(arange, max=num_reqs).to(
        query_start_loc.dtype
    )
    seq_lens[num_reqs:max_num_reqs] = 0


def _draft_update_inputs(
    grid,
    output_draft_tokens,
    _stride1,
    next_input_hidden_states,
    _stride2,
    input_ids,
    positions,
    seq_lens,
    draft_tokens,
    current_draft_step,
    hidden_states,
    _stride3,
    hidden_size,
    max_model_len,
    num_speculative_steps,
    ADVANCE_DRAFT_POSITIONS: bool = True,
    **_,
) -> None:
    num_reqs = grid[0]
    step = int(current_draft_step)
    output_draft_tokens[:num_reqs, step] = draft_tokens[:num_reqs]
    if step >= num_speculative_steps - 1:
        return
    input_ids[:num_reqs] = draft_tokens[:num_reqs].to(input_ids.dtype)
    next_input_hidden_states[:num_reqs, :hidden_size] = hidden_states[
        :num_reqs, :hidden_size
    ]
    if ADVANCE_DRAFT_POSITIONS:
        positions[:num_reqs] = torch.clamp(
            positions[:num_reqs] + 1, max=max_model_len - 1
        )
        seq_lens[:num_reqs] = torch.clamp(seq_lens[:num_reqs] + 1, max=max_model_len)


_KERNELS_BY_MODULE: dict[str, dict[str, Callable[..., Any]]] = {
    "input_batch": {
        "_prepare_prefill_inputs_kernel": _prepare_prefill_inputs,
        "_prepare_pos_seq_lens_kernel": _prepare_pos_seq_lens,
        "_combine_sampled_and_draft_tokens_kernel": _combine_sampled_and_draft_tokens,
        "_get_num_sampled_and_rejected_kernel": _get_num_sampled_and_rejected,
        "_post_update_kernel": _post_update,
        "_post_update_num_computed_tokens_kernel": _post_update_num_computed_tokens,
        "_expand_idx_mapping_kernel": _expand_idx_mapping,
    },
    "sample.gumbel": {
        "_temperature_kernel": _temperature,
        "_gumbel_sample_kernel": _gumbel_sample,
    },
    "sample.min_p": {
        "_min_p_kernel": _min_p,
    },
    "sample.penalties": {
        "_penalties_kernel": _penalties,
        "_bincount_kernel": _bincount,
    },
    "sample.logit_bias": {
        "_bias_kernel": _bias,
    },
    "sample.bad_words": {
        "_bad_words_kernel": _bad_words,
    },
    "sample.logprob": {
        "_topk_log_softmax_kernel": _topk_log_softmax,
        "_ranks_kernel": _ranks,
        "_fill_logprob_token_ids_kernel": _fill_logprob_token_ids,
    },
    "sample.prompt_logprob": {
        "_prompt_logprobs_token_ids_kernel": _prompt_logprobs_token_ids,
    },
    "metrics.logits": {
        "_num_nans_kernel": _num_nans,
    },
    "structured_outputs": {
        "_apply_grammar_bitmask_kernel": _apply_grammar_bitmask,
    },
    "spec_decode.rejection_sampler": {
        "_flatten_sampled_kernel": _flatten_sampled,
    },
    "buffer_utils": {
        "_apply_write_kernel": _apply_write,
    },
    "spec_decode.autoregressive.speculator": {
        "_prepare_prefill_inputs_kernel": _draft_prepare_prefill_inputs,
        "_prepare_decode_inputs_kernel": _draft_prepare_decode_inputs,
        "_update_draft_inputs_kernel": _draft_update_inputs,
    },
    "spec_decode.rejection_sampler_utils": {
        "_compute_local_logits_stats_kernel": _compute_local_logits_stats,
        "_compute_cumulative_log_p_kernel": _block_verification_unsupported,
        "_compute_local_residual_mass_kernel": _block_verification_unsupported,
        "_rejection_kernel": _rejection,
        "_resample_kernel": _resample,
        "_insert_resampled_kernel": _insert_resampled,
    },
}

KERNELS: dict[str, Callable[..., Any]] = {
    f"vllm.v1.worker.gpu.{module}.{kernel}": fn
    for module, kernels in _KERNELS_BY_MODULE.items()
    for kernel, fn in kernels.items()
}
