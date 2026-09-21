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

"""The torch stand-ins for upstream V2's triton kernels, driven through the
upstream wrappers on CPU tensors and checked against a plain-python reference
of each kernel."""

import numpy as np
import pytest
import torch
from vllm.triton_utils.importing import PlaceholderKernel
from vllm.v1.worker.gpu import input_batch as ib
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import apply_bad_words
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.penalties import apply_penalties, bincount
from vllm.v1.worker.gpu.sample.prompt_logprob import get_prompt_logprobs_token_ids
from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel

MAX_REQS = 8
MAX_LEN = 32
VOCAB = 50


def _i32(x):
    return torch.tensor(x, dtype=torch.int32)


def _i64(x):
    return torch.tensor(x, dtype=torch.int64)


def test_kernels_resolve_to_torch_implementations():
    from vllm.platforms import current_platform

    for kernel in (ib._prepare_prefill_inputs_kernel, _apply_grammar_bitmask_kernel):
        assert isinstance(kernel, PlaceholderKernel)
        assert current_platform.get_kernel_impl(kernel.qualname) is not None


def test_prepare_prefill_inputs_copies_prompt_chunks_and_next_token():
    all_token_ids = torch.arange(MAX_REQS * MAX_LEN, dtype=torch.int32).view(
        MAX_REQS, MAX_LEN
    )
    # req 5 prefills 4 tokens from 3, req 2 decodes, req 7 prefills to its end
    idx_mapping = _i32([5, 2, 7])
    query_start_loc = _i32([0, 4, 5, 8])
    prefill_len = _i32([0] * MAX_REQS)
    prefill_len[[5, 2, 7]] = _i32([10, 6, 8])
    num_computed = _i32([0] * MAX_REQS)
    num_computed[[5, 2, 7]] = _i32([3, 9, 5])
    input_ids = torch.full((8,), -1, dtype=torch.int32)
    next_prefill = torch.full((MAX_REQS,), -1, dtype=torch.int32)

    ib.prepare_prefill_inputs(
        input_ids,
        next_prefill,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed,
    )

    assert input_ids[:4].tolist() == all_token_ids[5, 3:7].tolist()
    assert input_ids[4].item() == -1  # decoding request untouched
    assert input_ids[5:8].tolist() == all_token_ids[7, 5:8].tolist()
    assert next_prefill[5].item() == all_token_ids[5, 7].item()
    assert next_prefill[7].item() == -1  # reached prefill_len: no next token


def test_prepare_pos_seq_lens_and_padding():
    idx_mapping = _i32([3, 1])
    query_start_loc = _i32([0, 3, 4])
    num_computed = _i32([0] * MAX_REQS)
    num_computed[[3, 1]] = _i32([10, 20])
    pos = torch.zeros(8, dtype=torch.int64)
    seq_lens = torch.full((MAX_REQS,), 99, dtype=torch.int32)

    ib.prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed, pos, seq_lens)

    assert pos[:4].tolist() == [10, 11, 12, 20]
    assert seq_lens.tolist() == [13, 21] + [0] * (MAX_REQS - 2)


@pytest.mark.parametrize("with_drafts", [False, True])
def test_combine_sampled_and_draft_tokens(with_drafts):
    num_spec = 2
    idx_mapping = _i32([4, 6])
    last_sampled = torch.zeros(MAX_REQS, 1, dtype=torch.int64)
    last_sampled[4] = 111
    last_sampled[6] = 222
    prefill_len = torch.full((MAX_REQS,), 5, dtype=torch.int32)
    draft_tokens = torch.zeros(MAX_REQS, num_spec, dtype=torch.int64)
    draft_tokens[4] = _i64([31, 32])
    if with_drafts:
        # req 4 runs 3 logits (1 bonus + 2 drafts), req 6 is still prefilling
        query_start_loc = _i32([0, 3, 4])
        cu_num_logits = _i32([0, 3, 4])
        seq_lens = _i32([20, 4])
        input_ids = torch.full((4,), -1, dtype=torch.int32)
        logits_indices = ib.combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            cu_num_logits,
            4,
        )
        assert logits_indices.tolist() == [0, 1, 2, 3]
        assert input_ids.tolist() == [111, 31, 32, -1]
    else:
        query_start_loc = _i32([0, 1, 2])
        cu_num_logits = _i32([0, 1, 2])
        seq_lens = _i32([20, 21])
        input_ids = torch.full((2,), -1, dtype=torch.int32)
        logits_indices = ib.combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            cu_num_logits,
            2,
        )
        assert logits_indices.tolist() == [0, 1]
        assert input_ids.tolist() == [111, 222]


def test_get_num_sampled_and_rejected_zeroes_chunked_prefills():
    num_sampled = _i32([1, 1, 2])
    seq_lens = _i32([3, 10, 12])
    cu_num_logits = _i32([0, 1, 2, 5])
    idx_mapping = _i32([0, 1, 2])
    prefill_len = _i32([5, 5, 5, 0, 0, 0, 0, 0])
    sampled, rejected = ib.get_num_sampled_and_rejected(
        num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
    )
    assert sampled.tolist() == [0, 1, 2]
    assert rejected.tolist() == [0, 0, 1]


def test_post_update_appends_tokens_and_advances_state():
    idx_mapping = _i32([2, 5, -1])
    num_computed = _i32([0] * MAX_REQS)
    num_computed[[2, 5]] = _i32([7, 9])
    last_sampled = torch.zeros(MAX_REQS, 1, dtype=torch.int64)
    bin_counts = torch.zeros(MAX_REQS, VOCAB, dtype=torch.int32)
    sampled_tokens = _i64([[3, 4, 0], [7, 0, 0], [9, 9, 9]])
    num_sampled = _i32([2, 1, 3])
    num_rejected = _i32([1, 0, 0])
    query_start_loc = _i32([0, 3, 4, 7])
    all_token_ids = torch.zeros(MAX_REQS, MAX_LEN, dtype=torch.int32)
    total_len = torch.zeros(MAX_REQS, dtype=torch.int32)
    total_len[[2, 5]] = _i32([10, 12])

    ib.post_update(
        idx_mapping,
        num_computed,
        last_sampled,
        bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )

    assert all_token_ids[2, 10:12].tolist() == [3, 4]
    assert all_token_ids[5, 12].item() == 7
    assert last_sampled[2].item() == 4 and last_sampled[5].item() == 7
    assert total_len[2].item() == 12 and total_len[5].item() == 13
    assert num_computed[2].item() == 7 + 3 - 1 and num_computed[5].item() == 9 + 1
    assert bin_counts[2, 3].item() == 1 and bin_counts[2, 4].item() == 1
    assert bin_counts[5, 7].item() == 1
    assert bin_counts.sum().item() == 3  # the -1 row was skipped


def test_expand_idx_mapping():
    idx_mapping = _i32([4, 1])
    cu_num_logits = _i32([0, 3, 4])
    expanded, local_pos = ib.expand_idx_mapping(idx_mapping, 4, cu_num_logits, 3)
    assert expanded.tolist() == [4, 4, 4, 1]
    assert local_pos.tolist() == [0, 1, 2, 0]


def test_apply_temperature_and_min_p():
    logits = torch.tensor([[2.0, 4.0, 0.0], [2.0, 4.0, 0.0], [2.0, 4.0, 0.0]])
    eim = _i32([0, 1, 2])
    temperature = torch.tensor([2.0, 1.0, 0.0])
    apply_temperature(logits, eim, temperature)
    assert logits.tolist() == [[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [2.0, 4.0, 0.0]]

    logits = torch.tensor([[0.0, 4.0, 3.0], [0.0, 4.0, 3.0]])
    min_p = torch.tensor([0.3, 0.0])  # threshold 4 + log(0.3) ~ 2.8
    apply_min_p(logits, eim[:2], min_p)
    assert logits[0].tolist() == [float("-inf"), 4.0, 3.0]
    assert logits[1].tolist() == [0.0, 4.0, 3.0]


def test_gumbel_sample_greedy_argmax_and_seeded_determinism():
    logits = torch.randn(3, VOCAB)
    logits[2] = logits[1]  # rows 1 and 2: same logits, seed and position
    eim = _i32([0, 1, 2])
    temperature = torch.tensor([0.0, 1.0, 1.0])
    seeds = _i64([0, 123, 123])
    pos = _i64([5, 7, 7])
    first = gumbel_sample(
        logits.clone(), eim, temperature, seeds, pos, apply_temperature=True
    )
    second = gumbel_sample(
        logits.clone(), eim, temperature, seeds, pos, apply_temperature=True
    )
    assert first[0].item() == logits[0].argmax().item()
    assert first.tolist() == second.tolist()
    assert first[1].item() == first[2].item()  # same seed, position and logits


def test_penalties_match_reference():
    vocab = 40
    req_states = 2
    all_token_ids = torch.zeros(req_states, MAX_LEN, dtype=torch.int32)
    all_token_ids[1, :6] = _i32([1, 2, 3, 2, 7, 7])  # prompt [1,2,3,2], output [7,7]
    prompt_len = _i32([0, 4])
    prefill_len = _i32([0, 6])
    prompt_bin_mask = torch.zeros(req_states, (vocab + 31) // 32, dtype=torch.int32)
    output_bin_counts = torch.zeros(req_states, vocab, dtype=torch.int32)
    bincount(
        _i32([1]),
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        6,
    )
    assert output_bin_counts[1, 7].item() == 2
    for tok in (1, 2, 3):
        assert (prompt_bin_mask[1, tok // 32] >> (tok % 32)) & 1 == 1

    logits = torch.randn(1, vocab)
    expected = logits.clone()[0]
    rep, freq, pres = 1.5, 0.3, 0.2
    counts = output_bin_counts[1].float()
    in_prompt_or_output = torch.zeros(vocab, dtype=torch.bool)
    in_prompt_or_output[[1, 2, 3, 7]] = True
    scale = torch.where(in_prompt_or_output, torch.tensor(rep), torch.tensor(1.0))
    expected = expected * torch.where(expected > 0, 1.0 / scale, scale)
    expected = expected - freq * counts - pres * (counts > 0).float()

    apply_penalties(
        logits,
        _i32([1]),
        _i32([0]),
        _i32([0]),
        torch.tensor([1.0, rep]),
        torch.tensor([0.0, freq]),
        torch.tensor([0.0, pres]),
        prompt_bin_mask,
        output_bin_counts,
    )
    torch.testing.assert_close(logits[0], expected)


def test_logit_bias_allowed_bias_and_min_tokens():
    logits = torch.zeros(2, VOCAB)
    eim = _i32([0, 1])
    pos = _i64([3, 3])
    num_allowed = _i32([2, 0])
    allowed = torch.zeros(2, 4, dtype=torch.int32)
    allowed[0, :2] = _i32([5, 6])
    num_bias = _i32([0, 1])
    bias_tok = torch.zeros(2, 4, dtype=torch.int32)
    bias_tok[1, 0] = 9
    bias = torch.zeros(2, 4)
    bias[1, 0] = -2.5
    min_lens = _i32([0, 10])
    num_stop = _i32([0, 1])
    stop_tok = torch.zeros(2, 4, dtype=torch.int32)
    stop_tok[1, 0] = 1
    apply_logit_bias(
        logits,
        eim,
        pos,
        num_allowed,
        allowed,
        num_bias,
        bias_tok,
        bias,
        min_lens,
        num_stop,
        stop_tok,
    )
    assert (
        torch.isinf(logits[0]).sum().item() == VOCAB - 2
        and logits[0, 5] == 0
        and logits[0, 6] == 0
    )
    assert logits[1, 9].item() == -2.5 and logits[1, 1].item() == float("-inf")
    assert torch.isfinite(logits[1, 2:]).all()


def test_bad_words_masks_completion_of_matching_prefix():
    logits = torch.zeros(1, VOCAB)
    eim = _i32([3])
    bw_tok = torch.zeros(MAX_REQS, 8, dtype=torch.int32)
    bw_off = torch.zeros(MAX_REQS, 4, dtype=torch.int32)
    bw_tok[3, :5] = _i32([10, 11, 12, 20, 21])  # words [10,11,12] and [20,21]
    bw_off[3, :3] = _i32([0, 3, 5])
    num_bw = _i32([0, 0, 0, 2, 0, 0, 0, 0])
    all_token_ids = torch.zeros(MAX_REQS, MAX_LEN, dtype=torch.int32)
    all_token_ids[3, :6] = _i32([1, 1, 1, 1, 10, 11])  # prompt 4, output [10, 11]
    prompt_len = torch.full((MAX_REQS,), 4, dtype=torch.int32)
    total_len = torch.full((MAX_REQS,), 6, dtype=torch.int32)
    apply_bad_words(
        logits,
        eim,
        bw_tok,
        bw_off,
        num_bw,
        all_token_ids,
        prompt_len,
        total_len,
        _i32([0]),
        _i32([0]),
        2,
    )
    assert logits[0, 12].item() == float("-inf")
    assert torch.isfinite(logits[0, 21])  # [20] does not precede


def test_topk_scores_match_log_softmax():
    logits = torch.randn(3, VOCAB)
    sampled = logits.argmax(-1)
    out = compute_topk_scores(logits, 2, sampled)
    ref = torch.log_softmax(logits, -1)
    torch.testing.assert_close(out.logprobs, ref.gather(1, out.logprob_token_ids))
    assert out.selected_token_ranks.tolist() == [1, 1, 1]
    assert out.logprob_token_ids[:, 0].tolist() == sampled.tolist()


def test_prompt_logprobs_token_ids_and_num_nans():
    all_token_ids = torch.arange(MAX_REQS * MAX_LEN, dtype=torch.int32).view(
        MAX_REQS, MAX_LEN
    )
    ids = get_prompt_logprobs_token_ids(
        3, _i32([0, 2, 3]), _i32([1, 4]), _i32([0, 5, 0, 0, 9]), all_token_ids
    )
    assert ids.tolist() == [
        all_token_ids[1, 6].item(),
        all_token_ids[1, 7].item(),
        all_token_ids[4, 10].item(),
    ]

    logits = torch.zeros(2, 5)
    logits[1, [0, 3]] = float("nan")
    assert get_num_nans(logits).tolist() == [0, 2]


def test_grammar_bitmask_masks_cleared_bits():
    logits = torch.zeros(3, 40)
    bitmask = torch.zeros(1, 2, dtype=torch.int32)
    bitmask[0, 0] = (
        0b101  # tokens 0 and 2 allowed in the first word, none in the second
    )
    _apply_grammar_bitmask_kernel[(1, 1)](
        logits, logits.stride(0), _i32([2]), bitmask, bitmask.stride(0), 40
    )
    assert torch.isfinite(logits[0]).all() and torch.isfinite(logits[1]).all()
    assert logits[2, 0] == 0 and logits[2, 2] == 0
    assert torch.isinf(logits[2]).sum().item() == 38


@pytest.mark.maybe_use_device
def test_staged_write_tensor_flushes_rows():
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    device = torch.device("cpu")
    t = StagedWriteTensor((4, 6), dtype=torch.int32, device=device)
    t.stage_write(2, 1, [7, 8, 9])
    t.stage_write(0, 0, [1])
    t.apply_write()
    assert t.gpu[2].tolist() == [0, 7, 8, 9, 0, 0]
    assert t.gpu[0, 0].item() == 1
    flat = StagedWriteTensor(4, dtype=torch.int32, device=device)
    flat.stage_write_elem(3, 5)
    flat.apply_write()
    assert flat.gpu.tolist() == [0, 0, 0, 5]
    assert np.array_equal(t.gpu.numpy()[1], np.zeros(6, dtype=np.int32))


def _rejection_inputs(num_reqs, cu, temps, seeds=None):
    cu_num_logits = _i32(cu)
    num_logits = cu[-1]
    idx_mapping = _i32(list(range(num_reqs)))
    lens = [cu[r + 1] - cu[r] for r in range(num_reqs)]
    expanded_idx_mapping = _i32([r for r, n in enumerate(lens) for _ in range(n)])
    expanded_local_pos = _i32([i for n in lens for i in range(n)])
    temperature = torch.zeros(MAX_REQS)
    temperature[:num_reqs] = torch.tensor(temps)
    seed = _i64(seeds or [0] * MAX_REQS)
    pos = _i32(list(range(num_logits)))
    return (
        cu_num_logits,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        pos,
    )


def test_rejection_sample_greedy_accepts_matching_prefix():
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs

    # req 0: drafts [3, 7] with target argmax [3, 9, 5] -> accept 3, recover 9
    # req 1: drafts [1] fully accepted -> bonus argmax 4 appended
    # req 2: no drafts -> plain bonus 2
    cu = [0, 3, 5, 6]
    logits = torch.full((6, VOCAB), -10.0)
    for row, tok in enumerate([3, 9, 5, 1, 4, 2]):
        logits[row, tok] = 0.0
    draft_sampled = _i32([-1, 3, 7, -1, 1, -1])
    cu_num_logits, idx_mapping, expanded_idx, expanded_pos, temperature, seed, pos = (
        _rejection_inputs(3, cu, [0.0, 0.0, 0.0])
    )
    sampled, num_sampled = rs.rejection_sample(
        logits,
        None,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx,
        expanded_pos,
        temperature,
        seed,
        num_speculative_steps=2,
    )
    assert num_sampled.tolist() == [2, 2, 1]
    assert sampled[0, :2].tolist() == [3, 9]
    assert sampled[1, :2].tolist() == [1, 4]
    assert sampled[2, :1].tolist() == [2]


def test_rejection_sample_random_never_resamples_rejected_draft():
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs

    # Target puts no mass on the draft token, so every seed rejects it, and
    # the recovered token must come from the residual: anything but 7.
    cu = [0, 2]
    logits = torch.zeros(2, VOCAB)
    logits[0, 7] = float("-inf")
    draft_sampled = _i32([-1, 7])
    for s in range(5):
        (
            cu_num_logits,
            idx_mapping,
            expanded_idx,
            expanded_pos,
            temperature,
            seed,
            pos,
        ) = _rejection_inputs(1, cu, [1.0], seeds=[s] * MAX_REQS)
        sampled, num_sampled = rs.rejection_sample(
            logits,
            None,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx,
            expanded_pos,
            temperature,
            seed,
            num_speculative_steps=1,
        )
        assert num_sampled.tolist() == [1]
        assert sampled[0, 0].item() != 7


def test_flatten_sampled_places_each_request_at_its_logit_offset():
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs

    sampled = _i64([[10, 11, 12], [20, 21, 22]])
    num_sampled = _i32([2, 1])
    cu_num_logits = _i32([0, 3, 6])
    flat = torch.zeros(6, dtype=torch.int64)
    rs._flatten_sampled_kernel[(2,)](
        flat, sampled, sampled.stride(0), num_sampled, cu_num_logits, num_warps=1
    )
    assert flat.tolist() == [10, 11, 0, 20, 0, 0]


def test_draft_prefill_inputs_shift_tokens_and_place_next_token():
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as sp

    # req 0: 4 target tokens, 1 rejected -> draft consumes 3; req 1: 2 tokens.
    buffers = InputBuffers(max_num_reqs=MAX_REQS, max_num_tokens=16, device="cpu")
    target_ids = _i32([10, 11, 12, 13, 20, 21])
    target_pos = _i64([5, 6, 7, 8, 0, 1])
    last_token_indices = torch.zeros(MAX_REQS, dtype=torch.int64)
    step = torch.tensor(7)
    sp._prepare_prefill_inputs_kernel[(2,)](
        last_token_indices,
        step,
        buffers.input_ids,
        buffers.positions,
        buffers.query_start_loc,
        buffers.seq_lens,
        target_ids,
        target_pos,
        _i32([3, 1]),  # idx_mapping
        _i64([[0], [0], [0], [99], [0], [0], [0], [0]]),  # last_sampled per state
        _i32([0, 55, 0, 0, 0, 0, 0, 0]),  # next_prefill_tokens per state
        _i32([1, 0]),  # req 0 sampled, req 1 still prefilling
        _i32([1, 0]),  # num_rejected
        _i32([0, 4, 6]),  # query_start_loc
        _i32([9, 2]),  # seq_lens
        MAX_REQS,
        BLOCK_SIZE=1024,
    )
    assert buffers.input_ids[:6].tolist() == [11, 12, 99, 0, 21, 55]
    assert last_token_indices[:2].tolist() == [2, 5]
    assert buffers.positions[:6].tolist() == [5, 6, 7, 0, 0, 1]
    assert buffers.query_start_loc[: MAX_REQS + 1].tolist() == [0, 4] + [6] * 7
    assert buffers.seq_lens[:3].tolist() == [9, 2, 0]
    assert step.item() == 0


def test_draft_decode_and_update_inputs_advance_positions():
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as sp

    buffers = InputBuffers(max_num_reqs=MAX_REQS, max_num_tokens=16, device="cpu")
    buffers.positions[:2] = _i64([7, 30])
    draft_tokens = _i64([[3, 0], [4, 0]])
    sp._prepare_decode_inputs_kernel[(3,)](
        draft_tokens[:, 0],
        draft_tokens.stride(0),
        _i32([9, 31]),  # target seq_lens
        _i32([1, 0]),  # num_rejected
        buffers.input_ids,
        buffers.positions,
        buffers.query_start_loc,
        buffers.seq_lens,
        32,  # max_model_len
        MAX_REQS,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=True,
    )
    assert buffers.input_ids[:2].tolist() == [3, 4]
    assert buffers.positions[:2].tolist() == [8, 31]  # clamped to max_model_len - 1
    assert buffers.seq_lens[:3].tolist() == [9, 32, 0]
    assert buffers.query_start_loc[:4].tolist() == [0, 1, 2, 2]

    hidden = torch.arange(2 * 4, dtype=torch.float32).view(2, 4)
    next_hidden = torch.zeros(MAX_REQS, 4)
    step = torch.tensor(0)
    sp._update_draft_inputs_kernel[(2,)](
        draft_tokens,
        draft_tokens.stride(0),
        next_hidden,
        next_hidden.stride(0),
        buffers.input_ids,
        buffers.positions,
        buffers.seq_lens,
        _i64([5, 6]),  # this step's draft tokens
        step,
        hidden,
        hidden.stride(0),
        4,
        32,
        2,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=True,
    )
    assert draft_tokens[:, 0].tolist() == [5, 6]
    assert buffers.input_ids[:2].tolist() == [5, 6]
    assert torch.equal(next_hidden[:2], hidden)
    assert buffers.positions[:2].tolist() == [9, 31]
