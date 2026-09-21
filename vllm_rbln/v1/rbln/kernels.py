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

"""Upstream's V2 model runner kernels on RBLN. Every kernel is an `rbln::`
custom op of rebel-compiler (`rebel.ops.torch_custom_ops.lm_serving`), so the
runner, its sampler stack and the drafter keep upstream's logic and this class
only forwards the call."""

import torch
from vllm.v1.worker.kernels import ModelRunnerKernels

ops = torch.ops.rbln


class RBLNKernels(ModelRunnerKernels):
    def prepare_prefill_inputs(
        self,
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    ) -> None:
        ops.prepare_prefill_inputs(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_len,
            num_computed_tokens,
        )

    def prepare_pos_seq_lens(
        self, idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens
    ) -> None:
        ops.prepare_pos_seq_lens(
            idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens
        )

    def combine_sampled_and_draft_tokens(
        self,
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits,
        num_new_sampled_tokens,
    ) -> torch.Tensor:
        return ops.combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            cu_num_logits,
            num_logits,
            num_new_sampled_tokens,
        )

    def expand_idx_mapping(
        self, idx_mapping, total_num_logits, cu_num_logits, max_expand_len
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ops.expand_idx_mapping(
            idx_mapping, total_num_logits, cu_num_logits, max_expand_len
        )

    def post_update(
        self,
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    ) -> None:
        ops.post_update(
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            total_len,
        )

    def post_update_num_computed_tokens(
        self, idx_mapping, num_computed_tokens, query_start_loc
    ) -> None:
        ops.post_update_num_computed_tokens(
            idx_mapping, num_computed_tokens, query_start_loc
        )

    def apply_temperature(self, logits, expanded_idx_mapping, temperature) -> None:
        ops.apply_temperature(logits, expanded_idx_mapping, temperature)

    def apply_min_p(self, logits, expanded_idx_mapping, min_p) -> None:
        ops.apply_min_p(logits, expanded_idx_mapping, min_p)

    def apply_penalties(
        self,
        logits,
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
    ) -> None:
        ops.apply_penalties(
            logits,
            expanded_idx_mapping,
            token_ids,
            expanded_local_pos,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            prompt_bin_mask,
            output_bin_counts,
        )

    def bincount(
        self,
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        max_prefill_len,
    ) -> None:
        ops.bincount(
            expanded_idx_mapping,
            all_token_ids,
            prompt_len,
            prefill_len,
            prompt_bin_mask,
            output_bin_counts,
            max_prefill_len,
        )

    def apply_logit_bias(
        self,
        logits,
        expanded_idx_mapping,
        pos,
        num_allowed_token_ids,
        allowed_token_ids,
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias,
        min_lens,
        num_stop_token_ids,
        stop_token_ids,
    ) -> None:
        ops.apply_logit_bias(
            logits,
            expanded_idx_mapping,
            pos,
            num_allowed_token_ids,
            allowed_token_ids,
            num_logit_bias,
            logit_bias_token_ids,
            logit_bias,
            min_lens,
            num_stop_token_ids,
            stop_token_ids,
        )

    def apply_bad_words(
        self,
        logits,
        expanded_idx_mapping,
        bad_word_token_ids,
        bad_word_offsets,
        num_bad_words,
        all_token_ids,
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
        max_num_bad_words,
    ) -> None:
        ops.apply_bad_words(
            logits,
            expanded_idx_mapping,
            bad_word_token_ids,
            bad_word_offsets,
            num_bad_words,
            all_token_ids,
            prompt_len,
            total_len,
            input_ids,
            expanded_local_pos,
            max_num_bad_words,
        )

    def gumbel_sample(
        self,
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        output_processed_logits=None,
        output_processed_logits_col=None,
        use_fp64=False,
    ) -> torch.Tensor:
        return ops.gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature,
            output_processed_logits,
            output_processed_logits_col,
            use_fp64,
        )

    def compute_token_logprobs(self, logits, token_ids) -> torch.Tensor:
        return ops.token_logprobs(logits, token_ids)

    def compute_token_ranks(self, logits, token_ids) -> torch.Tensor:
        return ops.token_ranks(logits, token_ids)

    def fill_logprob_token_ids(
        self,
        out_token_ids,
        out_valid_mask,
        sampled_token_ids,
        topk_token_ids,
        expanded_idx_mapping,
        num_per_req_token_ids,
        per_req_token_ids,
        num_topk,
    ) -> None:
        ops.fill_logprob_token_ids(
            out_token_ids,
            out_valid_mask,
            sampled_token_ids,
            topk_token_ids,
            expanded_idx_mapping,
            num_per_req_token_ids,
            per_req_token_ids,
            num_topk,
        )

    def get_num_nans(self, logits) -> torch.Tensor:
        return ops.num_nans(logits)

    def get_num_sampled_and_rejected(
        self, num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ops.num_sampled_and_rejected(
            num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
        )

    def get_prompt_logprobs_token_ids(
        self,
        num_tokens,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
    ) -> torch.Tensor:
        return ops.prompt_logprobs_token_ids(
            num_tokens, query_start_loc, idx_mapping, num_computed_tokens, all_token_ids
        )

    def rejection_sample(
        self,
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps,
        synthetic_conditional_rates=None,
        use_fp64=False,
        use_block_verification=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if synthetic_conditional_rates is not None or use_block_verification:
            raise NotImplementedError(
                "RBLN rejection sampling supports rejection_sample_method='standard' "
                "only"
            )
        return ops.rejection_sample_logits(
            target_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
            num_speculative_steps,
            use_fp64,
        )

    def flatten_sampled(
        self, flat_sampled, sampled, num_sampled, cu_num_logits
    ) -> None:
        ops.flatten_sampled(flat_sampled, sampled, num_sampled, cu_num_logits)

    def apply_grammar_bitmask(self, logits, logits_indices, bitmask) -> None:
        ops.apply_grammar_bitmask(logits, logits_indices, bitmask)

    def prepare_draft_prefill_inputs(
        self,
        last_token_indices,
        current_draft_step,
        input_buffers,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        max_num_reqs,
    ) -> torch.Tensor:
        ops.draft_prepare_prefill_inputs(
            last_token_indices,
            current_draft_step,
            input_buffers.input_ids,
            input_buffers.positions,
            input_buffers.query_start_loc,
            input_buffers.seq_lens,
            input_batch.input_ids,
            input_batch.positions,
            input_batch.idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
            input_batch.seq_lens,
            max_num_reqs,
        )
        return last_token_indices

    def prepare_draft_decode_inputs(
        self,
        draft_tokens,
        target_seq_lens,
        num_rejected,
        input_buffers,
        max_model_len,
        max_num_reqs,
        advance_draft_positions=True,
    ) -> None:
        ops.draft_prepare_decode_inputs(
            draft_tokens,
            target_seq_lens,
            num_rejected,
            input_buffers.input_ids,
            input_buffers.positions,
            input_buffers.query_start_loc,
            input_buffers.seq_lens,
            max_model_len,
            max_num_reqs,
            advance_draft_positions,
        )

    def update_draft_inputs(
        self,
        draft_tokens,
        current_draft_step,
        hidden_states,
        output_draft_tokens,
        next_input_hidden_states,
        input_buffers,
        num_reqs,
        max_model_len,
        num_speculative_steps,
        advance_draft_positions=True,
    ) -> None:
        ops.draft_update_inputs(
            output_draft_tokens,
            next_input_hidden_states,
            input_buffers.input_ids,
            input_buffers.positions,
            input_buffers.seq_lens,
            draft_tokens[:num_reqs],
            current_draft_step,
            hidden_states,
            max_model_len,
            num_speculative_steps,
            advance_draft_positions,
        )
