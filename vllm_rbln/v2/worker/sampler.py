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

"""Upstream's V2 `Sampler` on RBLN: its logic stays device-neutral and each
Triton kernel is a method, implemented here with the `rbln::` custom ops of
rebel-compiler (`rebel.ops.torch_custom_ops.lm_serving`)."""

import torch
from vllm.v1.worker.gpu.sample.sampler import Sampler

ops = torch.ops.rbln


class RBLNSamplerV2(Sampler):
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
