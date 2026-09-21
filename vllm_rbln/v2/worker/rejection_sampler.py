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

"""Upstream's V2 `RejectionSampler` on RBLN, with its kernels as `rbln::` ops."""

import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

ops = torch.ops.rbln


class RBLNRejectionSamplerV2(RejectionSampler):
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
