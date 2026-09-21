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

"""Upstream's V2 `PromptLogprobsWorker` on RBLN, with its kernel as an `rbln::` op."""

import torch
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

ops = torch.ops.rbln


class RBLNPromptLogprobsWorker(PromptLogprobsWorker):
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
