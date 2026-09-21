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

"""Method-level replacement in upstream's V2 model runner for what RBLN's
torch build cannot run. The runner's Triton kernels are not patched:
`vllm_rbln.v1.rbln.kernels` implements upstream's kernel interface.
"""

from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_rbln.patches.registry import register_patch


def _draft_tokens_handler_set(self, input_batch: InputBatch, draft_tokens) -> None:
    self.req_ids = input_batch.req_ids
    self.num_draft_tokens = draft_tokens.shape[1]
    self.draft_tokens_np = (
        draft_tokens.cpu().numpy() if input_batch.has_structured_output_reqs else None
    )


register_patch(
    target="vllm.v1.worker.gpu.spec_decode.utils.DraftTokensHandler.set_draft_tokens",
    reason=(
        "DraftTokensHandler.set_draft_tokens calls Tensor.record_stream on the "
        "draft tokens it copies out on a side stream; torch_rbln does not "
        "implement record_stream, so the copy runs synchronously instead."
    ),
)(_draft_tokens_handler_set)
