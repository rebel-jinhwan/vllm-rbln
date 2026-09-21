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

"""Method-level replacements in upstream's V2 model runner for what RBLN's
torch build cannot run: the PP sampled-token broadcast, which upstream runs
on a side stream over a sibling NCCL group, and the draft-token hand-off,
which calls `Tensor.record_stream`. The runner's Triton kernels are not
patched: the `vllm_rbln.v2` components override upstream's kernel methods.
"""

from collections import deque

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import (
    PendingRecv,
    PPHandler,
    compute_need_sampled_mask,
)

from vllm_rbln.patches.registry import register_patch

_NO_CUDA_PP_REASON = (
    "PPHandler broadcasts the last stage's sampled tokens over a sibling device "
    "group (make_sibling_device_group) on a side stream and hands the buffers "
    "across streams with Tensor.record_stream. RBLN's communicator has no "
    "sibling groups and torch_rbln does not implement record_stream, so the "
    "broadcast runs synchronously on the PP group's CPU process group."
)


def _to_cpu(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()


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


def _draft_tokens_handler_set(self, input_batch: InputBatch, draft_tokens) -> None:
    self.req_ids = input_batch.req_ids
    self.num_draft_tokens = draft_tokens.shape[1]
    self.draft_tokens_np = (
        _to_cpu(draft_tokens) if input_batch.has_structured_output_reqs else None
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

register_patch(
    target="vllm.v1.worker.gpu.spec_decode.utils.DraftTokensHandler.set_draft_tokens",
    reason=(
        "DraftTokensHandler.set_draft_tokens calls Tensor.record_stream on the "
        "draft tokens it copies out on a side stream; torch_rbln does not "
        "implement record_stream, so the copy runs synchronously instead."
    ),
)(_draft_tokens_handler_set)
