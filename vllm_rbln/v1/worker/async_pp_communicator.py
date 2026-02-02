# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Async P2P communicator for Chunked Pipeline Parallelism (CPP).

When pipeline parallelism is enabled, each pipeline stage processes batches
sequentially.  Without overlap, Stage 0 must wait for Stage 1 to receive
its output before starting the next batch — creating a pipeline bubble.

This module uses ``torch.distributed.isend`` (non-blocking send) so that
the sending stage can immediately begin computing the next batch.  The
engine core's ``step_with_batch_queue`` keeps multiple ``SchedulerOutput``
objects in flight (up to ``pp_size``), which fills the pipeline and hides
communication latency.

Protocol
--------
``send_tensor_dict`` in vLLM's ``GroupCoordinator`` sends:

1. **Metadata** (blocking) — pickle-serialized list of ``(key, value)``
   pairs where tensor values are replaced by ``TensorMetadata``.
   Two blocking sends: size tensor, then serialized object.
2. **Tensor data** — one ``send()`` per tensor in dict order.

We keep step 1 blocking (receiver needs metadata to allocate buffers)
and replace step 2 with ``isend()`` for non-blocking tensor transfer.

Correctness guarantee
---------------------
``wait_pending_send()`` **must** be called before the model runner
re-enters ``execute_model``, because the pre-allocated intermediate-tensor
buffers are reused across batches.  The current call-site places the wait
at the top of ``RBLNWorker.execute_model``, satisfying this invariant.
"""

from typing import Any

import torch
import torch.distributed

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


class AsyncPPCommunicator:
    """Non-blocking sender for pipeline-parallel P2P communication.

    Uses ``torch.distributed.isend`` for tensor data to overlap
    communication with computation.  Metadata is still sent blocking
    because the receiver needs it to allocate tensor buffers.

    One set of in-flight sends is allowed at a time.  The caller is
    responsible for invoking ``wait_pending_send`` before the underlying
    tensor buffers are overwritten (see module docstring).
    """

    def __init__(self, pp_group) -> None:
        self._pp_group = pp_group
        self._pending_works: list[torch.distributed.Work] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def async_send_tensor_dict(
        self,
        tensor_dict: dict[str, Any],
    ) -> None:
        """Send *tensor_dict* to the next PP stage with non-blocking
        tensor transfer.

        Metadata is sent synchronously (receiver needs it to allocate
        buffers).  Tensor payloads use ``isend`` and can be waited on
        later via ``wait_pending_send``.

        The caller **must not** mutate *tensor_dict* (or the tensors it
        references) until the next ``wait_pending_send`` returns.
        """
        # Ensure previous sends completed before starting new ones.
        self.wait_pending_send()

        pg = self._pp_group

        # Determine destination (next rank in PP ring).
        dst_local = (pg.rank_in_group + 1) % pg.world_size
        dst_global = pg.ranks[dst_local]
        metadata_group = pg.cpu_group

        # --- Step 1: metadata (blocking) ---
        # Reuse vLLM's existing protocol: send_object sends
        # (size_tensor, object_tensor) via two blocking sends.
        # _split_tensor_dict is a module-level helper in parallel_state.
        from vllm.distributed.parallel_state import _split_tensor_dict

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        pg.send_object(metadata_list, dst=dst_local)

        # --- Step 2: tensor data (non-blocking via isend) ---
        works: list[torch.distributed.Work] = []
        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            work = torch.distributed.isend(
                tensor,
                dst=dst_global,
                group=metadata_group,
            )
            works.append(work)

        self._pending_works = works

    def wait_pending_send(self) -> None:
        """Block until all in-flight ``isend`` operations complete.

        Must be called before reusing the tensor buffers that were passed
        to ``async_send_tensor_dict``.
        """
        for work in self._pending_works:
            work.wait()
        self._pending_works.clear()

    def shutdown(self) -> None:
        """Best-effort cleanup — wait for any outstanding sends."""
        try:
            self.wait_pending_send()
        except Exception:
            logger.warning("Error during AsyncPPCommunicator shutdown",
                           exc_info=True)
