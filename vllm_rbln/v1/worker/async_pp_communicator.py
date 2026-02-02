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

This module wraps the blocking ``send_tensor_dict`` call in a background
thread so that the sending stage can immediately begin computing the next
batch.  The engine core's ``step_with_batch_queue`` keeps multiple
``SchedulerOutput`` objects in flight (up to ``pp_size``), which fills the
pipeline and hides latency.

Correctness guarantee
---------------------
``wait_pending_send()`` **must** be called before the model runner
re-enters ``execute_model``, because the pre-allocated intermediate-tensor
buffers are reused across batches.  The current call-site places the wait
at the top of ``RBLNWorker.execute_model``, satisfying this invariant.
"""

import threading
from typing import Any, Optional

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


class AsyncPPCommunicator:
    """Thread-based async sender for pipeline-parallel P2P communication.

    One in-flight send is allowed at a time.  The caller is responsible for
    invoking ``wait_pending_send`` before the underlying tensor buffers are
    overwritten (see module docstring).
    """

    def __init__(self, pp_group) -> None:
        self._pp_group = pp_group
        self._send_thread: Optional[threading.Thread] = None
        self._send_error: Optional[BaseException] = None
        self._error_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def async_send_tensor_dict(
        self,
        tensor_dict: dict[str, Any],
    ) -> None:
        """Start a non-blocking send of *tensor_dict* to the next PP stage.

        The actual ``send_tensor_dict`` runs in a daemon thread.  The
        caller **must not** mutate *tensor_dict* (or the tensors it
        references) until the next ``wait_pending_send`` returns.
        """
        # Defensive: should already have been called by the worker, but
        # guard against misuse.
        self.wait_pending_send()

        self._send_error = None
        self._send_thread = threading.Thread(
            target=self._do_send,
            args=(tensor_dict, ),
            daemon=True,
        )
        self._send_thread.start()

    def wait_pending_send(self) -> None:
        """Block until the in-flight send completes (if any).

        Re-raises any exception captured by the background thread.
        """
        if self._send_thread is None:
            return

        self._send_thread.join()
        self._send_thread = None

        with self._error_lock:
            if self._send_error is not None:
                err = self._send_error
                self._send_error = None
                raise RuntimeError("Async PP send failed") from err

    def shutdown(self) -> None:
        """Best-effort cleanup — wait for the last send."""
        try:
            self.wait_pending_send()
        except Exception:
            logger.warning("Error during AsyncPPCommunicator shutdown",
                           exc_info=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_send(self, tensor_dict: dict[str, Any]) -> None:
        try:
            self._pp_group.send_tensor_dict(tensor_dict)
        except BaseException as exc:
            with self._error_lock:
                self._send_error = exc
            logger.error("Async PP send raised: %s", exc)
