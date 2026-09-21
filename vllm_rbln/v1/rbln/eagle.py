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

"""EAGLE drafting for RBLNModelRunnerV2 on upstream's V2 speculator.

Upstream's ``AutoRegressiveSpeculator`` owns the drafting loop and draft
sampling. This class overrides its hardware seams: ``dispatch_batch`` picks
the compiled shape, ``_run_model`` stages a pass into the compiled draft
graph, ``_build_draft_attn_metadata`` calls the RBLN builder, and the kernel
methods call ``rbln::`` ops. Upstream keeps tokens in flat token order; the
graph runs on padded ``[num_reqs, query_len]`` rows, so ``_run_model``
converts between the two on the way in and out.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_rbln import envs
from vllm_rbln.compilation import build_process_group_dict, compile
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.v1.worker.dp_utils import determine_draft_batch_execution_and_padding
from vllm_rbln.v1.worker.input_stager import InputLayout, InputStager

if TYPE_CHECKING:
    from vllm_rbln.v1.rbln.model_runner import RBLNModelRunnerV2


class RBLNEagleSpeculator(EagleSpeculator):
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, runner: "RBLNModelRunnerV2"
    ) -> None:
        super().__init__(vllm_config, device, runner.kernels)
        unsupported = [
            name
            for name, enabled in (
                ("multimodal drafts", self.supports_mm_inputs),
                (
                    "draft_sample_method="
                    f"{self.speculative_config.draft_sample_method!r}",
                    self.speculative_config.draft_sample_method != "greedy",
                ),
            )
            if enabled
        ]
        if unsupported:
            raise NotImplementedError(
                "RBLNModelRunnerV2 speculative decoding does not support "
                + ", ".join(unsupported)
            )
        self.runner = runner
        self.input_stager = InputStager(device)
        # Set by dispatch_batch for the pass that follows it.
        self._decode_pass = False
        self._num_reqs_padded = 0
        self._num_padded_tokens: int | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self) -> None:
        pass

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        hidden_size = self.hidden_size

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ):
            ret = self.model(
                input_ids=input_ids, positions=positions, hidden_states=hidden_states
            )
            last_hidden, hidden = ret if isinstance(ret, tuple) else (ret, ret)
            if last_hidden is hidden:
                # EAGLE returns one tensor twice; two aliased graph outputs
                # do not compile, so the pair is rebuilt in _run_model.
                return last_hidden.view(-1, hidden_size)
            return last_hidden.view(-1, hidden_size), hidden.view(-1, hidden_size)

        rbln_config = self.runner.rbln_config
        if self.speculative_config.enforce_eager or not rbln_config.compile_model:
            self.model_executable = model_wrapper
            return
        self.model_executable = compile(
            model_wrapper,
            dynamic=False,
            fullgraph=True,
            num_devices=rbln_config.num_devices_per_local_rank,
            model_trace_method="export",
            process_group_dict=build_process_group_dict(),
            guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
            runtime_holder=self.runner.runtime_holder,
            mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
            use_static_output=True,
            use_direct_dispatch=True,
        )

    def dispatch_batch(  # type: ignore[override]
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        *,
        decode: bool,
        need_eager: bool,
        **kwargs: Any,
    ) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
        runner = self.runner
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        batch_desc, num_tokens_across_dp = determine_draft_batch_execution_and_padding(
            cfg=runner.shape_config,
            status=None if dp_size == 1 else runner.dp_status,
            dp_rank=self.dp_rank,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            is_prefill=runner.is_prefill and not decode,
            draft_has_moe=False,
            first_pass=not decode,
        )
        self._decode_pass = decode
        self._num_reqs_padded = batch_desc.num_reqs_padded
        self._num_padded_tokens = batch_desc.num_tokens_padded
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=batch_desc.num_reqs_padded,
                uniform_token_count=uniform_token_count,
            ),
            num_tokens_across_dp,
        )

    def _run_model(  # type: ignore[override]
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._decode_pass:
            # One token per request: the flat order is the row order.
            num_reqs = num_tokens
            layout = InputLayout(
                num_reqs=num_reqs,
                num_reqs_padded=self._num_reqs_padded,
                query_len=1,
                query_len_padded=1,
            )
            token_rows = None
            positions = self.input_buffers.positions[:num_reqs].view(num_reqs, 1)
            input_ids = self.input_buffers.input_ids[:num_reqs].view(num_reqs, 1)
            hidden_states = self.hidden_states[:num_reqs].view(num_reqs, 1, -1)
        else:
            # The target step's rows: scatter the flat tokens into them.
            step = self.runner.step_shape
            assert step is not None
            layout = step.layout
            token_rows = step.token_rows
            num_flat = token_rows.shape[0]
            rows = layout.num_reqs * layout.query_len
            input_ids = self.input_buffers.input_ids.new_zeros(rows)
            input_ids[token_rows] = self.input_buffers.input_ids[:num_flat]
            input_ids = input_ids.view(layout.num_reqs, layout.query_len)
            hidden_states = self.hidden_states.new_zeros(rows, self.hidden_size)
            hidden_states[token_rows] = self.hidden_states[:num_flat]
            hidden_states = hidden_states.view(layout.num_reqs, layout.query_len, -1)
            positions = torch.from_numpy(step.positions)
        staged = self.input_stager.stage(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            layout=layout,
        )
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=layout.num_reqs_padded * layout.query_len_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=self._num_padded_tokens,
        ):
            out = self.model_executable(
                input_ids=staged.input_ids,
                positions=staged.positions,
                hidden_states=staged.hidden_states,
            )
        last_hidden, hidden = out if isinstance(out, tuple) else (out, out)
        if token_rows is None:
            return last_hidden[:num_reqs], hidden[:num_reqs]
        return last_hidden[token_rows], hidden[token_rows]

    def _build_draft_attn_metadata(  # type: ignore[override]
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        num_query_per_req: int = 1,
        causal: bool | Any = True,
    ) -> dict[str, Any] | None:
        assert num_query_per_req == 1
        positions = self.input_buffers.positions[:num_reqs].cpu()
        seq_lens = self.input_buffers.seq_lens[:num_reqs].cpu()
        return self.runner.rbln_attn_metadata(
            self.attn_groups,
            num_reqs,
            num_reqs_padded,
            torch.from_numpy(np.arange(num_reqs + 1, dtype=np.int32)),
            seq_lens,
            positions,
            tuple(self.block_tables.input_block_tables),
            False,
        )
