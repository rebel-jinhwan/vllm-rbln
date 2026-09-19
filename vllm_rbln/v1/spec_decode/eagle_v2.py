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

"""EAGLE drafting for RBLNModelRunnerV2 on upstream's V2 speculator interface.

Upstream's AutoRegressiveSpeculator drives the draft model eagerly on flat
token buffers. RBLN runs it as compiled per-shape graphs on the padded
``[num_reqs, query_len]`` rows the target step staged, so ``propose`` is
rewritten on that layout while model loading, weight sharing and the buffers
stay the base class's.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_rbln import envs
from vllm_rbln.compilation import build_process_group_dict, compile
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.v1.worker.dp_utils import determine_draft_batch_execution_and_padding
from vllm_rbln.v1.worker.input_stager import InputLayout, InputStager

if TYPE_CHECKING:
    from vllm_rbln.v1.worker.rbln_model_runner_v2 import RBLNModelRunnerV2


class RBLNEagleSpeculator(EagleSpeculator):
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, runner: "RBLNModelRunnerV2"
    ) -> None:
        super().__init__(vllm_config, device)
        unsupported = [
            name
            for name, enabled in (
                (f"method={self.method!r}", self.method != "eagle"),
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
            token_indices: torch.Tensor | None = None,
        ):
            ret = self.model(
                input_ids=input_ids, positions=positions, hidden_states=hidden_states
            )
            last_hidden, hidden = ret if isinstance(ret, tuple) else (ret, ret)
            hidden = hidden.view(-1, hidden_size)
            sample_hidden = last_hidden.view(-1, hidden_size)
            if token_indices is not None:
                hidden = hidden[token_indices]
                sample_hidden = sample_hidden[token_indices]
            logits = self.model.compute_logits(sample_hidden)
            return hidden, torch.ops.rbln.argmax(logits)

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

    def propose(  # type: ignore[override]
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        runner = self.runner
        step = runner.step_shape
        assert step is not None
        layout = step.layout
        num_reqs = input_batch.num_reqs
        query_len = layout.query_len
        device = self.device

        # The draft consumes the step's tokens minus the rejected tail, shifted
        # one to the left, with the token the target sampled last in the slot
        # the target's last accepted hidden state occupies.
        num_input = (
            input_batch.num_scheduled_tokens[:num_reqs] - num_rejected.cpu().numpy()
        )
        last = step.front_pad + num_input - 1
        cols = np.arange(query_len)
        idx = input_batch.idx_mapping.long()
        next_tokens = torch.where(
            num_sampled[:num_reqs] > 0,
            last_sampled[idx].view(-1).to(step.input_ids.dtype),
            next_prefill_tokens[idx].to(step.input_ids.dtype),
        )
        draft_ids = torch.gather(
            step.input_ids,
            1,
            torch.from_numpy(np.minimum(cols[None, :] + 1, last[:, None])).to(device),
        )
        draft_ids = torch.where(
            torch.from_numpy(cols[None, :] >= last[:, None]).to(device),
            next_tokens[:, None],
            draft_ids,
        )
        hidden = last_hidden_states.view(
            layout.num_reqs_padded, layout.query_len_padded, -1
        )[:num_reqs, :query_len]
        token_indices = torch.from_numpy(
            (np.arange(num_reqs) * layout.query_len_padded + last).astype(np.int32)
        )
        seq_lens = input_batch.num_computed_tokens_np[:num_reqs] + num_input
        hidden, ids = self._run(
            draft_ids,
            step.positions,
            hidden,
            token_indices,
            layout,
            seq_lens,
            step.block_tables,
            runner.is_prefill,
            num_reqs * query_len,
            first_pass=True,
        )
        self.draft_tokens[:num_reqs, 0] = ids[:num_reqs]
        if self.num_speculative_steps == 1:
            return self.draft_tokens[:num_reqs, :1]

        positions = step.positions[np.arange(num_reqs), last]
        for draft_step in range(1, self.num_speculative_steps):
            positions = np.minimum(positions + 1, self.max_model_len - 1)
            seq_lens = np.minimum(seq_lens + 1, self.max_model_len)
            batch_desc, _ = self._batch(num_reqs, num_reqs, False, first_pass=False)
            hidden, ids = self._run(
                ids[:num_reqs].view(-1, 1).to(step.input_ids.dtype),
                positions[:, None],
                hidden[:num_reqs].unsqueeze(1),
                None,
                InputLayout(
                    num_reqs=num_reqs,
                    num_reqs_padded=batch_desc.num_reqs_padded,
                    query_len=1,
                    query_len_padded=1,
                ),
                seq_lens,
                step.block_tables,
                False,
                num_reqs,
                first_pass=False,
            )
            self.draft_tokens[:num_reqs, draft_step] = ids[:num_reqs]
        return self.draft_tokens[:num_reqs]

    def _batch(
        self, num_reqs: int, num_tokens: int, is_prefill: bool, *, first_pass: bool
    ):
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        return determine_draft_batch_execution_and_padding(
            cfg=self.runner.shape_config,
            status=None if dp_size == 1 else self.runner.dp_status,
            dp_rank=self.dp_rank,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            is_prefill=is_prefill,
            draft_has_moe=False,
            first_pass=first_pass,
        )

    def _run(
        self,
        input_ids: torch.Tensor,
        positions: np.ndarray,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor | None,
        layout: InputLayout,
        seq_lens: np.ndarray,
        block_tables: tuple[torch.Tensor, ...],
        is_prefill: bool,
        num_tokens: int,
        *,
        first_pass: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_desc, num_tokens_across_dp = self._batch(
            layout.num_reqs, num_tokens, is_prefill, first_pass=first_pass
        )
        positions = positions.astype(np.int64)
        attn_metadata = self.runner.build_attn_metadata(
            self.attn_groups,
            layout.num_reqs,
            layout.num_reqs_padded,
            torch.from_numpy(
                np.arange(layout.num_reqs + 1, dtype=np.int32) * layout.query_len
            ),
            torch.from_numpy(seq_lens.astype(np.int32)),
            torch.from_numpy(positions.reshape(-1)),
            block_tables,
            is_prefill,
        )
        staged = self.input_stager.stage(
            input_ids=input_ids,
            positions=torch.from_numpy(positions),
            hidden_states=hidden_states,
            token_indices=token_indices,
            layout=layout,
        )
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=batch_desc.num_tokens_padded,
        ):
            return self.model_executable(
                input_ids=staged.input_ids,
                positions=staged.positions,
                hidden_states=staged.hidden_states,
                token_indices=staged.token_indices,
            )
