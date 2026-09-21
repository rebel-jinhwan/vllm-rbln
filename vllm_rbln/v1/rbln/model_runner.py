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

"""RBLN model runner built on upstream's V2 ``GPUModelRunner``.

Upstream owns request state, block tables, input preparation and sampling
(``vllm/v1/worker/gpu/``, with its triton kernels swapped for torch by
``vllm_rbln.patches.model_runner_v2``). This class overrides upstream's
three hardware seams: `dispatch_batch` picks the compiled shape, with the
prefill/decode phase taken from the scheduler output; `build_attn_metadata`
calls the RBLN builder and lays the step out as padded rows; `run_model`
stages those rows into the compiled graph, which returns the logits with the
hidden states.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.attn_utils import (
    get_shared_kv_cache_layers,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_rbln import envs
from vllm_rbln.compilation import (
    build_process_group_dict,
    compile,
    set_compile_stage,
)
from vllm_rbln.config import RBLNConfig
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.platform import USE_DEVICE_TENSOR
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadataBuilder,
)
from vllm_rbln.v1.attention.kv_cache_bindings import attach_kv_cache_bindings
from vllm_rbln.v1.core.rbln_kv_cache_manager import KVCacheCopyOp
from vllm_rbln.v1.core.rbln_scheduler import RBLNSchedulerOutput
from vllm_rbln.v1.core.utils import decode_batch_size, step_is_prefill
from vllm_rbln.v1.rbln.eagle import RBLNEagleSpeculator
from vllm_rbln.v1.rbln.kernels import RBLNKernels
from vllm_rbln.v1.worker import mega_cache
from vllm_rbln.v1.worker.bucketing import get_bucketing_manager
from vllm_rbln.v1.worker.dp_utils import (
    DPStatus,
    ShapeConfig,
    coordinate_batch_across_dp,
    determine_batch_execution_and_padding,
)
from vllm_rbln.v1.worker.input_stager import InputLayout, InputStager
from vllm_rbln.v1.worker.utils import (
    get_kv_cache_names,
    make_weights_contiguous,
)
from vllm_rbln.v1.worker.utils import num_attn_module as rbln_num_attn_module

logger = init_logger(__name__)


@dataclass
class StepShape:
    """The rows a step staged for the target, kept for the draft that follows:
    the draft runs on the same rows with its own tokens."""

    layout: InputLayout
    input_ids: torch.Tensor
    """``[num_reqs, query_len]`` on the device, before the stager's padding."""
    positions: np.ndarray
    """``[num_reqs, query_len]`` host positions of those rows."""
    front_pad: np.ndarray
    """Per request, how many already-computed tokens open its row: a decode
    window near ``max_model_len`` starts early rather than run past it."""
    block_tables: tuple[torch.Tensor, ...]
    token_rows: torch.Tensor
    """For each token in upstream's flat order, its index in the flattened
    rows; upstream and the drafter read and write flat, the graph reads rows."""


class RBLNModelRunnerV2(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        unsupported = [
            name
            for name, enabled in (
                ("pipeline parallelism", parallel_config.pipeline_parallel_size > 1),
                ("LoRA", vllm_config.lora_config is not None),
                ("multimodal models", model_config.is_multimodal_model),
                ("the host-tensor path", not USE_DEVICE_TENSOR),
                ("dynamic KV cache", envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE),
            )
            if enabled
        ]
        if unsupported:
            raise NotImplementedError(
                "RBLNModelRunnerV2 does not support "
                + ", ".join(unsupported)
                + " yet; unset VLLM_USE_V2_MODEL_RUNNER."
            )

        super().__init__(vllm_config, device)
        self.execute_model_state: ExecuteModelState | None = None

        # Step phase, stamped from the scheduler output before anything reads it.
        self.is_prefill: bool = False
        self.rbln_config: RBLNConfig = vllm_config.additional_config
        self.runtime_holder: list = []
        self.input_stager = InputStager(device)
        self.bucketing_manager = get_bucketing_manager(
            self.rbln_config.decode_batch_bucket_strategy,
            max_batch_size=decode_batch_size(
                self.max_num_reqs, parallel_config.pipeline_parallel_size
            ),
            min_batch_size=self.rbln_config.decode_batch_bucket_min,
            step=self.rbln_config.decode_batch_bucket_step,
            limit=self.rbln_config.decode_batch_bucket_limit,
            manual_buckets=self.rbln_config.decode_batch_bucket_manual_buckets,
        )
        logger.info(
            "Using %s. Decode batch buckets: %s",
            type(self.bucketing_manager).__name__,
            self.bucketing_manager.decode_batch_buckets,
        )
        self.specialized_moe_decode = False
        # Read by the worker's runtime count: with a drafter every decode row
        # is decode_query_len wide (see dispatch_batch), so one decode graph.
        self.uses_fixed_decode_window = True
        self.shape_config: ShapeConfig = ShapeConfig(
            decode_batch_buckets=self.bucketing_manager.decode_batch_buckets,
            find_bucket=self.bucketing_manager.find_decode_batch_bucket,
            max_num_tokens=self.max_num_tokens,
            specialized_moe_decode=False,
        )
        self.drafter = None

        self.offload_context = nullcontext
        if USE_DEVICE_TENSOR and not envs.VLLM_RBLN_DISABLE_OFFLOAD:
            self.offload_context = torch.rbln.offload

        # The worker reads these on its dynamic-KV and connector paths.
        self.kv_cache_bases: list[torch.Tensor] = []
        self.kv_cache_names: list[str] = []
        self.shared_kv_cache_layers: dict[str, str] = {}
        # Logits leave the compiled graph with the hidden states, so they are
        # carried from execute_model() to sample() here rather than recomputed.
        self._step_logits: torch.Tensor | None = None
        # Rows of _step_logits that upstream's logits_indices name, in the
        # staged layout; None on a prefill, whose graph gathered them already.
        self._step_logits_index: torch.Tensor | None = None
        self.step_shape: StepShape | None = None
        # What this step's DP ranks reported; None on a single rank.
        self.dp_status: DPStatus | None = None
        # Set by _dummy_run for the step a rank without work joins; the
        # dispatch tells the group so instead of driving the shape.
        self._dp_idle = False
        # The token dimension the DP group settled on this step, for the
        # forward context; None on a single rank.
        self._num_padded_tokens: int | None = None

    def init_kernels(self) -> RBLNKernels:
        return RBLNKernels()

    def init_speculator(self) -> RBLNEagleSpeculator:
        return RBLNEagleSpeculator(self.vllm_config, self.device, self)

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        with self.offload_context():
            super().load_model(load_dummy_weights, *args, **kwargs)
        make_weights_contiguous(self.model)
        if self.speculator is not None:
            make_weights_contiguous(self.speculator.model)

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            token_indices: torch.Tensor | None = None,
            **kwargs,
        ):
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            if self.is_pooling_model:
                return model_output, None
            # eagle3's aux states leave the graph as one concatenated tensor:
            # a list of outputs that alias each other does not compile.
            aux: tuple[torch.Tensor, ...] = ()
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
                aux = (torch.cat(aux_hidden_states, dim=-1),)
            else:
                hidden_states = model_output
            sample_hidden_states = hidden_states
            if token_indices is not None:
                sample_hidden_states = hidden_states[:, token_indices]
            logits = self.model.compute_logits(sample_hidden_states)
            return logits.view(-1, logits.size(-1)), hidden_states, *aux

        if self.model_config.enforce_eager or not self.rbln_config.compile_model:
            self.model_executable = model_wrapper
            return
        self.model_executable = compile(
            model_wrapper,
            dynamic=False,
            fullgraph=True,
            num_devices=self.rbln_config.num_devices_per_local_rank,
            model_trace_method="export",
            process_group_dict=build_process_group_dict(),
            guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
            runtime_holder=self.runtime_holder,
            mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
            use_static_output=True,
            use_direct_dispatch=True,
        )

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        if self.rbln_config.enable_sub_block_cache and (
            len(kv_cache_config.kv_cache_groups) > 1
        ):
            raise NotImplementedError(
                "Sub-block prefix caching does not support multi-group KV caches "
                "yet. Set VLLM_RBLN_SUB_BLOCK_CACHE=false to disable."
            )
        super().initialize_kv_cache(kv_cache_config)
        self._kernel_block_sizes = self.kernel_block_sizes
        self.shared_kv_cache_layers = get_shared_kv_cache_layers(self.vllm_config)
        layer_names = {
            name: None
            for group in self.kv_cache_config.kv_cache_groups
            for name in group.layer_names
        }
        self.kv_cache_names = get_kv_cache_names(
            layer_names,
            rbln_num_attn_module(self.model_config, self.cache_config.cache_dtype),
        )
        self.cache_config.num_gpu_blocks = self.kv_cache_config.num_blocks
        self.cache_config.num_cpu_blocks = 0

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        super().update_requests(scheduler_output)
        assert isinstance(scheduler_output, RBLNSchedulerOutput)
        if scheduler_output.kv_cache_copy_ops:
            self._copy_kv_cache_sub_blocks(scheduler_output.kv_cache_copy_ops)

    def _copy_kv_cache_sub_blocks(self, copy_ops: list[KVCacheCopyOp]) -> None:
        """Populate a partially cached block before the forward reads it."""
        # Upstream builds every view from the backend's get_kv_cache_shape:
        # blocks are axis 1 of an attention cache and axis 0 of an MLA cache.
        axis = 0 if self.model_config.use_mla else 1
        dsts: list[torch.Tensor] = []
        srcs: list[torch.Tensor] = []
        for op in copy_ops:
            for kv_cache in self.kv_caches:
                dst = kv_cache.select(axis, op.dst_block_id)
                src = kv_cache.select(axis, op.src_block_id)
                dsts.append(dst[..., : op.num_tokens, :])
                srcs.append(src[..., : op.num_tokens, :])
        torch._foreach_copy_(dsts, srcs)

    def dispatch_batch(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        max_query_len: int,
        *,
        dummy_run: bool,
        need_eager: bool,
        num_active_loras: int,
        **kwargs: Any,
    ) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
        """The compiled shape this step runs, agreed with the other DP ranks
        when there are any (see v1/worker/dp_utils.py). With a drafter every
        decode row is decode_query_len wide, whatever each request brought,
        so one decode graph serves every step."""
        if self.speculative_config is not None and not self.is_prefill:
            num_tokens = num_reqs * self.decode_query_len
        dp_size = self.parallel_config.data_parallel_size
        num_tokens_across_dp: torch.Tensor | None = None
        if dp_size == 1:
            batch_desc, _route = determine_batch_execution_and_padding(
                cfg=self.shape_config,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                is_prefill=self.is_prefill,
                status=None,
            )
        else:
            batch_desc, _route, dp_status = coordinate_batch_across_dp(
                cfg=self.shape_config,
                dp_size=dp_size,
                dp_rank=self.parallel_config.data_parallel_rank,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                is_prefill=self.is_prefill,
                is_idle=self._dp_idle,
            )
            self.dp_status = dp_status
            num_tokens_across_dp = dp_status.num_tokens_across_dp
        if batch_desc is None:
            # Every DP rank is idle: they all read the same status and stop here.
            return BatchExecutionDescriptor(CUDAGraphMode.NONE, 0, 0), None
        self._num_padded_tokens = batch_desc.num_tokens_padded
        query_len_padded = (
            self.max_num_tokens if self.is_prefill else batch_desc.query_len
        )
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=batch_desc.num_reqs_padded * query_len_padded,
                num_reqs=batch_desc.num_reqs_padded,
                uniform_token_count=batch_desc.query_len,
            ),
            num_tokens_across_dp,
        )

    def build_attn_metadata(  # type: ignore[override]
        self,
        input_batch: InputBatch,
        batch_desc: BatchExecutionDescriptor,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[Any]],
        *,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        """Lay the step out as padded rows and build the RBLN metadata on them.
        The rows are kept in step_shape for run_model and the drafter."""
        assert batch_desc.num_reqs is not None
        assert batch_desc.uniform_token_count is not None
        num_reqs = input_batch.num_reqs
        query_len = batch_desc.uniform_token_count
        query_len_padded = batch_desc.num_tokens // batch_desc.num_reqs
        num_computed = input_batch.num_computed_tokens_np[:num_reqs]
        num_scheduled = input_batch.num_scheduled_tokens[:num_reqs]
        cols = np.arange(query_len)
        front_pad = np.zeros(num_reqs, dtype=np.int32)
        fixed_window = self.speculative_config is not None and not self.is_prefill
        if not fixed_window:
            input_ids = input_batch.input_ids[: input_batch.num_tokens].view(
                num_reqs, query_len
            )
        else:
            front_pad = np.maximum(
                0, num_computed + query_len - self.max_model_len
            ).astype(np.int32)
            # Row column -> this request's scheduled token, the last one
            # repeated past the end and the already-computed ones before it.
            local = cols[None, :] - front_pad[:, None]
            src = input_batch.query_start_loc_np[:num_reqs, None] + np.clip(
                local, 0, num_scheduled[:, None] - 1
            )
            input_ids = input_batch.input_ids[torch.from_numpy(src).to(self.device)]
        positions_np = (num_computed - front_pad)[:, None] + cols[None, :]
        if front_pad.any():
            before_window = self.req_states.all_token_ids.gpu[
                input_batch.idx_mapping.long()[:, None],
                torch.from_numpy(positions_np).to(self.device),
            ]
            input_ids = torch.where(
                torch.from_numpy(local >= 0).to(self.device), input_ids, before_window
            )
        positions_np = positions_np.astype(np.int64)
        layout = InputLayout(
            num_reqs=num_reqs,
            num_reqs_padded=batch_desc.num_reqs,
            query_len=query_len,
            query_len_padded=query_len_padded,
        )
        # Flat token (req i, offset j) sits in row i at column front_pad_i + j.
        req_of_token = np.repeat(np.arange(num_reqs), num_scheduled)
        offset = np.arange(input_batch.num_tokens) - np.repeat(
            input_batch.query_start_loc_np[:num_reqs], num_scheduled
        )
        token_rows = torch.from_numpy(
            req_of_token * query_len_padded + front_pad[req_of_token] + offset
        ).to(self.device)
        self.step_shape = StepShape(
            layout, input_ids, positions_np, front_pad, block_tables, token_rows
        )
        # A prefill graph gathers its logits at token_indices already.
        self._step_logits_index = (
            None if self.is_prefill else token_rows[input_batch.logits_indices]
        )
        return self.rbln_attn_metadata(
            attn_groups,
            num_reqs,
            batch_desc.num_reqs,
            torch.from_numpy(np.arange(num_reqs + 1, dtype=np.int32) * query_len),
            input_batch.seq_lens_cpu_upper_bound[:num_reqs],
            torch.from_numpy(positions_np.reshape(-1)),
            block_tables,
            self.is_prefill,
        )

    def rbln_attn_metadata(
        self,
        attn_groups: list[list[Any]],
        num_reqs: int,
        num_reqs_padded: int,
        query_start_loc_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        positions_cpu: torch.Tensor,
        block_tables: tuple[torch.Tensor, ...],
        is_prefill: bool,
    ) -> dict[str, Any]:
        """Per-layer RBLN attention metadata for the given groups: the target's
        or the draft's, which share the KV cache groups and block tables."""
        attn_metadata: dict[str, Any] = {}
        for gid, groups in enumerate(attn_groups):
            common = CommonAttentionMetadata(
                query_start_loc=query_start_loc_cpu,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=int(query_start_loc_cpu[num_reqs]),
                max_query_len=int(query_start_loc_cpu[1]),
                max_seq_len=int(seq_lens_cpu.max()),
                block_table_tensor=block_tables[gid][:num_reqs],
                slot_mapping=torch.tensor(0),  # unused by the RBLN kernels
                causal=True,
            )
            for attn_group in groups:
                builder = attn_group.get_metadata_builder(0)
                assert isinstance(builder, RBLNFlashAttentionMetadataBuilder)
                metadata = builder.build(
                    common_attn_metadata=common,
                    positions=positions_cpu,
                    is_prefill=is_prefill,
                    batch_pad=num_reqs_padded,
                )
                attach_kv_cache_bindings(metadata, self.kv_caches, None, None)
                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = metadata
        return attn_metadata

    @torch.inference_mode()
    def execute_model(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            self.is_prefill = step_is_prefill(scheduler_output)
        output = super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            is_profile=is_profile,
        )
        if (
            self.is_pooling_model
            and not dummy_run
            and self.execute_model_state is not None
        ):
            # Upstream's worker calls pool() itself; RBLN's worker takes the
            # step's output from execute_model.
            return self.pool()
        return output

    def run_model(  # type: ignore[override]
        self,
        batch_desc: BatchExecutionDescriptor,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        attn_metadata: dict[str, Any] | None,
        slot_mappings_by_layer: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        skip_compiled: bool,
        scheduler_output: SchedulerOutput,
    ) -> Any:
        """Stage the rows build_attn_metadata laid out into the compiled graph
        for this shape. The graph returns the logits with the hidden states;
        sample() reads them from _step_logits."""
        step = self.step_shape
        assert step is not None
        layout = step.layout
        staged = self.input_stager.stage(
            input_ids=step.input_ids,
            positions=torch.from_numpy(step.positions),
            token_indices=(
                input_batch.logits_indices.to(torch.int32)
                if self.is_prefill and not self.is_pooling_model
                else None
            ),
            layout=layout,
        )
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=batch_desc.num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=self._num_padded_tokens,
        ):
            self.kv_connector.pre_forward(scheduler_output)
            outputs = self.model_executable(**staged.as_kwargs())
        if self.is_pooling_model:
            hidden_states, _ = outputs
            aux_hidden_states = []
        else:
            self._step_logits, hidden_states, *aux_hidden_states = outputs

        def flat(rows: torch.Tensor) -> torch.Tensor:
            # Upstream reads num_tokens_after_padding rows in flat token order.
            out = rows.new_zeros(batch_desc.num_tokens, rows.shape[-1])
            out[: step.token_rows.shape[0]] = rows.reshape(-1, rows.shape[-1])[
                step.token_rows
            ]
            return out

        if self.use_aux_hidden_state_outputs:
            (aux_cat,) = aux_hidden_states
            num_aux = aux_cat.shape[-1] // hidden_states.shape[-1]
            return flat(hidden_states), [flat(h) for h in aux_cat.chunk(num_aux, -1)]
        return flat(hidden_states)

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        assert self._step_logits is not None
        if self._step_logits_index is None:
            logits = self._step_logits[: input_batch.logits_indices.shape[0]]
        else:
            logits = self._step_logits[self._step_logits_index]
        self._step_logits = None
        if grammar_output is not None:
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )
        assert self.sampler is not None
        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            sampler_output = self.sampler(logits, input_batch)
        else:
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits, input_batch, self.speculator.draft_logits
            )
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    @torch.inference_mode()
    def _dummy_run(  # type: ignore[override]
        self,
        num_reqs: int,
        num_tokens_per_req: int,
        is_prefill: bool,
        *,
        warmup: bool = True,
    ) -> None:
        """Compile the graph for this shape (warmup=True), or, as the DP-idle
        step of a rank with no work, join the group's shape agreement without
        driving it and run whatever the busy ranks run (warmup=False)."""
        self.is_prefill = is_prefill
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.total_num_scheduled_tokens = num_reqs * num_tokens_per_req
        scheduler_output.num_scheduled_tokens = {
            f"_dummy_req_{i}": num_tokens_per_req for i in range(num_reqs)
        }
        self.kv_connector.set_disabled(True)
        self._dp_idle = not warmup
        self.execute_model(scheduler_output, dummy_run=True)
        self._dp_idle = False
        self.kv_connector.set_disabled(False)
        state = self.execute_model_state
        if self.speculator is not None and state is not None:
            assert self.sampler is not None
            self.speculator.propose(
                input_batch=state.input_batch,
                attn_metadata=state.attn_metadata,
                slot_mappings=state.slot_mappings_by_layer,
                last_hidden_states=state.hidden_states,
                aux_hidden_states=None,
                num_sampled=torch.ones(num_reqs, dtype=torch.int32, device=self.device),
                num_rejected=torch.zeros(
                    num_reqs, dtype=torch.int32, device=self.device
                ),
                last_sampled=self.req_states.last_sampled_tokens,
                next_prefill_tokens=self.req_states.next_prefill_tokens,
                temperature=self.sampler.sampling_states.temperature.gpu,
                seeds=self.sampler.sampling_states.seeds.gpu,
                dummy_run=True,
            )
        self.execute_model_state = None
        self._step_logits = None

    def warmup_model(self) -> None:
        logger.info("Compile and warming up model.")
        sig = mega_cache.config_signature(self.vllm_config)
        mega_cache.load(self.model_config.model, sig)
        with set_compile_stage("warmup"), self.offload_context():
            self._dummy_run(1, self.max_num_tokens, True)
            for num_reqs in self.bucketing_manager.decode_batch_buckets:
                self._dummy_run(num_reqs, self.decode_query_len, False)
        mega_cache.save(self.model_config.model, sig)
