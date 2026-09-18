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
``vllm_rbln.patches.model_runner_v2``). This class adds what RBLN cannot
express there: one compiled graph per padded shape, the prefill/decode phase
taken from the scheduler output, the RBLN attention metadata builder, and
logits computed inside the graph.
"""

import functools
from contextlib import contextmanager, nullcontext
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
    build_slot_mappings_by_layer,
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
    get_or_create_intermediate_tensors,
    make_weights_contiguous,
    recv_intermediate_tensors,
)
from vllm_rbln.v1.worker.utils import num_attn_module as rbln_num_attn_module

logger = init_logger(__name__)


@contextmanager
def _without_cuda_streams():
    """``GPUModelRunner.__init__`` constructs CUDA streams and events for its
    copy paths; this torch build has none. The V2 runner never uses them."""

    class _Placeholder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    saved = torch.cuda.Stream, torch.cuda.Event
    torch.cuda.Stream = torch.cuda.Event = _Placeholder  # type: ignore[misc, assignment]
    try:
        yield
    finally:
        torch.cuda.Stream, torch.cuda.Event = saved  # type: ignore[misc]


class RBLNModelRunnerV2(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        unsupported = [
            name
            for name, enabled in (
                ("speculative decoding", vllm_config.speculative_config is not None),
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

        with _without_cuda_streams():
            super().__init__(vllm_config, device)
        self.output_copy_stream = None

        # Step phase, stamped from the scheduler output before anything reads it.
        self.is_prefill = False
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
        self.shape_config = ShapeConfig(
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
        # PP hand-off buffers per staged shape; see recv_intermediate_tensors.
        self.intermediate_tensors_dict: dict[tuple[int, int], IntermediateTensors] = {}
        # What this step's DP ranks reported; None on a single rank.
        self.dp_status: DPStatus | None = None

    @functools.cached_property
    def main_stream(self) -> None:  # type: ignore[override]
        return None

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        with self.offload_context():
            super().load_model(load_dummy_weights, *args, **kwargs)
        make_weights_contiguous(self.model)

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            token_indices: torch.Tensor | None = None,
            **kwargs,
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            if self.is_pooling_model or not self.is_last_pp_rank:
                return hidden_states, None
            sample_hidden_states = hidden_states
            if token_indices is not None:
                sample_hidden_states = hidden_states[:, token_indices]
            logits = self.model.compute_logits(sample_hidden_states)
            return hidden_states, logits.view(-1, logits.size(-1))

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

    def recv_intermediate_tensors(self) -> IntermediateTensors:
        return recv_intermediate_tensors(
            self.intermediate_tensors_dict,
            self.model,
            self.model_config.dtype,
            self.device,
        )

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

    def _batch_descriptor(
        self, num_reqs: int, num_tokens: int, is_idle: bool = False
    ) -> tuple[BatchExecutionDescriptor | None, torch.Tensor | None, int | None]:
        """This step's padded batch, agreed with the other DP ranks when there
        are any (see v1/worker/dp_utils.py); None when the whole group is idle.
        Also returns what RBLN's forward context needs under DP: the per-rank
        token counts and the token dimension the group settled on."""
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
                is_idle=is_idle,
            )
            self.dp_status = dp_status
            num_tokens_across_dp = dp_status.num_tokens_across_dp
        if batch_desc is None:
            return None, None, None
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
            batch_desc.num_tokens_padded,
        )

    def _build_attn_metadata(
        self,
        input_batch: InputBatch,
        num_reqs_padded: int,
        block_tables: tuple[torch.Tensor, ...],
    ) -> dict[str, Any]:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        num_scheduled = input_batch.num_scheduled_tokens
        query_start_loc_np = input_batch.query_start_loc_np[: num_reqs + 1]
        # The RBLN builder reads each request's first position from a host
        # positions tensor; upstream keeps positions on the device only.
        positions_np = np.repeat(
            input_batch.num_computed_tokens_np[:num_reqs], num_scheduled
        ) + (np.arange(num_tokens) - np.repeat(query_start_loc_np[:-1], num_scheduled))
        positions_cpu = torch.from_numpy(positions_np.astype(np.int64))
        query_start_loc_cpu = torch.from_numpy(query_start_loc_np)
        seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[:num_reqs]

        attn_metadata: dict[str, Any] = {}
        for gid in range(len(self.kv_cache_config.kv_cache_groups)):
            common = CommonAttentionMetadata(
                query_start_loc=query_start_loc_cpu,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=int(num_scheduled.max()),
                max_seq_len=int(seq_lens_cpu.max()),
                block_table_tensor=block_tables[gid][:num_reqs],
                slot_mapping=torch.tensor(0),  # unused by the RBLN kernels
                causal=True,
            )
            for attn_group in self.attn_groups[gid]:
                builder = attn_group.get_metadata_builder(0)
                assert isinstance(builder, RBLNFlashAttentionMetadataBuilder)
                metadata = builder.build(
                    common_attn_metadata=common,
                    positions=positions_cpu,
                    is_prefill=self.is_prefill,
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
        is_idle: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        # Mirrors GPUModelRunner.execute_model up to the model call, which RBLN
        # routes to the compiled per-shape graph with fused logits; upstream
        # has no override point between set_forward_context and self.model().
        if not dummy_run:
            self.is_prefill = step_is_prefill(scheduler_output)
            self.update_pp_decode_requests()
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                return self.kv_connector.no_forward(scheduler_output)

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_tokens = scheduler_output.total_num_scheduled_tokens
        batch_desc, num_tokens_across_dp, num_padded_tokens = self._batch_descriptor(
            num_reqs, num_tokens, is_idle
        )
        if batch_desc is None:
            # Every DP rank is idle: they all read the same status and stop here.
            return None
        assert batch_desc.num_reqs is not None
        assert batch_desc.uniform_token_count is not None

        if not dummy_run:
            input_batch = self.prepare_inputs(scheduler_output, batch_desc)
            block_tables, slot_mappings = self.prepare_attn(input_batch)
        else:
            input_batch = InputBatch.make_dummy(
                num_reqs, num_tokens, self.input_buffers
            )
            block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )
        attn_metadata = self._build_attn_metadata(
            input_batch, batch_desc.num_reqs, block_tables
        )

        query_len = batch_desc.uniform_token_count
        query_len_padded = batch_desc.num_tokens // batch_desc.num_reqs
        if not self.is_first_pp_rank and (dummy_run or is_idle):
            # A previous stage's output arrives already padded; a dummy step has
            # to build its stand-in at the same padded shape.
            intermediate_tensors = get_or_create_intermediate_tensors(
                self.intermediate_tensors_dict,
                self.model,
                batch_desc.num_reqs,
                query_len_padded,
                self.model_config.dtype,
                self.device,
            )
        layout = InputLayout(
            num_reqs=num_reqs,
            num_reqs_padded=batch_desc.num_reqs,
            query_len=query_len,
            query_len_padded=query_len_padded,
        )
        staged = self.input_stager.stage(
            input_ids=input_batch.input_ids[:num_tokens].view(num_reqs, query_len),
            positions=input_batch.positions[:num_tokens].view(num_reqs, query_len),
            intermediate_tensors=intermediate_tensors,
            token_indices=(
                input_batch.logits_indices.to(torch.int32)
                if self.is_prefill
                and self.is_last_pp_rank
                and not self.is_pooling_model
                else None
            ),
            layout=layout,
        )
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=batch_desc.num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
        ):
            model_output, logits = self.model_executable(**staged.as_kwargs())

        if self.is_last_pp_rank:
            hidden_states = model_output.reshape(-1, model_output.shape[-1])
        else:
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
        self._step_logits = logits
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=None,
            finished_req_ids=scheduler_output.finished_req_ids,
        )
        if not self.is_last_pp_rank:
            return model_output
        if self.is_pooling_model and not dummy_run:
            return self.pool()
        return None

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        assert self._step_logits is not None
        logits = self._step_logits[: input_batch.logits_indices.shape[0]]
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
        sampler_output = self.sampler(logits, input_batch)
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
        self.execute_model(scheduler_output, dummy_run=True, is_idle=not warmup)
        self.kv_connector.set_disabled(False)
        self.execute_model_state = None
        self._step_logits = None

    def warmup_model(self) -> None:
        logger.info("Compile and warming up model.")
        sig = mega_cache.config_signature(self.vllm_config)
        mega_cache.load(self.model_config.model, sig)
        with set_compile_stage("warmup"), self.offload_context():
            self._dummy_run(1, self.max_num_tokens, True)
            for num_reqs in self.bucketing_manager.decode_batch_buckets:
                self._dummy_run(num_reqs, 1, False)
        mega_cache.save(self.model_config.model, sig)
