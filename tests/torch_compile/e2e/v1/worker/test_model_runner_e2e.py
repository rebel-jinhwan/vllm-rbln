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

"""End-to-end tests for RBLNModelRunner using facebook/opt-125m on CPU.

These tests exercise the real model runner code paths (init, load_model,
get_kv_cache_spec, initialize_kv_cache, _update_states, _prepare_inputs,
execute_model, sample_tokens) to maximise coverage of rbln_model_runner.py.
"""

import os
import tempfile

# Must be set before any vllm import so that the env-dependent branches
# inside the model runner take effect.
os.environ["VLLM_RBLN_COMPILE_MODEL"] = "0"
os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "0"
os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
os.environ["VLLM_USE_V1"] = "1"

# load_general_plugins() must run BEFORE importing rbln_model_runner so that
# the monkey-patched set_forward_context (with num_padded_tokens support) is
# in place when rbln_model_runner does its top-level import.
from vllm.plugins import load_general_plugins  # noqa: E402

load_general_plugins()

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (  # noqa: E402
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.sched.output import (  # noqa: E402
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner  # noqa: E402

MODEL_NAME = "facebook/opt-125m"
BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vllm_config_for_module():
    """Module-scoped VllmConfig so we only build it once."""
    max_model_len = 2048
    max_num_batched_tokens = 512
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        is_encoder_decoder=False,
    )
    model_config = ModelConfig(
        model=MODEL_NAME,
        dtype=torch.float32,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig(
        data_parallel_size=1,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


@pytest.fixture(scope="module")
def dist_init_module(vllm_config_for_module):
    """Module-scoped distributed init so we only pay the cost once."""
    temp_file = tempfile.mkstemp()[1]
    with set_current_vllm_config(vllm_config_for_module, check_compile=False):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield
        try:
            cleanup_dist_env_and_memory(shutdown_ray=True)
        except RuntimeError:
            # No accelerator device available in CPU-only test env
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


@pytest.fixture(scope="module")
def runner_with_model(vllm_config_for_module, dist_init_module):
    """Module-scoped runner with model loaded (expensive, done once)."""
    with set_current_vllm_config(vllm_config_for_module, check_compile=False):
        runner = RBLNModelRunner(
            vllm_config=vllm_config_for_module,
            device=torch.device("cpu"),
        )
        runner.load_model()
        yield runner


@pytest.fixture(scope="module")
def kv_cache_spec(runner_with_model, vllm_config_for_module):
    """Module-scoped KV cache spec from the loaded model."""
    with set_current_vllm_config(vllm_config_for_module, check_compile=False):
        return runner_with_model.get_kv_cache_spec()


@pytest.fixture(scope="module")
def runner_with_kv_cache(
    runner_with_model, kv_cache_spec, vllm_config_for_module
):
    """Module-scoped runner with KV cache fully initialised."""
    num_blocks = 32
    with set_current_vllm_config(vllm_config_for_module, check_compile=False):
        # Build a KVCacheConfig from the spec.
        kv_cache_groups = []
        for layer_name, spec in kv_cache_spec.items():
            kv_cache_groups.append(
                KVCacheGroupSpec(
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                )
            )

        # Merge layers that share the same spec into one group.
        merged_groups: dict[str, KVCacheGroupSpec] = {}
        for group in kv_cache_groups:
            key = repr(group.kv_cache_spec)
            if key in merged_groups:
                merged_groups[key].layer_names.extend(group.layer_names)
            else:
                merged_groups[key] = KVCacheGroupSpec(
                    layer_names=list(group.layer_names),
                    kv_cache_spec=group.kv_cache_spec,
                )

        merged_group_list = list(merged_groups.values())

        # Build KVCacheTensor list - one per group
        kv_cache_tensors = []
        for group in merged_group_list:
            spec = group.kv_cache_spec
            # page_size_bytes = spec.page_size_bytes
            size = num_blocks * spec.page_size_bytes
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    shared_by=group.layer_names,
                )
            )

        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=merged_group_list,
        )

        runner_with_model.initialize_kv_cache(kv_cache_config)
        yield runner_with_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduler_output_for_new_request(
    req_id: str,
    prompt_token_ids: list[int],
    block_ids: tuple[list[int], ...],
    num_computed_tokens: int = 0,
) -> SchedulerOutput:
    """Build a minimal SchedulerOutput containing one new request."""
    num_tokens = len(prompt_token_ids)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    new_req = NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=[],
        sampling_params=sampling_params,
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
    )
    return SchedulerOutput(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: num_tokens},
        total_num_scheduled_tokens=num_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _cleanup_request(runner, req_id: str):
    """Remove a request from the runner state so the next test starts clean."""
    runner.requests.pop(req_id, None)
    runner.num_prompt_logprobs.pop(req_id, None)
    runner.input_batch.remove_request(req_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelRunnerInit:
    """Test that RBLNModelRunner can be constructed with real config."""

    def test_runner_construction(
        self, vllm_config_for_module, dist_init_module
    ):
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner = RBLNModelRunner(
                vllm_config=vllm_config_for_module,
                device=torch.device("cpu"),
            )
        assert runner.device == torch.device("cpu")
        assert runner.max_model_len == 2048
        assert runner.max_num_reqs == 4
        assert runner.is_pooling_model is False
        assert runner.use_alibi is False


class TestLoadModel:
    """Test model loading."""

    def test_model_loaded(self, runner_with_model):
        assert hasattr(runner_with_model, "model")
        assert runner_with_model.model is not None

    def test_model_executable_set(self, runner_with_model):
        assert hasattr(runner_with_model, "model_executable")
        assert runner_with_model.model_executable is not None

    def test_logits_processor_set(self, runner_with_model):
        assert runner_with_model.logits_processor is not None

    def test_compute_logits_model_set(self, runner_with_model):
        assert runner_with_model.compute_logits_model is not None


class TestGetKVCacheSpec:
    """Test get_kv_cache_spec after model loading."""

    def test_returns_dict(self, kv_cache_spec):
        assert isinstance(kv_cache_spec, dict)
        assert len(kv_cache_spec) > 0

    def test_layer_names_contain_attn(self, kv_cache_spec):
        for layer_name in kv_cache_spec:
            assert "attn" in layer_name.lower() or "attention" in layer_name.lower(), (
                f"Unexpected layer name without 'attn': {layer_name}"
            )

    def test_spec_has_block_size(self, kv_cache_spec):
        for spec in kv_cache_spec.values():
            assert hasattr(spec, "block_size")
            assert spec.block_size == BLOCK_SIZE


class TestGetSupportedTasks:
    """Test get_supported_tasks for a generation model."""

    def test_returns_generate(self, runner_with_model, vllm_config_for_module):
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            tasks = runner_with_model.get_supported_tasks()
        assert "generate" in tasks

    def test_generation_tasks_list(
        self, runner_with_model, vllm_config_for_module
    ):
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            gen_tasks = runner_with_model.get_supported_generation_tasks()
        assert "generate" in gen_tasks


class TestInitializeKVCache:
    """Test KV cache initialisation."""

    def test_kv_caches_populated(self, runner_with_kv_cache):
        assert len(runner_with_kv_cache.kv_caches) > 0

    def test_kv_cache_config_stored(self, runner_with_kv_cache):
        assert hasattr(runner_with_kv_cache, "kv_cache_config")
        assert runner_with_kv_cache.kv_cache_config is not None

    def test_attn_groups_populated(self, runner_with_kv_cache):
        assert len(runner_with_kv_cache.attn_groups) > 0


class TestUpdateStates:
    """Test _update_states with a new request."""

    def test_update_states_adds_request(
        self, runner_with_kv_cache, vllm_config_for_module
    ):
        runner = runner_with_kv_cache
        req_id = "test-update-states-001"
        prompt = [1, 2, 3, 4, 5]
        num_blocks_needed = (len(prompt) + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_ids = ([i for i in range(num_blocks_needed)],)

        scheduler_output = _make_scheduler_output_for_new_request(
            req_id=req_id,
            prompt_token_ids=prompt,
            block_ids=block_ids,
        )

        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner._update_states(scheduler_output)

        assert req_id in runner.requests
        assert req_id in runner.input_batch.req_id_to_index

        # Cleanup
        _cleanup_request(runner, req_id)

    def test_update_states_finished_removes_request(
        self, runner_with_kv_cache, vllm_config_for_module
    ):
        runner = runner_with_kv_cache
        req_id = "test-update-states-002"
        prompt = [1, 2, 3]
        block_ids = ([0],)

        # First add it
        so = _make_scheduler_output_for_new_request(
            req_id=req_id,
            prompt_token_ids=prompt,
            block_ids=block_ids,
        )
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner._update_states(so)

        assert req_id in runner.requests

        # Now finish it
        finish_so = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids={req_id},
            free_encoder_mm_hashes=[],
        )
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner._update_states(finish_so)

        assert req_id not in runner.requests


class TestExecuteModel:
    """Test execute_model with a real forward pass on CPU."""

    def test_execute_model_returns_none_for_sample(
        self, runner_with_kv_cache, vllm_config_for_module
    ):
        """execute_model should return None, storing state for sample_tokens."""
        runner = runner_with_kv_cache
        req_id = "test-exec-001"
        prompt = [2, 50, 100, 200, 300]  # 5 tokens
        num_blocks = (len(prompt) + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_ids = ([i for i in range(num_blocks)],)

        scheduler_output = _make_scheduler_output_for_new_request(
            req_id=req_id,
            prompt_token_ids=prompt,
            block_ids=block_ids,
        )

        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            result = runner.execute_model(scheduler_output)

        # execute_model returns None, state stored for sample_tokens
        assert result is None
        assert runner.execute_model_state is not None

        # Verify state contents
        state = runner.execute_model_state
        assert state.logits is not None
        assert state.logits.dim() == 2  # [num_sampled_tokens, vocab_size]
        assert state.scheduler_output is scheduler_output

        # Cleanup: call sample_tokens to clear state, then remove request
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            output = runner.sample_tokens(grammar_output=None)

        assert output is not None
        assert hasattr(output, "sampled_token_ids")
        assert hasattr(output, "req_ids")
        assert req_id in output.req_ids

        # Verify state is cleared
        assert runner.execute_model_state is None

        # Cleanup request
        finish_so = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids={req_id},
            free_encoder_mm_hashes=[],
        )
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner._update_states(finish_so)

    def test_execute_model_empty_returns_empty(
        self, runner_with_kv_cache, vllm_config_for_module
    ):
        """An empty SchedulerOutput should return EMPTY_MODEL_RUNNER_OUTPUT."""
        runner = runner_with_kv_cache
        empty_so = SchedulerOutput.make_empty()

        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            result = runner.execute_model(empty_so)

        # Should return the empty sentinel
        assert result is not None
        assert result.sampled_token_ids == []

    def test_logits_shape(
        self, runner_with_kv_cache, vllm_config_for_module
    ):
        """Verify logits have the correct vocab dimension."""
        runner = runner_with_kv_cache
        req_id = "test-logits-shape-001"
        prompt = [2, 10, 20]
        block_ids = ([0],)

        scheduler_output = _make_scheduler_output_for_new_request(
            req_id=req_id,
            prompt_token_ids=prompt,
            block_ids=block_ids,
        )

        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner.execute_model(scheduler_output)

        state = runner.execute_model_state
        vocab_size = runner.model_config.get_vocab_size()
        assert state.logits.shape[-1] == vocab_size

        # Cleanup
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner.sample_tokens(grammar_output=None)

        finish_so = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids={req_id},
            free_encoder_mm_hashes=[],
        )
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner._update_states(finish_so)


class TestSampleTokens:
    """Test the sample_tokens path in isolation."""

    def test_sample_returns_valid_token_ids(
        self, runner_with_kv_cache, vllm_config_for_module
    ):
        runner = runner_with_kv_cache
        req_id = "test-sample-001"
        prompt = [2, 5, 15, 25, 35, 45]
        num_blocks = (len(prompt) + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_ids = ([i for i in range(num_blocks)],)

        scheduler_output = _make_scheduler_output_for_new_request(
            req_id=req_id,
            prompt_token_ids=prompt,
            block_ids=block_ids,
        )

        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner.execute_model(scheduler_output)
            output = runner.sample_tokens(grammar_output=None)

        assert output is not None
        # sampled_token_ids is a list of lists
        assert isinstance(output.sampled_token_ids, list)
        assert len(output.sampled_token_ids) > 0
        # Each element should be a list with at least one token
        for token_list in output.sampled_token_ids:
            assert isinstance(token_list, list)
            if len(token_list) > 0:
                assert all(isinstance(t, int) for t in token_list)

        # Cleanup
        finish_so = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids={req_id},
            free_encoder_mm_hashes=[],
        )
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            runner._update_states(finish_so)

    def test_sample_without_execute_returns_none(
        self, runner_with_kv_cache, vllm_config_for_module
    ):
        """Calling sample_tokens without prior execute_model should
        return None (no state)."""
        runner = runner_with_kv_cache
        assert runner.execute_model_state is None

        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            result = runner.sample_tokens(grammar_output=None)
        # When there is no execute_model_state and no kv_connector_output,
        # sample_tokens returns None.
        assert result is None


class TestHelperMethods:
    """Test various helper and utility methods on the runner."""

    def test_get_model(self, runner_with_model):
        model = runner_with_model.get_model()
        assert model is not None
        assert hasattr(model, "forward")

    def test_is_prefills(self, runner_with_kv_cache, vllm_config_for_module):
        runner = runner_with_kv_cache
        with set_current_vllm_config(
            vllm_config_for_module, check_compile=False
        ):
            prefills = runner.is_prefills()
        assert isinstance(prefills, np.ndarray)

    def test_use_wrapped_compute_logits(self, runner_with_kv_cache):
        # No lora and no eagle spec decode, should use wrapped compute logits
        result = runner_with_kv_cache.use_wrapped_compute_logits()
        assert result is True

    def test_get_cumsum_and_arange(self, runner_with_kv_cache):
        runner = runner_with_kv_cache
        arr = np.array([2, 5, 3], dtype=np.int32)
        cu, arange = runner._get_cumsum_and_arange(arr)
        np.testing.assert_array_equal(cu, [2, 7, 10])
        assert len(arange) == 10
        expected_arange = np.array([0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        np.testing.assert_array_equal(arange, expected_arange)


class TestNamedTuples:
    """Verify named tuple structures used by the runner."""

    def test_execute_model_state_fields(self):
        from vllm_rbln.v1.worker.rbln_model_runner import ExecuteModelState

        fields = ExecuteModelState._fields
        assert "scheduler_output" in fields
        assert "logits" in fields
        assert "hidden_states" in fields
        assert "kv_connector_output" in fields

    def test_dummy_run_state_fields(self):
        from vllm_rbln.v1.worker.rbln_model_runner import DummyRunState

        fields = DummyRunState._fields
        assert "attn_metadata" in fields
        assert "num_input_tokens" in fields
        assert "input_ids" in fields
        assert "positions" in fields
