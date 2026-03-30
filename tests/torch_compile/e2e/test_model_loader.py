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

"""End-to-end tests for model loading via vllm.LLM and RBLNModelRunner.

Tests model loading, KV cache spec retrieval, supported task detection,
model layer count, and different max_model_len configurations.
"""

import os
import tempfile
import time

os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_RBLN_COMPILE_MODEL"] = "0"
os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "0"

# load_general_plugins() must run BEFORE importing rbln_model_runner so that
# the monkey-patched set_forward_context (with num_padded_tokens support) is
# in place when rbln_model_runner does its top-level import.
from vllm.plugins import load_general_plugins  # noqa: E402

load_general_plugins()

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
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

from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner  # noqa: E402

MODEL_NAME = "facebook/opt-125m"
# OPT-125m has 12 decoder layers
EXPECTED_NUM_LAYERS = 12


# ---------------------------------------------------------------------------
# Runner-level fixtures (for direct RBLNModelRunner tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vllm_config():
    """Module-scoped VllmConfig for runner-level tests."""
    scheduler_config = SchedulerConfig(
        max_num_seqs=2,
        max_num_batched_tokens=128,
        max_model_len=128,
        is_encoder_decoder=False,
    )
    model_config = ModelConfig(
        model=MODEL_NAME,
        dtype=torch.float32,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.5,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig(data_parallel_size=1)
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )


@pytest.fixture(scope="module")
def dist_init(vllm_config):
    """Module-scoped distributed init."""
    temp_file = tempfile.mkstemp()[1]
    with set_current_vllm_config(vllm_config, check_compile=False):
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
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


@pytest.fixture(scope="module")
def runner_with_model(vllm_config, dist_init):
    """Module-scoped runner with loaded model."""
    with set_current_vllm_config(vllm_config, check_compile=False):
        runner = RBLNModelRunner(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )
        runner.load_model()
        yield runner


# ---------------------------------------------------------------------------
# Tests using RBLNModelRunner directly
# ---------------------------------------------------------------------------


class TestModelLoading:
    """Test that facebook/opt-125m loads successfully via RBLNModelRunner."""

    def test_loading_succeeds(self, runner_with_model):
        """RBLNModelRunner should load facebook/opt-125m successfully."""
        assert runner_with_model is not None
        assert runner_with_model.model is not None

    def test_model_has_forward(self, runner_with_model):
        """Loaded model should have a forward method."""
        model = runner_with_model.get_model()
        assert hasattr(model, "forward")


class TestKVCacheSpec:
    """Test get_kv_cache_spec returns valid specs."""

    def test_kv_cache_spec_returns_dict(self, runner_with_model, vllm_config):
        """get_kv_cache_spec should return a non-empty dict."""
        with set_current_vllm_config(vllm_config, check_compile=False):
            spec = runner_with_model.get_kv_cache_spec()
        assert isinstance(spec, dict)
        assert len(spec) > 0

    def test_kv_cache_spec_layer_names(self, runner_with_model, vllm_config):
        """Each key should reference an attention layer."""
        with set_current_vllm_config(vllm_config, check_compile=False):
            spec = runner_with_model.get_kv_cache_spec()
        for layer_name in spec:
            assert "attn" in layer_name.lower() or "attention" in layer_name.lower(), (
                f"Unexpected layer name: {layer_name}"
            )

    def test_kv_cache_spec_has_block_size(self, runner_with_model, vllm_config):
        """Each spec should have a valid block_size attribute."""
        with set_current_vllm_config(vllm_config, check_compile=False):
            spec = runner_with_model.get_kv_cache_spec()
        for layer_spec in spec.values():
            assert hasattr(layer_spec, "block_size")
            assert layer_spec.block_size > 0


class TestSupportedTasks:
    """Test get_supported_tasks returns generate task."""

    def test_generate_task_supported(self, runner_with_model, vllm_config):
        """OPT-125m should support the 'generate' task."""
        with set_current_vllm_config(vllm_config, check_compile=False):
            tasks = runner_with_model.get_supported_tasks()
        assert "generate" in tasks

    def test_generation_tasks_list(self, runner_with_model, vllm_config):
        """get_supported_generation_tasks should include 'generate'."""
        with set_current_vllm_config(vllm_config, check_compile=False):
            gen_tasks = runner_with_model.get_supported_generation_tasks()
        assert "generate" in gen_tasks


class TestModelLayers:
    """Test model has the correct number of layers."""

    def test_correct_layer_count(self, runner_with_model, vllm_config):
        """OPT-125m should have 12 decoder layers in KV cache spec."""
        with set_current_vllm_config(vllm_config, check_compile=False):
            spec = runner_with_model.get_kv_cache_spec()
        assert len(spec) == EXPECTED_NUM_LAYERS, (
            f"Expected {EXPECTED_NUM_LAYERS} layers, got {len(spec)}"
        )


# ---------------------------------------------------------------------------
# Tests using vllm.LLM (full pipeline)
# ---------------------------------------------------------------------------


class TestDifferentMaxModelLen:
    """Test loading with different max_model_len values."""

    @pytest.mark.parametrize("max_model_len", [64, 128, 256])
    def test_max_model_len(self, max_model_len):
        """LLM should load and generate with various max_model_len values."""
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=max_model_len,
            max_num_seqs=2,
            gpu_memory_utilization=0.5,
        )
        params = SamplingParams(max_tokens=5, temperature=0.0)
        outputs = llm.generate(["Hello"], params)
        assert len(outputs[0].outputs[0].text) > 0
        del llm
        time.sleep(2)
