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

"""Feature tests for RBLNModelRunner: mixin compliance, async output,
named tuples, get_supported_tasks integration, and bug-catching scenarios."""

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput, SamplerOutput

from vllm_rbln.v1.worker.rbln_model_runner import (
    AsyncRBLNModelRunnerOutput,
    DummyRunState,
    ExecuteModelState,
    RBLNModelRunner,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_runner_stub(**overrides):
    """Create a lightweight stub mimicking RBLNModelRunner attributes."""
    defaults = dict(
        model=MagicMock(),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_rank=0,
                tensor_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sequence_parallelism=False)
            ),
        ),
        model_config=SimpleNamespace(
            runner_type="generate",
            logprobs_mode="raw_logprobs",
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
        ),
        lora_config=None,
        speculative_config=None,
        input_batch=MagicMock(),
        arange_np=np.arange(10000, dtype=np.int64),
        intermediate_tensors=None,
        bucketing_manager=MagicMock(),
        max_num_batched_tokens=256,
        specialized_moe_decode=False,
        sampler=MagicMock(),
        rejection_sampler=MagicMock(),
        performance_tracker=None,
        sampler_performance_tracker=None,
        e2e_performance_tracker=None,
        uses_mrope=False,
        positions=MagicMock(),
        device=torch.device("cpu"),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _bind(stub, method_name):
    """Bind an RBLNModelRunner unbound method to a stub."""
    method = getattr(RBLNModelRunner, method_name)
    return types.MethodType(method, stub)


# ===========================================================================
# 1. Mixin interface compliance
# ===========================================================================


class TestMixinInterfaceCompliance:
    """Verify that RBLNModelRunner inherits the expected mixin methods."""

    def test_inherits_lora_mixin(self):
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        assert issubclass(RBLNModelRunner, LoRAModelRunnerMixin)

    def test_inherits_kv_connector_mixin(self):
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )

        assert issubclass(RBLNModelRunner, KVConnectorModelRunnerMixin)

    @pytest.mark.parametrize(
        "method_name",
        ["add_lora", "remove_lora", "list_loras", "pin_lora"],
    )
    def test_lora_mixin_methods_exist(self, method_name):
        assert hasattr(RBLNModelRunner, method_name), (
            f"RBLNModelRunner missing LoRA mixin method: {method_name}"
        )
        assert callable(getattr(RBLNModelRunner, method_name))

    @pytest.mark.parametrize(
        "method_name",
        [
            "load_lora_model",
            "set_active_loras",
            "maybe_remove_all_loras",
            "maybe_setup_dummy_loras",
            "maybe_select_dummy_loras",
            "maybe_dummy_run_with_lora",
        ],
    )
    def test_lora_mixin_extended_methods_exist(self, method_name):
        assert hasattr(RBLNModelRunner, method_name)

    @pytest.mark.parametrize(
        "method_name",
        [
            "allocate_uniform_kv_caches",
            "ensure_kv_transfer_shutdown",
            "finalize_kv_connector",
            "kv_connector_no_forward",
            "maybe_get_kv_connector_output",
            "use_uniform_kv_cache",
        ],
    )
    def test_kv_connector_mixin_methods_exist(self, method_name):
        assert hasattr(RBLNModelRunner, method_name), (
            f"RBLNModelRunner missing KV connector mixin method: {method_name}"
        )

    @pytest.mark.parametrize(
        "method_name",
        [
            "get_kv_cache_spec",
            "load_model",
            "execute_model",
            "sample_tokens",
            "get_supported_tasks",
            "get_model",
            "warm_up_model",
            "dummy_run",
            "initialize_attn_backend",
            "initialize_kv_cache",
        ],
    )
    def test_expected_public_methods_exist(self, method_name):
        assert hasattr(RBLNModelRunner, method_name), (
            f"RBLNModelRunner missing public method: {method_name}"
        )


# ===========================================================================
# 2. AsyncRBLNModelRunnerOutput
# ===========================================================================


class TestAsyncRBLNModelRunnerOutputFeature:
    def _make_output(self, num_reqs=3, invalid_indices=None):
        """Helper to create an AsyncRBLNModelRunnerOutput with controllable state."""
        mro = MagicMock(spec=ModelRunnerOutput)
        sampled = torch.tensor([[10], [20], [30]][:num_reqs])
        stream = MagicMock()
        output = AsyncRBLNModelRunnerOutput(
            model_runner_output=mro,
            sampled_token_ids=sampled,
            invalid_req_indices=invalid_indices or [],
            async_output_copy_stream=stream,
        )
        return output, mro

    def test_invalid_req_indices_clears_sampled_tokens(self):
        """Verify that invalid_req_indices correctly clears the corresponding
        sampled token entries when get_output() is called."""
        output, mro = self._make_output(num_reqs=3, invalid_indices=[0, 2])

        # Simulate what would happen after the async copy completes:
        # Manually set the internal state that get_output() reads.
        output._sampled_token_ids_cpu = torch.tensor([[10], [20], [30]])
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()

        # Index 0 and 2 should be cleared (empty lists)
        assert result.sampled_token_ids[0] == []
        assert result.sampled_token_ids[1] == [20]  # index 1 is not invalid, preserved
        assert result.sampled_token_ids[2] == []

    def test_get_output_returns_model_runner_output(self):
        """Verify get_output() returns the underlying ModelRunnerOutput."""
        output, mro = self._make_output(num_reqs=2, invalid_indices=[])

        output._sampled_token_ids_cpu = torch.tensor([[10], [20]])
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()
        assert result is mro
        assert result.sampled_token_ids == [[10], [20]]

    def test_get_output_synchronizes_event(self):
        """Verify get_output() calls synchronize on the copy event."""
        output, mro = self._make_output(num_reqs=1, invalid_indices=[])
        mock_event = MagicMock()
        output._async_copy_ready_event = mock_event
        output._sampled_token_ids_cpu = torch.tensor([[42]])

        output.get_output()
        mock_event.synchronize.assert_called_once()

    def test_get_output_deletes_device_tensor(self):
        """After get_output(), the device tensor reference should be released."""
        output, _ = self._make_output(num_reqs=1, invalid_indices=[])
        output._async_copy_ready_event = MagicMock()
        output._sampled_token_ids_cpu = torch.tensor([[1]])

        output.get_output()
        assert not hasattr(output, "_sampled_token_ids")

    def test_all_invalid_indices(self):
        """When all indices are invalid, all sampled tokens should be cleared."""
        output, mro = self._make_output(num_reqs=3, invalid_indices=[0, 1, 2])
        output._sampled_token_ids_cpu = torch.tensor([[1], [2], [3]])
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()
        assert result.sampled_token_ids == [[], [], []]

    def test_no_invalid_indices(self):
        """When no indices are invalid, all tokens should be preserved."""
        output, mro = self._make_output(num_reqs=2, invalid_indices=[])
        output._sampled_token_ids_cpu = torch.tensor([[5], [6]])
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()
        assert result.sampled_token_ids == [[5], [6]]


# ===========================================================================
# 3. ExecuteModelState / DummyRunState named tuples
# ===========================================================================


class TestExecuteModelStateFeature:
    def test_field_names(self):
        expected_fields = (
            "scheduler_output",
            "logits",
            "spec_decode_metadata",
            "spec_decode_common_attn_metadata",
            "hidden_states",
            "sample_hidden_states",
            "aux_hidden_states",
            "kv_connector_output",
            "slot_mappings",
        )
        assert ExecuteModelState._fields == expected_fields

    def test_field_count(self):
        assert len(ExecuteModelState._fields) == 9

    def test_is_named_tuple(self):
        assert issubclass(ExecuteModelState, tuple)

    def test_construct_with_all_none_optionals(self):
        state = ExecuteModelState(
            scheduler_output=MagicMock(),
            logits=torch.zeros(2, 10),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.ones(2, 10),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
            slot_mappings=None,
        )
        assert state.spec_decode_metadata is None
        assert isinstance(state.logits, torch.Tensor)
        assert isinstance(state.hidden_states, torch.Tensor)

    def test_slot_mappings_dict(self):
        """slot_mappings can be a dict of tensors."""
        mappings = {"layer_0": torch.tensor([0, 1, 2])}
        state = ExecuteModelState(
            scheduler_output=MagicMock(),
            logits=torch.zeros(1),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.zeros(1),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
            slot_mappings=mappings,
        )
        assert "layer_0" in state.slot_mappings

    def test_slot_mappings_list_of_dicts(self):
        """slot_mappings can also be a list of dicts."""
        mappings = [
            {"layer_0": torch.tensor([0])},
            {"layer_1": torch.tensor([1])},
        ]
        state = ExecuteModelState(
            scheduler_output=MagicMock(),
            logits=torch.zeros(1),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.zeros(1),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
            slot_mappings=mappings,
        )
        assert len(state.slot_mappings) == 2


class TestDummyRunStateFeature:
    def test_field_names(self):
        expected_fields = (
            "attn_metadata",
            "num_input_tokens",
            "input_ids",
            "positions",
        )
        assert DummyRunState._fields == expected_fields

    def test_field_count(self):
        assert len(DummyRunState._fields) == 4

    def test_is_named_tuple(self):
        assert issubclass(DummyRunState, tuple)

    def test_typical_construction(self):
        state = DummyRunState(
            attn_metadata={0: {"key": "value"}},
            num_input_tokens=32,
            input_ids={0: torch.zeros(32, dtype=torch.long)},
            positions={0: torch.arange(32)},
        )
        assert state.num_input_tokens == 32
        assert isinstance(state.attn_metadata, dict)
        assert isinstance(state.input_ids, dict)
        assert isinstance(state.positions, dict)

    def test_empty_dicts(self):
        state = DummyRunState(
            attn_metadata={},
            num_input_tokens=0,
            input_ids={},
            positions={},
        )
        assert state.num_input_tokens == 0
        assert len(state.attn_metadata) == 0


# ===========================================================================
# 4. get_supported_tasks integration
# ===========================================================================


class TestGetSupportedTasksFeature:
    def _make_stub_with_task_methods(self, **kw):
        stub = _make_runner_stub(**kw)
        stub.get_model = _bind(stub, "get_model")
        stub.get_supported_generation_tasks = _bind(
            stub, "get_supported_generation_tasks"
        )
        stub.get_supported_pooling_tasks = _bind(
            stub, "get_supported_pooling_tasks"
        )
        stub.get_supported_tasks = _bind(stub, "get_supported_tasks")
        return stub

    def test_text_generation_model_returns_generate_task(self):
        stub = self._make_stub_with_task_methods()
        stub.model_config.runner_type = "generate"
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_text_generation_model",
            return_value=True,
        ), patch(
            "vllm_rbln.v1.worker.rbln_model_runner.supports_transcription",
            return_value=False,
        ):
            tasks = stub.get_supported_tasks()
        task_names = [t if isinstance(t, str) else t for t in tasks]
        assert "generate" in task_names

    def test_pooling_model_returns_pooling_tasks(self):
        model = MagicMock()
        model.pooler.get_supported_tasks.return_value = ["embed", "classify"]
        stub = self._make_stub_with_task_methods(model=model)
        stub.model_config.runner_type = "pooling"
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_pooling_model",
            return_value=True,
        ):
            tasks = stub.get_supported_tasks()
        task_names = list(tasks)
        assert "embed" in task_names
        assert "classify" in task_names

    def test_generate_runner_returns_no_pooling_tasks(self):
        """A generate runner should not include pooling tasks."""
        stub = self._make_stub_with_task_methods()
        stub.model_config.runner_type = "generate"
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_text_generation_model",
            return_value=True,
        ), patch(
            "vllm_rbln.v1.worker.rbln_model_runner.supports_transcription",
            return_value=False,
        ):
            tasks = stub.get_supported_tasks()
        # Should only contain generation tasks, not pooling
        for t in tasks:
            assert t not in ("embed", "classify", "score", "encode")

    def test_pooling_runner_returns_no_generate_tasks(self):
        """A pooling runner should not include generate tasks."""
        model = MagicMock()
        model.pooler.get_supported_tasks.return_value = ["embed"]
        stub = self._make_stub_with_task_methods(model=model)
        stub.model_config.runner_type = "pooling"
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_pooling_model",
            return_value=True,
        ):
            tasks = stub.get_supported_tasks()
        assert "generate" not in tasks

    def test_returns_tuple(self):
        stub = self._make_stub_with_task_methods()
        stub.model_config.runner_type = "generate"
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_text_generation_model",
            return_value=True,
        ), patch(
            "vllm_rbln.v1.worker.rbln_model_runner.supports_transcription",
            return_value=False,
        ):
            tasks = stub.get_supported_tasks()
        assert isinstance(tasks, tuple)


# ===========================================================================
# 5. Bug-catching: edge cases and integration
# ===========================================================================


class TestBucketingManagerIntegration:
    """Test bucketing manager usage patterns in RBLNModelRunner."""

    def test_bucketing_manager_find_decode_batch_bucket(self):
        """The bucketing manager stub should be used for decode bucket lookup."""
        from vllm_rbln.v1.worker.bucketing.exponential_bucketing_manager import (
            ExponentialBucketingManager,
        )

        mgr = ExponentialBucketingManager(
            max_batch_size=32, min_batch_size=1, limit=8, step=2
        )
        # Should find a bucket >= the given batch_size
        bucket = mgr.find_decode_batch_bucket(1)
        assert bucket >= 1

    def test_bucketing_manager_no_bucket_raises(self):
        """Requesting a bucket larger than max should raise."""
        from vllm_rbln.v1.worker.bucketing.exponential_bucketing_manager import (
            ExponentialBucketingManager,
        )

        mgr = ExponentialBucketingManager(
            max_batch_size=4, min_batch_size=1, limit=4, step=2
        )
        with pytest.raises(ValueError):
            mgr.find_decode_batch_bucket(10000)

    def test_bucketing_manager_batch_buckets_include_prefill(self):
        """batch_buckets should always include 1 (reserved for prefill)."""
        from vllm_rbln.v1.worker.bucketing.linear_bucketing_manager import (
            LinearBucketingManager,
        )

        mgr = LinearBucketingManager(
            max_batch_size=16, min_batch_size=1, limit=8, step=4
        )
        assert 1 in mgr.batch_buckets


class TestEdgeCases:
    """Bug-catching tests for edge conditions."""

    def test_execute_model_state_unpacking(self):
        """ExecuteModelState should support tuple unpacking (it is a NamedTuple)."""
        state = ExecuteModelState(
            scheduler_output="sched",
            logits=torch.zeros(1),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.ones(1),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
            slot_mappings=None,
        )
        (
            sched,
            logits,
            spec_meta,
            spec_common,
            hidden,
            sample_hidden,
            aux,
            kv_out,
            slots,
        ) = state
        assert sched == "sched"
        assert spec_meta is None

    def test_dummy_run_state_tuple_unpacking(self):
        """DummyRunState should support tuple unpacking."""
        state = DummyRunState(
            attn_metadata={"a": 1},
            num_input_tokens=5,
            input_ids={"b": 2},
            positions={"c": 3},
        )
        attn, num_tokens, ids, pos = state
        assert num_tokens == 5

    def test_async_output_with_empty_sampled_tokens(self):
        """AsyncRBLNModelRunnerOutput should handle zero requests gracefully."""
        mro = MagicMock(spec=ModelRunnerOutput)
        sampled = torch.zeros(0, 1, dtype=torch.long)
        output = AsyncRBLNModelRunnerOutput(
            model_runner_output=mro,
            sampled_token_ids=sampled,
            invalid_req_indices=[],
            async_output_copy_stream=MagicMock(),
        )
        output._sampled_token_ids_cpu = sampled
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()
        assert result.sampled_token_ids == []

    def test_get_model_returns_model_attribute(self):
        """get_model should return the model attribute directly."""
        model = MagicMock()
        stub = _make_runner_stub(model=model)
        result = RBLNModelRunner.get_model(stub)
        assert result is model

    def test_compute_logits_delegates_to_model(self):
        """compute_logits should delegate to model.compute_logits."""
        stub = _make_runner_stub()
        hidden = torch.randn(2, 10)
        expected = torch.randn(2, 100)
        stub.model.compute_logits.return_value = expected
        result = RBLNModelRunner.compute_logits(stub, hidden)
        stub.model.compute_logits.assert_called_once_with(hidden)

    def test_use_wrapped_compute_logits_default_true(self):
        """By default (no lora, no spec_decode), use_wrapped_compute_logits is True."""
        stub = _make_runner_stub(lora_config=None, speculative_config=None)
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is True

    def test_use_wrapped_compute_logits_false_with_lora(self):
        """With lora_config, use_wrapped_compute_logits is False."""
        stub = _make_runner_stub(lora_config=MagicMock())
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is False

    def test_mixin_mro_order(self):
        """LoRAModelRunnerMixin should come before KVConnectorModelRunnerMixin in MRO."""
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        mro = RBLNModelRunner.__mro__
        lora_idx = mro.index(LoRAModelRunnerMixin)
        kv_idx = mro.index(KVConnectorModelRunnerMixin)
        assert lora_idx < kv_idx, (
            "LoRAModelRunnerMixin should precede KVConnectorModelRunnerMixin in MRO"
        )

    def test_constructor_signature(self):
        """RBLNModelRunner.__init__ should accept (self, vllm_config, device)."""
        import inspect

        sig = inspect.signature(RBLNModelRunner.__init__)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "vllm_config" in params
        assert "device" in params
