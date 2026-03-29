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

"""Feature tests for RBLNWorker: interface compliance, WorkerBase contract,
device env initialization, memory estimation, lifecycle, and edge cases."""

import inspect
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed
from vllm.v1.worker.worker_base import WorkerBase


# ---------------------------------------------------------------------------
# Helpers (mirrors the unit test factory pattern)
# ---------------------------------------------------------------------------


def _make_profiler_config(trace_dir=None):
    return SimpleNamespace(
        torch_profiler_dir=trace_dir,
        torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False,
        torch_profiler_with_stack=False,
        torch_profiler_with_flops=False,
        torch_profiler_use_gzip=False,
    )


def _make_parallel_config(
    world_size=1,
    data_parallel_size=1,
    data_parallel_rank=0,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    world_size_across_dp=1,
):
    return SimpleNamespace(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        data_parallel_rank=data_parallel_rank,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        disable_custom_all_reduce=False,
        distributed_executor_backend=None,
        world_size_across_dp=world_size_across_dp,
    )


def _make_model_config(
    trust_remote_code=False,
    seed=42,
    quantization=None,
    enforce_eager=False,
):
    return SimpleNamespace(
        trust_remote_code=trust_remote_code,
        seed=seed,
        quantization=quantization,
        enforce_eager=enforce_eager,
    )


def _make_cache_config(gpu_memory_utilization=0.9):
    return SimpleNamespace(
        gpu_memory_utilization=gpu_memory_utilization,
        num_gpu_blocks=0,
        num_cpu_blocks=0,
    )


def _make_scheduler_config(max_num_batched_tokens=256, max_num_seqs=32):
    return SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )


def _make_vllm_config(
    profiler_trace_dir=None,
    trust_remote_code=False,
    quantization=None,
    enforce_eager=False,
    data_parallel_size=1,
    data_parallel_rank=0,
    world_size=1,
    world_size_across_dp=1,
):
    return SimpleNamespace(
        profiler_config=_make_profiler_config(profiler_trace_dir),
        parallel_config=_make_parallel_config(
            world_size=world_size,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            world_size_across_dp=world_size_across_dp,
        ),
        model_config=_make_model_config(
            trust_remote_code=trust_remote_code,
            quantization=quantization,
            enforce_eager=enforce_eager,
        ),
        cache_config=_make_cache_config(),
        scheduler_config=_make_scheduler_config(),
        instance_id="test-instance",
    )


@pytest.fixture(autouse=True)
def env_cleanup():
    """Save and restore environment variables touched by tests."""
    keys = [
        "RBLN_DEVICES",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "RBLN_NPUS_PER_DEVICE",
        "RCCL_PORT_GEN",
        "RBLN_NUM_THREADS",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# Worker factory
# ---------------------------------------------------------------------------

_INIT_PATCHES = {
    "current_platform": "vllm_rbln.v1.worker.rbln_worker.current_platform",
    "envs_tp": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_TP_SIZE",
    "envs_ray": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_NUM_RAY_NODES",
    "envs_auto_port": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_AUTO_PORT",
    "envs_compile": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_COMPILE_MODEL",
    "envs_warmup": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_ENABLE_WARM_UP",
    "envs_metrics": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS",
    "envs_dp_impl": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_DP_IMPL",
    "envs_numa": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_NUMA",
    "has_torch_rbln": "vllm_rbln.v1.worker.rbln_worker.has_torch_rbln",
}


def _fake_super_init(
    self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False
):
    self.vllm_config = vllm_config
    self.local_rank = local_rank
    self.rank = rank
    self.distributed_init_method = distributed_init_method
    self.is_driver_worker = is_driver_worker
    self.model_config = vllm_config.model_config
    self.parallel_config = vllm_config.parallel_config
    self.cache_config = vllm_config.cache_config
    self.scheduler_config = vllm_config.scheduler_config


def _create_worker(
    vllm_config=None,
    local_rank=0,
    rank=0,
    is_driver_worker=True,
    *,
    tp_size=1,
    num_ray_nodes=1,
    has_torch_rbln_val=False,
    envs_overrides=None,
):
    """Instantiate RBLNWorker with mocked-out heavy dependencies."""
    from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

    if vllm_config is None:
        vllm_config = _make_vllm_config()

    defaults = {
        "envs_tp": tp_size,
        "envs_ray": num_ray_nodes,
        "envs_auto_port": False,
        "envs_compile": True,
        "envs_warmup": True,
        "envs_metrics": False,
        "envs_dp_impl": "padded_decode",
        "envs_numa": False,
        "has_torch_rbln": has_torch_rbln_val,
    }
    if envs_overrides:
        defaults.update(envs_overrides)

    active = []
    try:
        # Patch WorkerBase.__init__
        p = patch.object(
            RBLNWorker.__bases__[0],
            "__init__",
            _fake_super_init,
        )
        active.append(p)
        p.start()

        # Patch current_platform
        platform_mock = MagicMock()
        platform_mock.device_type = "cpu"
        platform_mock.device_control_env_var = "RBLN_DEVICES"
        platform_mock.dist_backend = "gloo"
        platform_mock.get_device_name.return_value = "RBLN-CA25"
        p = patch(_INIT_PATCHES["current_platform"], platform_mock)
        active.append(p)
        p.start()

        # Patch scalar env values
        for key in (
            "envs_tp",
            "envs_ray",
            "envs_auto_port",
            "envs_compile",
            "envs_warmup",
            "envs_metrics",
            "envs_dp_impl",
            "envs_numa",
            "has_torch_rbln",
        ):
            p = patch(_INIT_PATCHES[key], defaults[key])
            active.append(p)
            p.start()

        worker = RBLNWorker(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method="tcp://localhost:12345",
            is_driver_worker=is_driver_worker,
        )
    finally:
        for p in active:
            p.stop()

    return worker


# ===========================================================================
# 1. Interface compliance: RBLNWorker implements all WorkerBase methods
# ===========================================================================


class TestInterfaceCompliance:
    """Verify RBLNWorker provides implementations for every method that
    WorkerBase declares (both abstract-style raise-NotImplementedError
    and regular methods)."""

    def _get_worker_base_interface_methods(self):
        """Return names of WorkerBase methods that subclasses should provide."""
        base_methods = []
        for name, obj in inspect.getmembers(WorkerBase, predicate=inspect.isfunction):
            if name.startswith("_") and name != "__init__":
                continue
            base_methods.append(name)
        return base_methods

    def _get_notimplemented_methods(self):
        """Return names of WorkerBase methods that raise NotImplementedError."""
        ni_methods = []
        for name, obj in inspect.getmembers(WorkerBase, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            src = inspect.getsource(obj)
            if "NotImplementedError" in src:
                ni_methods.append(name)
        return ni_methods

    def test_rbln_worker_extends_worker_base(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert issubclass(RBLNWorker, WorkerBase)

    def test_all_not_implemented_methods_are_overridden(self):
        """Every WorkerBase method that raises NotImplementedError must be
        overridden by RBLNWorker (except known intentional gaps)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        # Methods intentionally not overridden (e.g. speculative-decoding only)
        KNOWN_GAPS = {"get_cache_block_size_bytes"}

        ni_methods = self._get_notimplemented_methods()
        assert len(ni_methods) > 0, "Expected some NotImplementedError methods"

        missing = []
        for name in ni_methods:
            if name in KNOWN_GAPS:
                continue
            base_method = getattr(WorkerBase, name)
            child_method = getattr(RBLNWorker, name)
            if child_method is base_method:
                missing.append(name)

        assert missing == [], (
            f"RBLNWorker does not override these WorkerBase methods: {missing}"
        )

    def test_known_gaps_documented(self):
        """Verify that get_cache_block_size_bytes is indeed not overridden
        (intentional gap for speculative decoding)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert RBLNWorker.get_cache_block_size_bytes is WorkerBase.get_cache_block_size_bytes

    def test_init_signature_matches_worker_base(self):
        """RBLNWorker.__init__ must accept the same parameters as WorkerBase."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        base_sig = inspect.signature(WorkerBase.__init__)
        child_sig = inspect.signature(RBLNWorker.__init__)

        base_params = list(base_sig.parameters.keys())
        child_params = list(child_sig.parameters.keys())

        assert base_params == child_params, (
            f"Signature mismatch: base={base_params}, child={child_params}"
        )

    def test_execute_model_signature_compatible(self):
        """execute_model must accept scheduler_output positional arg."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.execute_model)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "scheduler_output" in params

    def test_compile_or_warm_up_model_returns_float(self):
        """compile_or_warm_up_model must return a float (elapsed time)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.compile_or_warm_up_model)
        # The return annotation should be float
        assert sig.return_annotation is float or sig.return_annotation == inspect.Parameter.empty

    def test_shutdown_is_overridden(self):
        """shutdown must be overridden (not the base no-op)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert RBLNWorker.shutdown is not WorkerBase.shutdown

    def test_check_health_is_overridden(self):
        """check_health must be overridden."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert RBLNWorker.check_health is not WorkerBase.check_health

    def test_sleep_wake_up_are_defined(self):
        """sleep and wake_up must be defined on RBLNWorker."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert hasattr(RBLNWorker, "sleep")
        assert hasattr(RBLNWorker, "wake_up")
        assert callable(getattr(RBLNWorker, "sleep"))
        assert callable(getattr(RBLNWorker, "wake_up"))


# ===========================================================================
# 2. WorkerBase contract: class hierarchy and method signatures
# ===========================================================================


class TestWorkerBaseContract:
    """Verify the class hierarchy and that method signatures match vllm
    expectations for pluggable workers."""

    def test_mro_includes_worker_base(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert WorkerBase in RBLNWorker.__mro__

    def test_direct_parent_is_worker_base(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert RBLNWorker.__bases__[0] is WorkerBase

    def test_determine_available_memory_returns_int(self):
        """determine_available_memory should return int (bytes)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.determine_available_memory)
        ret = sig.return_annotation
        assert ret is int or ret == inspect.Parameter.empty

    def test_initialize_cache_signature(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.initialize_cache)
        params = list(sig.parameters.keys())
        assert "num_gpu_blocks" in params
        assert "num_cpu_blocks" in params

    def test_load_model_takes_no_args(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.load_model)
        # Only self
        non_self = [p for p in sig.parameters if p != "self"]
        assert non_self == []

    def test_get_kv_cache_spec_returns_dict_annotation(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.get_kv_cache_spec)
        # Should have no positional args beyond self
        non_self = [p for p in sig.parameters if p != "self"]
        assert non_self == []


# ===========================================================================
# 3. Device env initialization
# ===========================================================================


class TestDeviceEnvInitialization:
    """Test _init_device_env under various TP, ray node, and rank configs."""

    def test_single_device_tp1(self):
        """TP=1, world_size=1 => RBLN_DEVICES=0."""
        worker = _create_worker(tp_size=1)
        assert os.environ["RBLN_DEVICES"] == "0"

    def test_tp4_single_worker(self):
        """TP=4, world_size=1, local_rank=0 => devices 0,1,2,3."""
        worker = _create_worker(tp_size=4)
        assert os.environ["RBLN_DEVICES"] == "0,1,2,3"

    def test_tp4_multi_worker_rank1(self):
        """TP=4, world_size=2, local_rank=1 => devices 4,5,6,7."""
        cfg = _make_vllm_config(world_size=2)
        worker = _create_worker(vllm_config=cfg, local_rank=1, rank=1, tp_size=4)
        assert os.environ["RBLN_DEVICES"] == "4,5,6,7"

    def test_multiple_ray_nodes(self):
        """world_size=8, num_ray_nodes=2 => local_world_size=4.
        local_rank=0, tp_size=1 => device 0."""
        cfg = _make_vllm_config(world_size=8)
        worker = _create_worker(
            vllm_config=cfg, local_rank=0, rank=0, tp_size=1, num_ray_nodes=2
        )
        assert worker.local_world_size == 4
        assert os.environ["RBLN_DEVICES"] == "0"

    def test_multiple_ray_nodes_rank3(self):
        """world_size=8, num_ray_nodes=2, local_rank=3, tp_size=1
        => local_world_size=4, device 3."""
        cfg = _make_vllm_config(world_size=8)
        worker = _create_worker(
            vllm_config=cfg, local_rank=3, rank=3, tp_size=1, num_ray_nodes=2
        )
        assert os.environ["RBLN_DEVICES"] == "3"

    def test_dp_rank_offsets_device_ids(self):
        """data_parallel_rank=1 should offset device IDs.
        world_size=2, dp_rank=1, tp_size=1, local_rank=0 =>
        dev_begin = 2*1*1 = 2, dev_end = 4, selected = device_ids[0] = '2'."""
        cfg = _make_vllm_config(world_size=2, data_parallel_rank=1)
        worker = _create_worker(vllm_config=cfg, local_rank=0, rank=0, tp_size=1)
        assert os.environ["RBLN_DEVICES"] == "2"

    def test_explicit_rbln_devices_with_tp(self):
        """When RBLN_DEVICES is set externally with TP>1, it should expand device IDs."""
        os.environ["RBLN_DEVICES"] = "0,1"
        cfg = _make_vllm_config(world_size=2)
        worker = _create_worker(
            vllm_config=cfg, local_rank=0, rank=0, tp_size=2
        )
        # device_id=0, start_idx=0*2=0, end_idx=2 => "0,1"
        assert os.environ["RBLN_DEVICES"] == "0,1"

    def test_explicit_rbln_devices_tp_rank1(self):
        """RBLN_DEVICES=0,1, local_rank=1, tp_size=2 =>
        device_id=1, start=2, end=4 => '2,3'."""
        os.environ["RBLN_DEVICES"] = "0,1"
        cfg = _make_vllm_config(world_size=2)
        worker = _create_worker(
            vllm_config=cfg, local_rank=1, rank=1, tp_size=2
        )
        assert os.environ["RBLN_DEVICES"] == "2,3"

    def test_tp_gt1_sets_npus_per_device_when_torch_rbln(self):
        """TP > 1 with torch_rbln should set RBLN_NPUS_PER_DEVICE."""
        worker = _create_worker(tp_size=4, has_torch_rbln_val=True)
        assert os.environ.get("RBLN_NPUS_PER_DEVICE") == "4"

    def test_tp_gt1_no_npus_without_torch_rbln(self):
        """TP > 1 without torch_rbln should NOT set RBLN_NPUS_PER_DEVICE."""
        worker = _create_worker(tp_size=4, has_torch_rbln_val=False)
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ

    def test_tp1_no_npus_per_device(self):
        """TP=1 should never set RBLN_NPUS_PER_DEVICE even with torch_rbln."""
        worker = _create_worker(tp_size=1, has_torch_rbln_val=True)
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ


# ===========================================================================
# 4. Memory estimation integration
# ===========================================================================


class TestMemoryEstimation:
    """Test determine_available_memory with various quantization and MOE configs."""

    def _setup_worker(self, quantization=None, specialized_moe=False,
                      bucket_count=1, extra_quant_params=None):
        cfg = _make_vllm_config(quantization=quantization)
        worker = _create_worker(vllm_config=cfg)

        mock_model = MagicMock()
        p_bf16 = torch.zeros(200, dtype=torch.bfloat16)
        params = [("attn.weight", p_bf16)]
        if extra_quant_params:
            params.extend(extra_quant_params)
        else:
            p_quant = torch.zeros(100, dtype=torch.uint8)
            params.append(("mlp.weight", p_quant))
        mock_model.named_parameters.return_value = params

        runner = MagicMock()
        runner.model = mock_model
        runner.specialized_moe_decode = specialized_moe
        runner.bucketing_manager.decode_batch_buckets_count = bucket_count
        worker.model_runner = runner
        return worker

    def test_fp8_quantization_nbits(self):
        worker = self._setup_worker(quantization="fp8")
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.current_platform"
        ) as plat, patch(
            "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
            return_value=2 * 10**9,
        ) as est:
            plat.get_device_name.return_value = "RBLN-CA25"
            result = worker.determine_available_memory()

        assert est.call_args.kwargs["nbits_per_param"] == 8
        assert result == 2 * 10**9

    def test_mxfp4_on_atom_ca_device(self):
        """mxfp4 on ATOM (ca) device uses bf16 (16 bits)."""
        worker = self._setup_worker(quantization="mxfp4")
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.current_platform"
        ) as plat, patch(
            "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
            return_value=10**9,
        ) as est:
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()

        assert est.call_args.kwargs["nbits_per_param"] == 16

    def test_mxfp4_on_rebel_cr_device(self):
        """mxfp4 on REBEL (cr) device uses 4 bits."""
        worker = self._setup_worker(quantization="mxfp4")
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.current_platform"
        ) as plat, patch(
            "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
            return_value=10**9,
        ) as est:
            plat.get_device_name.return_value = "RBLN-CR100"
            worker.determine_available_memory()

        assert est.call_args.kwargs["nbits_per_param"] == 4

    def test_mxfp4_unknown_device_raises(self):
        """mxfp4 on unknown device architecture should raise ValueError."""
        worker = self._setup_worker(quantization="mxfp4")
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.current_platform"
        ) as plat:
            plat.get_device_name.return_value = "RBLN-XX99"
            with pytest.raises(ValueError, match="invalid RBLN architecture"):
                worker.determine_available_memory()

    def test_moe_config_num_runtimes(self):
        """specialized_moe_decode=True, bucket_count=3 =>
        num_runtimes = 1 + (1+1)*3 = 7."""
        worker = self._setup_worker(specialized_moe=True, bucket_count=3)
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.current_platform"
        ) as plat, patch(
            "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
            return_value=10**9,
        ) as est:
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()

        assert est.call_args.kwargs["num_runtimes"] == 7

    def test_no_quantization_default_16bits(self):
        worker = self._setup_worker(quantization=None)
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.current_platform"
        ) as plat, patch(
            "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
            return_value=10**9,
        ) as est:
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()

        assert est.call_args.kwargs["nbits_per_param"] == 16

    def test_mxfp4_atom_ratio_applied_to_experts(self):
        """On ATOM with mxfp4, non-bf16 params get ratio=16/17 applied.
        100 uint8 elems * packed_num_elems=2 * ratio=16/17."""
        worker = self._setup_worker(quantization="mxfp4")
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.current_platform"
        ) as plat, patch(
            "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
            return_value=10**9,
        ) as est:
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()

        n_model_params = est.call_args.kwargs["n_model_params"]
        # 200 bf16 attention params + 100 * 2 * (16/17) expert params
        expected_experts = 100 * 2 * (16 / 17)
        expected_total = 200 + expected_experts
        assert abs(n_model_params - expected_total) < 1e-6

    def test_fp8_mixed_dtype_params(self):
        """fp8: bf16 params counted as attention, uint8 as experts with packed=1."""
        extra = [("expert.weight", torch.zeros(50, dtype=torch.uint8))]
        worker = self._setup_worker(quantization="fp8", extra_quant_params=extra)
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.current_platform"
        ) as plat, patch(
            "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
            return_value=10**9,
        ) as est:
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()

        # 200 bf16 + 50 * 1 * 1.0 = 250
        assert est.call_args.kwargs["n_model_params"] == 250.0


# ===========================================================================
# 5. Lifecycle: shutdown, sleep, wake_up
# ===========================================================================


class TestLifecycle:
    """Test lifecycle methods: shutdown with metrics, sleep/wake_up no-ops."""

    def test_shutdown_with_metrics_calls_all_trackers(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.performance_tracker = MagicMock()
        worker.model_runner.sampler_performance_tracker = MagicMock()
        worker.model_runner.e2e_performance_tracker = MagicMock()

        with patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS", True):
            worker.shutdown()

        worker.model_runner.performance_tracker.print_final_stats.assert_called_once()
        worker.model_runner.sampler_performance_tracker.print_final_stats.assert_called_once()
        worker.model_runner.e2e_performance_tracker.print_final_stats.assert_called_once()

    def test_shutdown_without_metrics_skips_trackers(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        perf = MagicMock()
        worker.model_runner.performance_tracker = perf

        with patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS", False):
            worker.shutdown()

        perf.print_final_stats.assert_not_called()

    def test_shutdown_with_none_trackers_no_crash(self):
        """When metrics enabled but trackers are None, shutdown should not crash."""
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.performance_tracker = None
        worker.model_runner.sampler_performance_tracker = None
        worker.model_runner.e2e_performance_tracker = None

        with patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS", True):
            # Should not raise
            worker.shutdown()

    def test_sleep_is_noop(self):
        """sleep() should not raise or modify state."""
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.sleep(level=1)
        worker.sleep(level=0)
        worker.sleep()

    def test_wake_up_is_noop(self):
        """wake_up() should not raise or modify state."""
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.wake_up(tags=["tag1"])
        worker.wake_up(tags=None)
        worker.wake_up()

    def test_sleep_then_wake_up_sequence(self):
        """Calling sleep then wake_up in sequence should be safe."""
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.sleep(level=2)
        worker.wake_up(tags=["restore"])
        # No exception means success


# ===========================================================================
# 6. Bug-catching edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases that could reveal bugs in the implementation."""

    def test_local_world_size_not_divisible(self):
        """world_size=7, num_ray_nodes=2 => 7 // 2 = 3 (integer division).
        This is a potential bug if the code expects even division.
        The worker should still be created (Python integer division truncates)."""
        cfg = _make_vllm_config(world_size=7)
        worker = _create_worker(vllm_config=cfg, num_ray_nodes=2)
        # 7 // 2 = 3 due to integer division
        assert worker.local_world_size == 3

    def test_local_world_size_single_ray_node(self):
        """world_size=4, num_ray_nodes=1 => local_world_size=4."""
        cfg = _make_vllm_config(world_size=4)
        worker = _create_worker(vllm_config=cfg, num_ray_nodes=1)
        assert worker.local_world_size == 4

    def test_tp4_but_only_2_devices_explicit(self):
        """If RBLN_DEVICES has only 2 entries but world_size expects 2,
        and tp_size=4, the device expansion works per-device.
        RBLN_DEVICES='0,1', local_rank=0, tp_size=4 =>
        device_id=0, start=0, end=4 => '0,1,2,3'."""
        os.environ["RBLN_DEVICES"] = "0,1"
        cfg = _make_vllm_config(world_size=2)
        worker = _create_worker(
            vllm_config=cfg, local_rank=0, rank=0, tp_size=4
        )
        assert os.environ["RBLN_DEVICES"] == "0,1,2,3"

    def test_wrong_device_count_in_env(self):
        """RBLN_DEVICES has 3 entries but local_world_size is 2 => AssertionError."""
        os.environ["RBLN_DEVICES"] = "0,1,2"
        cfg = _make_vllm_config(world_size=2)
        with pytest.raises(AssertionError, match="should have device count"):
            _create_worker(vllm_config=cfg)

    def test_invalid_device_ids_non_integer(self):
        """Non-integer device IDs should raise ValueError."""
        os.environ["RBLN_DEVICES"] = "abc"
        with pytest.raises(ValueError, match="should be a list of integers"):
            _create_worker()

    def test_compile_warm_up_backend_compiler_failed_oom_enomem(self):
        """BackendCompilerFailed with OOM inner should raise RuntimeError."""
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.kv_cache_config.num_blocks = 64

        inner = RuntimeError("SYS_ENOMEM: Out of memory")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warm_up_model.side_effect = exc

        with pytest.raises(RuntimeError, match="Not enough memory"):
            worker.compile_or_warm_up_model()

    def test_compile_warm_up_backend_compiler_failed_ebusy(self):
        """BackendCompilerFailed with EBUSY inner should raise RuntimeError."""
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.kv_cache_config.num_blocks = 32

        inner = RuntimeError("SYS_EBUSY: Lack of device memory")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warm_up_model.side_effect = exc

        with pytest.raises(RuntimeError, match="Not enough memory"):
            worker.compile_or_warm_up_model()

    def test_compile_warm_up_non_oom_reraises(self):
        """BackendCompilerFailed with non-OOM inner should re-raise as-is."""
        worker = _create_worker()
        worker.model_runner = MagicMock()

        inner = RuntimeError("Some other compiler error")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warm_up_model.side_effect = exc

        with pytest.raises(BackendCompilerFailed):
            worker.compile_or_warm_up_model()

    def test_compile_warm_up_non_runtime_inner(self):
        """BackendCompilerFailed wrapping a non-RuntimeError should re-raise."""
        worker = _create_worker()
        worker.model_runner = MagicMock()

        inner = TypeError("wrong type")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warm_up_model.side_effect = exc

        with pytest.raises(BackendCompilerFailed):
            worker.compile_or_warm_up_model()

    def test_compile_warm_up_dp_dummy_prefill_raises(self):
        """dummy_prefill DP impl should raise ValueError in v1."""
        cfg = _make_vllm_config(data_parallel_size=2)
        worker = _create_worker(vllm_config=cfg)
        worker.model_runner = MagicMock()

        with patch(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_DP_IMPL",
            "dummy_prefill",
        ):
            with pytest.raises(ValueError, match="dummy_prefill is not supported"):
                worker.compile_or_warm_up_model()

    def test_compile_warm_up_skipped_when_enforce_eager(self):
        """enforce_eager=True should skip warm_up_model but still enable tracker."""
        cfg = _make_vllm_config(enforce_eager=True)
        worker = _create_worker(vllm_config=cfg)
        worker.model_runner = MagicMock()

        elapsed = worker.compile_or_warm_up_model()
        worker.model_runner.warm_up_model.assert_not_called()
        worker.model_runner._enable_performance_tracker.assert_called_once()
        assert isinstance(elapsed, float)
        assert elapsed >= 0

    def test_device_env_with_dp_rank1_tp2(self):
        """data_parallel_rank=1, world_size=2, tp_size=2 =>
        dev_begin=2*2*1=4, dev_end=8, device_ids=[4,5,6,7],
        local_rank=0 => start=0, end=2 => '4,5'."""
        cfg = _make_vllm_config(world_size=2, data_parallel_rank=1)
        worker = _create_worker(
            vllm_config=cfg, local_rank=0, rank=0, tp_size=2
        )
        assert os.environ["RBLN_DEVICES"] == "4,5"

    def test_check_health_returns_none(self):
        """check_health should always return None (healthy)."""
        worker = _create_worker()
        assert worker.check_health() is None
