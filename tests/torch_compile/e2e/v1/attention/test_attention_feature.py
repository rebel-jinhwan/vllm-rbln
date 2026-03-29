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

"""Feature tests for the RBLN attention backend: interface compliance,
custom op registration, and reference implementation correctness."""

import pytest
import torch


@pytest.fixture
def backend_module():
    from vllm_rbln.v1.attention.backends import flash_attention

    return flash_attention


@pytest.fixture
def backend_cls(backend_module):
    return backend_module.RBLNAttentionBackend


# ---------------------------------------------------------------------------
# 1. AttentionBackend interface compliance
# ---------------------------------------------------------------------------


class TestAttentionBackendInterface:
    """Verify RBLNAttentionBackend satisfies the vllm AttentionBackend ABC."""

    def test_get_name_returns_rbln_attn(self, backend_cls):
        assert backend_cls.get_name() == "RBLN_ATTN"

    def test_get_impl_cls_returns_rbln_flash_attention_impl(
        self, backend_module, backend_cls
    ):
        assert backend_cls.get_impl_cls() is backend_module.RBLNFlashAttentionImpl

    def test_get_builder_cls_returns_rbln_flash_attention_metadata_builder(
        self, backend_module, backend_cls
    ):
        assert (
            backend_cls.get_builder_cls()
            is backend_module.RBLNFlashAttentionMetadataBuilder
        )

    def test_get_supported_head_sizes_contains_common_sizes(self, backend_cls):
        sizes = backend_cls.get_supported_head_sizes()
        for expected in [32, 64, 128, 256]:
            assert expected in sizes

    def test_get_supported_head_sizes_is_sorted(self, backend_cls):
        sizes = backend_cls.get_supported_head_sizes()
        assert sizes == sorted(sizes)

    @pytest.mark.parametrize(
        "num_blocks, block_size, num_kv_heads, head_size",
        [
            (4, 128, 8, 64),
            (1, 32, 1, 32),
            (16, 1024, 32, 128),
        ],
    )
    def test_get_kv_cache_shape(
        self, backend_cls, num_blocks, block_size, num_kv_heads, head_size
    ):
        shape = backend_cls.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )
        assert shape == (2, num_blocks, num_kv_heads, 1, block_size, head_size)

    def test_swap_blocks_raises_runtime_error(self, backend_cls):
        dummy = torch.empty(0)
        with pytest.raises(RuntimeError):
            backend_cls.swap_blocks(dummy, dummy, {})

    def test_copy_blocks_raises_runtime_error(self, backend_cls):
        with pytest.raises(RuntimeError):
            backend_cls.copy_blocks([], {})


# ---------------------------------------------------------------------------
# 2. Custom op registration
# ---------------------------------------------------------------------------


class TestCustomOpRegistration:
    """All expected custom ops should be registered and callable."""

    @pytest.mark.parametrize(
        "op_name",
        [
            "attention_naive_prefill",
            "attention_naive_decode",
            "causal_attention_naive_prefill",
            "causal_attention_naive_decode",
            "flash_attention_naive_prefill",
            "flash_attention_naive_decode",
            "flash_causal_attention_naive_prefill",
            "flash_causal_attention_naive_decode",
            "sliding_window_attention_naive_prefill",
            "sliding_window_attention_naive_decode",
        ],
    )
    def test_op_is_registered_in_rbln_custom_ops(self, backend_module, op_name):
        """The op should be accessible via torch.ops.rbln_custom_ops."""
        op = getattr(torch.ops.rbln_custom_ops, op_name, None)
        assert op is not None, f"torch.ops.rbln_custom_ops.{op_name} not registered"
        assert callable(op)


# ---------------------------------------------------------------------------
# 3. Reference implementation correctness (flash_attention_naive_prefill_impl)
# ---------------------------------------------------------------------------

# Helpers to build tensors in the 5-D layout expected by the reference impls.


def _make_prefill_inputs(
    seq_len: int = 4,
    n_kv_heads: int = 1,
    n_groups: int = 4,
    head_dim: int = 32,
    partition_size: int = 8,
    num_blocks: int = 2,
    cache_start: int = 0,
):
    """Return (q, k, v, kv_cache, mask, scale, seq_idx, block_tables, slot_mapping)
    suitable for flash_attention_naive_prefill_impl."""
    batch = 1
    q = torch.randn(batch, n_kv_heads, n_groups, seq_len, head_dim)
    k = torch.randn(batch, n_kv_heads, 1, seq_len, head_dim)
    v = torch.randn(batch, n_kv_heads, 1, seq_len, head_dim)
    kv_cache = torch.zeros(2, num_blocks, n_kv_heads, 1, partition_size, head_dim)
    # mask: 1 means attend, 0 means mask-out
    mask = torch.zeros(batch, 1, 1, seq_len, partition_size)
    for i in range(seq_len):
        mask[0, 0, 0, i, : cache_start + i + 1] = 1.0
    scale = torch.tensor(1.0 / (head_dim**0.5))
    seq_idx = torch.tensor([[cache_start]], dtype=torch.int32)
    block_tables = torch.tensor([0], dtype=torch.int32)
    slot_mapping = torch.zeros(seq_len, dtype=torch.int32)
    return q, k, v, kv_cache, mask, scale, seq_idx, block_tables, slot_mapping


def _make_decode_inputs(
    n_kv_heads: int = 1,
    n_groups: int = 4,
    head_dim: int = 32,
    partition_size: int = 8,
    num_blocks: int = 2,
    cache_start: int = 3,
):
    """Return (q, k, v, kv_cache, mask, scale, seq_idx, block_tables, slot_mapping)
    suitable for flash_attention_naive_decode_impl."""
    batch = 1
    seq_len = 1
    q = torch.randn(batch, n_kv_heads, n_groups, seq_len, head_dim)
    k = torch.randn(batch, n_kv_heads, 1, seq_len, head_dim)
    v = torch.randn(batch, n_kv_heads, 1, seq_len, head_dim)
    kv_cache = torch.zeros(2, num_blocks, n_kv_heads, 1, partition_size, head_dim)
    # Pre-fill the cache with some data up to cache_start
    if cache_start > 0:
        kv_cache[0, 0, :, :, :cache_start, :] = torch.randn(
            n_kv_heads, 1, cache_start, head_dim
        )
        kv_cache[1, 0, :, :, :cache_start, :] = torch.randn(
            n_kv_heads, 1, cache_start, head_dim
        )
    mask = torch.zeros(batch, 1, 1, seq_len, partition_size)
    mask[0, 0, 0, 0, : cache_start + 1] = 1.0
    scale = torch.tensor(1.0 / (head_dim**0.5))
    seq_idx = torch.tensor([[cache_start]], dtype=torch.int32)
    block_tables = torch.tensor([[0]], dtype=torch.int32)
    slot_mapping = torch.zeros(seq_len, dtype=torch.int32)
    return q, k, v, kv_cache, mask, scale, seq_idx, block_tables, slot_mapping


class TestFlashAttentionNaivePrefillImpl:
    """Test the reference implementation in flash_attention_naive_prefill_impl
    with VLLM_RBLN_COMPILE_MODEL=False."""

    def test_output_shape_matches_query(self, monkeypatch, backend_module):
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        q, k, v, kv_cache, mask, scale, seq_idx, bt, sm = _make_prefill_inputs(
            seq_len=4
        )
        out = backend_module.flash_attention_naive_prefill_impl(
            q, k, v, kv_cache, mask, scale, seq_idx, bt, sm
        )
        assert out.shape == q.shape

    def test_kv_cache_is_mutated(self, monkeypatch, backend_module):
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        q, k, v, kv_cache, mask, scale, seq_idx, bt, sm = _make_prefill_inputs(
            seq_len=2
        )
        cache_before = kv_cache.clone()
        backend_module.flash_attention_naive_prefill_impl(
            q, k, v, kv_cache, mask, scale, seq_idx, bt, sm
        )
        # The K and V caches should have been written to
        assert not torch.equal(kv_cache, cache_before)

    def test_causal_masking_prevents_future_attention(
        self, monkeypatch, backend_module
    ):
        """First token should not attend to later positions."""
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        head_dim = 32
        seq_len = 4
        partition_size = 8
        q, k, v, kv_cache, mask, scale, seq_idx, bt, sm = _make_prefill_inputs(
            seq_len=seq_len, head_dim=head_dim, partition_size=partition_size
        )
        # Set all values to 1 so attention differences are observable
        k[:] = 1.0
        v[:] = 0.0
        # Make one specific position's value distinctive
        v[0, 0, 0, seq_len - 1, :] = 100.0

        out = backend_module.flash_attention_naive_prefill_impl(
            q, k, v, kv_cache, mask, scale, seq_idx, bt, sm
        )
        # First token (position 0) should NOT attend to the last token,
        # so its output should be 0 (all v it can attend to are 0).
        first_token_out = out[0, 0, 0, 0, :]
        assert torch.allclose(first_token_out, torch.zeros_like(first_token_out))

    def test_scaling_factor_is_applied(self, monkeypatch, backend_module):
        """Doubling the scale should change the output (softmax is not linear)."""
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        q, k, v, kv_cache, mask, scale, seq_idx, bt, sm = _make_prefill_inputs(
            seq_len=3
        )
        out_s1 = backend_module.flash_attention_naive_prefill_impl(
            q, k, v, kv_cache.clone(), mask, scale, seq_idx, bt, sm
        )
        out_s2 = backend_module.flash_attention_naive_prefill_impl(
            q, k, v, kv_cache.clone(), mask, scale * 2.0, seq_idx, bt, sm
        )
        assert not torch.allclose(out_s1, out_s2)

    def test_compile_model_true_returns_empty_like(self, monkeypatch, backend_module):
        """When VLLM_RBLN_COMPILE_MODEL is True, should return empty_like(q)."""
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", True)
        q, k, v, kv_cache, mask, scale, seq_idx, bt, sm = _make_prefill_inputs()
        out = backend_module.flash_attention_naive_prefill_impl(
            q, k, v, kv_cache, mask, scale, seq_idx, bt, sm
        )
        assert out.shape == q.shape


class TestFlashAttentionNaiveDecodeImpl:
    """Test the reference implementation in flash_attention_naive_decode_impl
    with VLLM_RBLN_COMPILE_MODEL=False."""

    def test_output_shape_matches_query(self, monkeypatch, backend_module):
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        q, k, v, kv_cache, mask, scale, seq_idx, bt, sm = _make_decode_inputs()
        out = backend_module.flash_attention_naive_decode_impl(
            q, k, v, kv_cache, mask, scale, seq_idx, bt, sm
        )
        assert out.shape == q.shape

    def test_kv_cache_is_mutated_on_decode(self, monkeypatch, backend_module):
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        q, k, v, kv_cache, mask, scale, seq_idx, bt, sm = _make_decode_inputs(
            cache_start=2
        )
        cache_before = kv_cache.clone()
        backend_module.flash_attention_naive_decode_impl(
            q, k, v, kv_cache, mask, scale, seq_idx, bt, sm
        )
        assert not torch.equal(kv_cache, cache_before)

    def test_batch_size_gt1_raises_assertion(self, monkeypatch, backend_module):
        """The reference decode impl asserts batch_size == 1."""
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        head_dim = 32
        n_kv_heads = 1
        n_groups = 4
        partition_size = 8
        batch = 2
        seq_len = 1
        q = torch.randn(batch, n_kv_heads, n_groups, seq_len, head_dim)
        k = torch.randn(batch, n_kv_heads, 1, seq_len, head_dim)
        v = torch.randn(batch, n_kv_heads, 1, seq_len, head_dim)
        kv_cache = torch.zeros(2, 2, n_kv_heads, 1, partition_size, head_dim)
        mask = torch.ones(batch, 1, 1, seq_len, partition_size)
        scale = torch.tensor(1.0 / (head_dim**0.5))
        seq_idx = torch.tensor([[0], [0]], dtype=torch.int32)
        block_tables = torch.tensor([[0], [1]], dtype=torch.int32)
        slot_mapping = torch.zeros(seq_len, dtype=torch.int32)
        with pytest.raises(AssertionError):
            backend_module.flash_attention_naive_decode_impl(
                q, k, v, kv_cache, mask, scale, seq_idx, block_tables, slot_mapping
            )


# ---------------------------------------------------------------------------
# 4. Bug-catching / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and bug-catching tests."""

    def test_head_size_not_in_supported_list(self, backend_cls):
        supported = backend_cls.get_supported_head_sizes()
        assert 17 not in supported, "17 should not be a supported head size"
        assert 0 not in supported, "0 should not be a supported head size"

    def test_kv_cache_shape_is_tuple(self, backend_cls):
        shape = backend_cls.get_kv_cache_shape(4, 128, 8, 64)
        assert isinstance(shape, tuple)

    def test_prefill_single_token(self, monkeypatch, backend_module):
        """Prefill with a single token should work and write to cache."""
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        q, k, v, kv_cache, mask, scale, seq_idx, bt, sm = _make_prefill_inputs(
            seq_len=1
        )
        cache_before = kv_cache.clone()
        out = backend_module.flash_attention_naive_prefill_impl(
            q, k, v, kv_cache, mask, scale, seq_idx, bt, sm
        )
        assert out.shape == q.shape
        assert not torch.equal(kv_cache, cache_before)

    def test_decode_after_prefill_consistency(self, monkeypatch, backend_module):
        """Decode using the same cache written by prefill should produce
        a valid (finite) output."""
        monkeypatch.setattr(backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        head_dim = 32
        n_kv_heads = 1
        n_groups = 4
        partition_size = 8
        num_blocks = 2

        # First: prefill 3 tokens
        pq, pk, pv, kv_cache, pmask, scale, seq_idx, bt_p, sm_p = (
            _make_prefill_inputs(
                seq_len=3,
                n_kv_heads=n_kv_heads,
                n_groups=n_groups,
                head_dim=head_dim,
                partition_size=partition_size,
                num_blocks=num_blocks,
                cache_start=0,
            )
        )
        backend_module.flash_attention_naive_prefill_impl(
            pq, pk, pv, kv_cache, pmask, scale, seq_idx, bt_p, sm_p
        )

        # Then: decode 1 token at position 3
        batch = 1
        seq_len = 1
        cache_start = 3
        dq = torch.randn(batch, n_kv_heads, n_groups, seq_len, head_dim)
        dk = torch.randn(batch, n_kv_heads, 1, seq_len, head_dim)
        dv = torch.randn(batch, n_kv_heads, 1, seq_len, head_dim)
        dmask = torch.zeros(batch, 1, 1, seq_len, partition_size)
        dmask[0, 0, 0, 0, : cache_start + 1] = 1.0
        d_seq_idx = torch.tensor([[cache_start]], dtype=torch.int32)
        d_bt = torch.tensor([[0]], dtype=torch.int32)
        d_sm = torch.zeros(seq_len, dtype=torch.int32)

        out = backend_module.flash_attention_naive_decode_impl(
            dq, dk, dv, kv_cache, dmask, scale, d_seq_idx, d_bt, d_sm
        )
        assert out.shape == dq.shape
        assert torch.isfinite(out).all()
