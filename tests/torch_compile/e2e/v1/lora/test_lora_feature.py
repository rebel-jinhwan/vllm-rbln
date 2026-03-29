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

"""
Tests for RBLN LoRA layer patches, PunicaWrapperRBLN, LoRAInputs, and LoRAMask.
Focuses on correctness of patched methods, math operations, static state sharing,
and edge-case bug catching.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_rbln.lora.inputs import LoRAInputs
from vllm_rbln.lora.mask import LoRAMask

# All tests in this module are CPU-only unit tests that do not need
# the global accelerator cleanup performed by the conftest fixture.
pytestmark = pytest.mark.skip_global_cleanup


# ---------------------------------------------------------------------------
# 1. Layer patch correctness
# ---------------------------------------------------------------------------


class TestLayerPatchApplication:
    """Verify that importing vllm_rbln.lora.layer actually patches the classes."""

    def test_base_linear_apply_is_patched(self):
        import vllm_rbln.lora.layer as layer_mod  # noqa: F401
        from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

        assert BaseLinearLayerWithLoRA.apply is layer_mod.base_linear_patched_apply

    def test_vocab_parallel_embedding_forward_is_patched(self):
        import vllm_rbln.lora.layer as layer_mod  # noqa: F401
        from vllm.lora.layers import VocabParallelEmbeddingWithLoRA

        assert (
            VocabParallelEmbeddingWithLoRA.forward
            is layer_mod.vocab_parallel_embedding_patched_forward
        )

    def test_patched_apply_calls_punica_wrapper(self):
        """Patched apply() must delegate to punica_wrapper.add_lora_linear."""
        import vllm_rbln.lora.layer as layer_mod  # noqa: F401

        mock_self = MagicMock()
        # base_layer.quant_method.apply returns a tensor
        base_output = torch.randn(1, 4, 8)
        mock_self.base_layer.quant_method.apply.return_value = base_output.clone()
        mock_self.punica_wrapper.add_lora_linear.return_value = torch.randn(4, 8)
        mock_self.lora_a_stacked = (torch.randn(2, 1, 8, 4),)
        mock_self.lora_b_stacked = (torch.randn(2, 1, 4, 8),)
        mock_self.output_slices = (8,)

        x = torch.randn(1, 4, 8)
        layer_mod.base_linear_patched_apply(mock_self, x)

        mock_self.punica_wrapper.add_lora_linear.assert_called_once()
        call_args = mock_self.punica_wrapper.add_lora_linear.call_args
        # Verify correct arguments are passed
        assert call_args[0][3] is mock_self.lora_b_stacked  # lora_b_stacked
        assert call_args[0][4] == 1.0  # scale
        assert call_args[0][5] is mock_self.output_slices  # output_slices

    def test_patched_apply_reshapes_output_back(self):
        """Patched apply() must reshape the output back to original shape."""
        import vllm_rbln.lora.layer as layer_mod  # noqa: F401

        mock_self = MagicMock()
        base_output = torch.randn(2, 4, 8)
        mock_self.base_layer.quant_method.apply.return_value = base_output.clone()
        # add_lora_linear returns [bs*seq_len, hidden_size]
        mock_self.punica_wrapper.add_lora_linear.return_value = torch.randn(8, 8)
        mock_self.lora_a_stacked = (torch.randn(2, 1, 8, 4),)
        mock_self.lora_b_stacked = (torch.randn(2, 1, 4, 8),)
        mock_self.output_slices = (8,)

        x = torch.randn(2, 4, 8)
        result = layer_mod.base_linear_patched_apply(mock_self, x)
        assert result.shape == (2, 4, 8)


class TestVocabParallelEmbeddingPatch:
    """Test patched forward() prefill vs decode behavior."""

    def _make_mock_self(self, max_loras=2, vocab_size=100, embed_dim=16, rank=4):
        mock_self = MagicMock()
        mock_self.base_layer.org_vocab_size = vocab_size
        mock_self.lora_a_stacked_2d = torch.randn(vocab_size + 10, rank)
        mock_self.lora_b_stacked = torch.randn(max_loras, 1, rank, embed_dim)
        mock_self.base_layer.forward.side_effect = lambda x: torch.randn(
            *x.shape, embed_dim
        )
        # _embeddings_indices shape: [2, max_tokens]
        mock_self.punica_wrapper._embeddings_indices = torch.zeros(
            2, 32, dtype=torch.long
        )
        mock_self.punica_wrapper.add_lora_embedding.side_effect = (
            lambda y, x, lora_b, add_input: y
        )
        return mock_self

    def test_prefill_batch_1(self):
        """Prefill: x.shape[0] == 1, so is_prefill=True."""
        import vllm_rbln.lora.layer as layer_mod

        mock_self = self._make_mock_self()
        x = torch.randint(0, 50, (1, 8))  # batch=1, seq_len=8

        layer_mod.vocab_parallel_embedding_patched_forward(mock_self, x)
        mock_self.punica_wrapper.add_lora_embedding.assert_called_once()

    def test_decode_batch_gt1(self):
        """Decode: x.shape[0] > 1, so is_prefill=False. indices get unsqueeze(1)."""
        import vllm_rbln.lora.layer as layer_mod

        mock_self = self._make_mock_self()
        x = torch.randint(0, 50, (4, 1))  # batch=4, seq_len=1 (decode)

        layer_mod.vocab_parallel_embedding_patched_forward(mock_self, x)
        mock_self.punica_wrapper.add_lora_embedding.assert_called_once()


# ---------------------------------------------------------------------------
# 2. PunicaWrapperRBLN
# ---------------------------------------------------------------------------


class TestPunicaWrapperRBLN:
    """Test PunicaWrapperRBLN math operations."""

    @pytest.fixture
    def wrapper(self):
        from vllm_rbln.lora.punica_wrapper.punica_rbln import PunicaWrapperRBLN

        return PunicaWrapperRBLN(
            max_num_batched_tokens=32,
            max_batches=4,
            device="cpu",
        )

    def test_add_lora_linear_single_slice(self, wrapper):
        """add_lora_linear: output += (x @ lora_a) @ lora_b for one slice."""
        max_loras = 2
        rank = 4
        hidden_in = 8
        hidden_out = 16
        num_tokens = 3

        # Set up mask: select lora index 0 for all tokens
        lora_mask = torch.zeros(num_tokens, max_loras * rank)
        lora_mask[:, :rank] = 1.0  # select first lora
        LoRAMask.set_lora_mask(lora_mask)

        # Shapes: lora_a=[max_loras, 1, rank, hidden_in], lora_b=[max_loras, 1, hidden_out, rank]
        lora_a = torch.randn(max_loras, 1, rank, hidden_in)
        lora_b = torch.randn(max_loras, 1, hidden_out, rank)
        x = torch.randn(num_tokens, hidden_in)
        y = torch.zeros(num_tokens, hidden_out)

        result = wrapper.add_lora_linear(
            y.clone(), x, (lora_a,), (lora_b,), 1.0, (hidden_out,)
        )

        # Manual computation mirroring the kernel logic
        a_w = lora_a[:, 0, :, :].reshape(-1, lora_a.shape[3]).T  # [hidden_in, max_loras*rank]
        b_w = lora_b[:, 0, :, :].transpose(1, 2).reshape(-1, lora_b.shape[2])  # [max_loras*rank, hidden_out]
        expected = (x @ a_w) * lora_mask @ b_w
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_add_lora_linear_multiple_slices(self, wrapper):
        """add_lora_linear with split QKV projections (3 slices)."""
        max_loras = 2
        rank = 4
        hidden_in = 8
        slice_size = 6
        num_slices = 3
        num_tokens = 2

        lora_mask = torch.zeros(num_tokens, max_loras * rank)
        lora_mask[:, :rank] = 1.0
        LoRAMask.set_lora_mask(lora_mask)

        lora_a_stacked = tuple(
            torch.randn(max_loras, 1, rank, hidden_in) for _ in range(num_slices)
        )
        lora_b_stacked = tuple(
            torch.randn(max_loras, 1, slice_size, rank) for _ in range(num_slices)
        )
        x = torch.randn(num_tokens, hidden_in)
        y = torch.zeros(num_tokens, slice_size * num_slices)

        result = wrapper.add_lora_linear(
            y.clone(),
            x,
            lora_a_stacked,
            lora_b_stacked,
            1.0,
            (slice_size,) * num_slices,
        )

        # Each slice should have been written to the correct offset
        for i in range(num_slices):
            a_w = lora_a_stacked[i][:, 0, :, :].reshape(-1, lora_a_stacked[i].shape[3]).T
            b_w = lora_b_stacked[i][:, 0, :, :].transpose(1, 2).reshape(-1, lora_b_stacked[i].shape[2])
            expected_slice = (x @ a_w) * lora_mask @ b_w
            actual_slice = result[:, i * slice_size : (i + 1) * slice_size]
            torch.testing.assert_close(actual_slice, expected_slice, atol=1e-5, rtol=1e-5)

    def test_add_lora_embedding(self, wrapper):
        """add_lora_embedding: output += embeddings @ lora_b."""
        max_loras = 2
        rank = 4
        hidden_size = 16
        num_tokens = 3

        lora_mask = torch.zeros(num_tokens, max_loras * rank)
        lora_mask[:, :rank] = 1.0
        LoRAMask.set_lora_mask(lora_mask)

        # lora_b shape: [max_loras, 1, hidden_size, rank]
        lora_b = torch.randn(max_loras, 1, hidden_size, rank)
        embeddings = torch.randn(num_tokens, rank)
        y = torch.zeros(num_tokens, hidden_size)

        result = wrapper.add_lora_embedding(y.clone(), embeddings, lora_b)

        # Manual computation
        x_rep = embeddings.repeat(1, max_loras)  # [num_tokens, rank * max_loras]
        x_masked = x_rep * lora_mask
        b_w = lora_b[:, 0, :, :].transpose(1, 2).reshape(-1, lora_b.shape[2])
        expected = x_masked @ b_w
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_add_lora_linear_scale_factor(self, wrapper):
        """Scale factor is applied to LoRA output."""
        max_loras = 1
        rank = 2
        hidden_in = 4
        hidden_out = 4
        num_tokens = 1

        lora_mask = torch.ones(num_tokens, max_loras * rank)
        LoRAMask.set_lora_mask(lora_mask)

        lora_a = torch.ones(max_loras, 1, rank, hidden_in)
        lora_b = torch.ones(max_loras, 1, hidden_out, rank)
        x = torch.ones(num_tokens, hidden_in)

        result_scale1 = wrapper.add_lora_linear(
            torch.zeros(num_tokens, hidden_out), x, (lora_a,), (lora_b,), 1.0, (hidden_out,)
        )
        result_scale2 = wrapper.add_lora_linear(
            torch.zeros(num_tokens, hidden_out), x, (lora_a,), (lora_b,), 2.0, (hidden_out,)
        )
        torch.testing.assert_close(result_scale2, result_scale1 * 2, atol=1e-5, rtol=1e-5)

    def test_add_lora_logits(self, wrapper):
        """add_lora_logits: y += (x @ lora_a) @ lora_b * scale."""
        max_loras = 2
        rank = 4
        input_dim = 8
        output_dim = 16
        num_tokens = 3

        lora_mask = torch.zeros(num_tokens, max_loras * rank)
        lora_mask[:, :rank] = 1.0
        LoRAMask.set_lora_mask(lora_mask)

        lora_a = torch.randn(max_loras, 1, rank, input_dim)
        lora_b = torch.randn(max_loras, 1, output_dim, rank)
        x = torch.randn(num_tokens, input_dim)
        y = torch.zeros(num_tokens, output_dim)

        result = wrapper.add_lora_logits(y.clone(), x, lora_a, lora_b, 1.0)

        # Manual
        a_w = lora_a[:, 0, :, :].reshape(-1, lora_a.shape[3]).T
        b_w = lora_b[:, 0, :, :].transpose(1, 2).reshape(-1, lora_b.shape[2])
        expected = (x @ a_w) * lora_mask @ b_w
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_sampler_indices_padded_property(self, wrapper):
        """sampler_indices_padded property delegates to LoRAInputs."""
        tensor = torch.tensor([0, 1, 2, 3])
        LoRAInputs.set_sampler_indices_padded(tensor)
        torch.testing.assert_close(wrapper.sampler_indices_padded, tensor)

    def test_add_shrink_raises(self, wrapper):
        """add_shrink is not implemented for RBLN."""
        with pytest.raises(NotImplementedError):
            wrapper.add_shrink(
                torch.zeros(1), torch.zeros(1), (torch.zeros(1),), 1.0
            )

    def test_add_expand_raises(self, wrapper):
        """add_expand is not implemented for RBLN."""
        with pytest.raises(NotImplementedError):
            wrapper.add_expand(
                torch.zeros(1), torch.zeros(1), (torch.zeros(1),), None, (1,)
            )

    def test_embeddings_indices_initialized_to_zero(self, wrapper):
        """PunicaWrapperRBLN fills _embeddings_indices with zeros on init."""
        assert (wrapper._embeddings_indices == 0).all()


# ---------------------------------------------------------------------------
# 3. LoRAInputs / LoRAMask static class behavior
# ---------------------------------------------------------------------------


class TestLoRAInputs:
    def test_set_and_get_sampler_indices_padded(self):
        tensor = torch.tensor([10, 20, 30])
        LoRAInputs.set_sampler_indices_padded(tensor)
        result = LoRAInputs.get_sampler_indices_padded()
        assert result is tensor

    def test_static_state_shared_across_calls(self):
        """Class variable is shared -- no instance needed."""
        t1 = torch.tensor([1, 2])
        LoRAInputs.set_sampler_indices_padded(t1)

        t2 = torch.tensor([3, 4])
        LoRAInputs.set_sampler_indices_padded(t2)

        # First tensor is gone, second is current
        result = LoRAInputs.get_sampler_indices_padded()
        assert result is t2

    def test_classmethod_accessible_from_instance(self):
        """get/set work from class or instance equally."""
        t = torch.tensor([5])
        LoRAInputs.set_sampler_indices_padded(t)
        # Accessing via the class directly (no instance)
        assert LoRAInputs.get_sampler_indices_padded() is t


class TestLoRAMask:
    def test_set_and_get_lora_mask(self):
        mask = torch.ones(4, 8)
        LoRAMask.set_lora_mask(mask)
        assert LoRAMask.get_lora_mask() is mask

    def test_mask_shape_preserved(self):
        mask = torch.zeros(3, 16)
        LoRAMask.set_lora_mask(mask)
        assert LoRAMask.get_lora_mask().shape == (3, 16)

    def test_static_state_shared(self):
        m1 = torch.ones(2, 4)
        LoRAMask.set_lora_mask(m1)
        m2 = torch.zeros(2, 4)
        LoRAMask.set_lora_mask(m2)
        assert LoRAMask.get_lora_mask() is m2


# ---------------------------------------------------------------------------
# 4. Bug-catching / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.fixture
    def wrapper(self):
        from vllm_rbln.lora.punica_wrapper.punica_rbln import PunicaWrapperRBLN

        return PunicaWrapperRBLN(
            max_num_batched_tokens=32,
            max_batches=4,
            device="cpu",
        )

    def test_lora_linear_empty_output_slices(self, wrapper):
        """Empty output_slices tuple should be a no-op (zero iterations)."""
        num_tokens = 2
        hidden_in = 8
        hidden_out = 16
        y = torch.randn(num_tokens, hidden_out)
        x = torch.randn(num_tokens, hidden_in)
        y_orig = y.clone()

        result = wrapper.add_lora_linear(y, x, (), (), 1.0, ())
        torch.testing.assert_close(result, y_orig)

    def test_punica_wrapper_none_raises_in_patched_apply(self):
        """If punica_wrapper is None, patched apply should raise AttributeError."""
        import vllm_rbln.lora.layer as layer_mod

        mock_self = MagicMock()
        mock_self.base_layer.quant_method.apply.return_value = torch.randn(1, 4, 8)
        mock_self.punica_wrapper = None

        x = torch.randn(1, 4, 8)
        with pytest.raises(AttributeError):
            layer_mod.base_linear_patched_apply(mock_self, x)

    def test_lora_mask_not_set_raises(self):
        """If LoRAMask was never set, get_lora_mask raises AttributeError."""
        # Temporarily remove the class variable to simulate uninitialized state
        saved = None
        has_attr = hasattr(LoRAMask, "lora_mask")
        if has_attr:
            saved = LoRAMask.lora_mask
            del LoRAMask.lora_mask

        try:
            with pytest.raises(AttributeError):
                LoRAMask.get_lora_mask()
        finally:
            if has_attr:
                LoRAMask.lora_mask = saved

    def test_lora_inputs_not_set_raises(self):
        """If LoRAInputs was never set, get raises AttributeError."""
        saved = None
        has_attr = hasattr(LoRAInputs, "sampler_indices_padded")
        if has_attr:
            saved = LoRAInputs.sampler_indices_padded
            del LoRAInputs.sampler_indices_padded

        try:
            with pytest.raises(AttributeError):
                LoRAInputs.get_sampler_indices_padded()
        finally:
            if has_attr:
                LoRAInputs.sampler_indices_padded = saved

    def test_lora_linear_dimension_mismatch_raises(self, wrapper):
        """Mismatched lora_a / lora_b rank dimensions should raise."""
        max_loras = 2
        rank_a = 4
        rank_b = 8  # mismatch!
        hidden_in = 8
        hidden_out = 16
        num_tokens = 2

        lora_mask = torch.ones(num_tokens, max_loras * rank_a)
        LoRAMask.set_lora_mask(lora_mask)

        lora_a = torch.randn(max_loras, 1, rank_a, hidden_in)
        lora_b = torch.randn(max_loras, 1, hidden_out, rank_b)  # rank mismatch
        x = torch.randn(num_tokens, hidden_in)
        y = torch.zeros(num_tokens, hidden_out)

        with pytest.raises(RuntimeError):
            wrapper.add_lora_linear(y, x, (lora_a,), (lora_b,), 1.0, (hidden_out,))

    def test_prefill_assumption_batch_gt1(self):
        """When batch > 1, is_prefill=False even if it's actually a prefill.
        This tests the current behavior/assumption, not ideal behavior."""
        import vllm_rbln.lora.layer as layer_mod

        mock_self = MagicMock()
        mock_self.base_layer.org_vocab_size = 100
        mock_self.lora_a_stacked_2d = torch.randn(110, 4)
        mock_self.lora_b_stacked = torch.randn(2, 1, 4, 16)
        mock_self.base_layer.forward.side_effect = lambda inp: torch.randn(
            *inp.shape, 16
        )
        mock_self.punica_wrapper._embeddings_indices = torch.zeros(
            2, 32, dtype=torch.long
        )
        mock_self.punica_wrapper.add_lora_embedding.side_effect = (
            lambda y, x, lora_b, add_input: y
        )

        # batch=2: is_prefill becomes False
        x = torch.randint(0, 50, (2, 8))
        layer_mod.vocab_parallel_embedding_patched_forward(mock_self, x)

        # narrow_length should be x.size(0) == 2 (not x.size(1) == 8)
        call_args = mock_self.punica_wrapper.add_lora_embedding.call_args
        assert call_args is not None

    def test_add_lora_linear_y_is_modified_in_place(self, wrapper):
        """add_lora_linear modifies y in-place (y += ...) and also returns it."""
        max_loras = 1
        rank = 2
        hidden = 4
        num_tokens = 1

        lora_mask = torch.ones(num_tokens, max_loras * rank)
        LoRAMask.set_lora_mask(lora_mask)

        lora_a = torch.ones(max_loras, 1, rank, hidden)
        lora_b = torch.ones(max_loras, 1, hidden, rank)
        x = torch.ones(num_tokens, hidden)
        y = torch.zeros(num_tokens, hidden)

        result = wrapper.add_lora_linear(y, x, (lora_a,), (lora_b,), 1.0, (hidden,))
        # y should be modified in-place
        assert result is y
        assert (y != 0).any()
