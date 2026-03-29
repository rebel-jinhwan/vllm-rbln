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

"""Unit tests for lora/layer.py and lora/punica_wrapper/punica_rbln.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_rbln.lora.inputs import LoRAInputs
from vllm_rbln.lora.mask import LoRAMask


# ===========================================================================
# Tests: base_linear_patched_apply
# ===========================================================================


class TestBaseLinearPatchedApply:
    def test_basic_forward(self):
        from vllm_rbln.lora.layer import base_linear_patched_apply

        batch, seq_len, hidden = 2, 4, 8
        out_features = 16

        x = torch.randn(batch, seq_len, hidden)
        base_output = torch.randn(batch, seq_len, out_features)
        lora_output = torch.randn(batch * seq_len, out_features)

        mock_self = MagicMock()
        mock_self.base_layer.quant_method.apply.return_value = base_output.clone()
        mock_self.punica_wrapper.add_lora_linear.return_value = lora_output
        mock_self.lora_a_stacked = MagicMock()
        mock_self.lora_b_stacked = MagicMock()
        mock_self.output_slices = (out_features,)

        result = base_linear_patched_apply(mock_self, x)
        assert result.shape == (batch, seq_len, out_features)
        mock_self.base_layer.quant_method.apply.assert_called_once()
        mock_self.punica_wrapper.add_lora_linear.assert_called_once()

    def test_with_bias(self):
        from vllm_rbln.lora.layer import base_linear_patched_apply

        x = torch.randn(1, 3, 8)
        base_output = torch.randn(1, 3, 16)
        lora_output = torch.randn(3, 16)
        bias = torch.randn(16)

        mock_self = MagicMock()
        mock_self.base_layer.quant_method.apply.return_value = base_output.clone()
        mock_self.punica_wrapper.add_lora_linear.return_value = lora_output
        mock_self.lora_a_stacked = MagicMock()
        mock_self.lora_b_stacked = MagicMock()
        mock_self.output_slices = (16,)

        result = base_linear_patched_apply(mock_self, x, bias)
        assert result.shape == (1, 3, 16)
        mock_self.base_layer.quant_method.apply.assert_called_once_with(
            mock_self.base_layer, x, bias
        )

    def test_2d_input(self):
        from vllm_rbln.lora.layer import base_linear_patched_apply

        x = torch.randn(6, 8)
        base_output = torch.randn(6, 16)
        lora_output = torch.randn(6, 16)

        mock_self = MagicMock()
        mock_self.base_layer.quant_method.apply.return_value = base_output.clone()
        mock_self.punica_wrapper.add_lora_linear.return_value = lora_output
        mock_self.lora_a_stacked = MagicMock()
        mock_self.lora_b_stacked = MagicMock()
        mock_self.output_slices = (16,)

        result = base_linear_patched_apply(mock_self, x)
        assert result.shape == (6, 16)


# ===========================================================================
# Tests: vocab_parallel_embedding_patched_forward
# ===========================================================================


class TestVocabParallelEmbeddingPatchedForward:
    def _make_mock_self(self, vocab_size=1000, embed_dim=8, max_loras=2, rank=4):
        mock_self = MagicMock()
        mock_self.base_layer.org_vocab_size = vocab_size
        mock_self.lora_a_stacked_2d = torch.randn(vocab_size + 100, rank)
        mock_self.lora_b_stacked = torch.randn(max_loras, 1, rank, embed_dim)
        return mock_self

    def test_prefill(self):
        from vllm_rbln.lora.layer import vocab_parallel_embedding_patched_forward

        mock_self = self._make_mock_self()
        seq_len = 5
        x = torch.randint(0, 100, (1, seq_len))

        embeddings_indices = torch.zeros(2, seq_len, dtype=torch.long)
        mock_self.punica_wrapper._embeddings_indices = embeddings_indices
        mock_self.base_layer.forward.return_value = torch.randn(1, seq_len, 8)
        mock_self.punica_wrapper.add_lora_embedding.return_value = torch.randn(
            seq_len, 8
        )

        result = vocab_parallel_embedding_patched_forward(mock_self, x)
        assert result.shape == (1, seq_len, 8)
        mock_self.punica_wrapper.add_lora_embedding.assert_called_once()

    def test_decode(self):
        from vllm_rbln.lora.layer import vocab_parallel_embedding_patched_forward

        mock_self = self._make_mock_self()
        batch_size = 3
        x = torch.randint(0, 100, (batch_size,))

        embeddings_indices = torch.zeros(2, batch_size, dtype=torch.long)
        mock_self.punica_wrapper._embeddings_indices = embeddings_indices
        mock_self.base_layer.forward.return_value = torch.randn(batch_size, 8)
        mock_self.punica_wrapper.add_lora_embedding.return_value = torch.randn(
            batch_size, 8
        )

        result = vocab_parallel_embedding_patched_forward(mock_self, x)
        assert result.shape == (batch_size, 8)


# ===========================================================================
# Tests: PunicaWrapperRBLN
# ===========================================================================


class TestPunicaWrapperRBLN:
    @pytest.fixture
    def wrapper(self):
        from vllm_rbln.lora.punica_wrapper.punica_rbln import PunicaWrapperRBLN

        return PunicaWrapperRBLN(
            max_num_batched_tokens=256, max_batches=8, device="cpu"
        )

    def test_init(self, wrapper):
        assert wrapper is not None

    def test_add_shrink_not_implemented(self, wrapper):
        with pytest.raises(NotImplementedError):
            wrapper.add_shrink(
                y=torch.zeros(4, 8),
                x=torch.randn(4, 16),
                lora_a_stacked=(torch.randn(2, 1, 16, 8),),
                scale=1.0,
            )

    def test_add_expand_not_implemented(self, wrapper):
        with pytest.raises(NotImplementedError):
            wrapper.add_expand(
                y=torch.zeros(4, 8),
                x=(torch.randn(4, 8),),
                lora_b_stacked=(torch.randn(2, 1, 8, 8),),
                lora_bias_stacked=None,
                output_slices=(8,),
            )

    def test_add_lora_embedding(self, wrapper):
        # lora_b_stacked: [max_loras, 1, rank, hidden]
        # After processing: lora_b_w = [max_loras * hidden, rank]
        # x after repeat: [n, rank * max_loras]
        # So we need rank * max_loras == max_loras * hidden → rank == hidden
        max_loras = 2
        rank = 8
        hidden = 8  # must equal rank for matmul to work
        num_tokens = 6

        y = torch.zeros(num_tokens, hidden)
        x = torch.randn(num_tokens, rank)
        lora_b = torch.randn(max_loras, 1, rank, hidden)
        lora_mask = torch.ones(num_tokens, max_loras * rank)
        LoRAMask.set_lora_mask(lora_mask)

        result = wrapper.add_lora_embedding(y, x, lora_b, add_input=True)
        assert result.shape == (num_tokens, hidden)

    def test_add_lora_linear(self, wrapper):
        # lora_a: [max_loras, 1, h_in, rank] → reshape to [max_loras*h_in, rank]
        #   → transpose to [rank, max_loras*h_in]
        # x @ lora_a_w needs x.shape[1] == rank → h_in must work as input dim
        # Actually: lora_a_w.shape = [rank, max_loras*h_in] so x.shape[1] must = rank
        # But x.shape[1] = h_in. So we need h_in == rank.
        max_loras = 2
        rank = 8
        h_in = 8  # must equal rank for x @ lora_a_w
        h_out = 8  # must equal rank for out @ lora_b_w
        num_tokens = 6

        y = torch.zeros(num_tokens, h_out)
        x = torch.randn(num_tokens, h_in)
        lora_a = (torch.randn(max_loras, 1, h_in, rank),)
        lora_b = (torch.randn(max_loras, 1, rank, h_out),)
        lora_mask = torch.ones(num_tokens, max_loras * rank)
        LoRAMask.set_lora_mask(lora_mask)

        result = wrapper.add_lora_linear(
            y, x, lora_a, lora_b, scale=1.0, output_slices=(h_out,)
        )
        assert result.shape == (num_tokens, h_out)

    def test_add_lora_linear_multi_slice(self, wrapper):
        max_loras = 2
        rank = 8
        h_in = 8
        h_out1, h_out2 = 8, 8
        num_tokens = 3

        y = torch.zeros(num_tokens, h_out1 + h_out2)
        x = torch.randn(num_tokens, h_in)
        lora_a = (
            torch.randn(max_loras, 1, h_in, rank),
            torch.randn(max_loras, 1, h_in, rank),
        )
        lora_b = (
            torch.randn(max_loras, 1, rank, h_out1),
            torch.randn(max_loras, 1, rank, h_out2),
        )
        lora_mask = torch.ones(num_tokens, max_loras * rank)
        LoRAMask.set_lora_mask(lora_mask)

        result = wrapper.add_lora_linear(
            y, x, lora_a, lora_b, scale=0.5, output_slices=(h_out1, h_out2)
        )
        assert result.shape == (num_tokens, h_out1 + h_out2)

    def test_add_lora_logits(self, wrapper):
        max_loras = 2
        rank = 8
        h_in = 8  # must equal rank
        vocab = 8  # must equal rank
        num_tokens = 3

        y = torch.zeros(num_tokens, vocab)
        x = torch.randn(num_tokens, h_in)
        lora_a = torch.randn(max_loras, 1, h_in, rank)
        lora_b = torch.randn(max_loras, 1, rank, vocab)
        lora_mask = torch.ones(num_tokens, max_loras * rank)
        LoRAMask.set_lora_mask(lora_mask)

        result = wrapper.add_lora_logits(y, x, lora_a, lora_b, scale=1.0)
        assert result.shape == (num_tokens, vocab)

    def test_sampler_indices_padded_property(self, wrapper):
        expected = torch.tensor([0, 1, 2])
        LoRAInputs.set_sampler_indices_padded(expected)
        result = wrapper.sampler_indices_padded
        assert torch.equal(result, expected)
