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

"""End-to-end tests for basic LLM inference using vllm.LLM on CPU.

These tests exercise the full inference pipeline including:
RBLNWorker.__init__, init_device, load_model, RBLNModelRunner.__init__,
load_model, _prepare_inputs, execute_model, sample_tokens,
RBLNScheduler.schedule, attention backend, and sampler.
"""

import os
import time

os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_RBLN_COMPILE_MODEL"] = "0"
os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "0"

import pytest  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL_NAME = "facebook/opt-125m"
MAX_MODEL_LEN = 128
MAX_NUM_SEQS = 2
GPU_MEMORY_UTILIZATION = 0.5


@pytest.fixture(scope="module")
def llm_instance():
    """Module-scoped LLM fixture to avoid reloading model per test."""
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    yield llm
    del llm
    time.sleep(2)


class TestGreedyDecoding:
    """Test greedy decoding produces deterministic output."""

    def test_greedy_decoding_deterministic(self, llm_instance):
        """Running greedy decoding twice should produce identical output."""
        params = SamplingParams(max_tokens=10, temperature=0.0)
        prompt = "The capital of France is"

        outputs_1 = llm_instance.generate([prompt], params)
        outputs_2 = llm_instance.generate([prompt], params)

        text_1 = outputs_1[0].outputs[0].text
        text_2 = outputs_2[0].outputs[0].text

        assert len(text_1) > 0, "Greedy decoding produced empty output"
        assert text_1 == text_2, (
            f"Greedy decoding not deterministic: {text_1!r} vs {text_2!r}"
        )

    def test_greedy_decoding_nonempty(self, llm_instance):
        """Greedy decoding should produce non-empty text."""
        params = SamplingParams(max_tokens=10, temperature=0.0)
        outputs = llm_instance.generate(["Hello, world"], params)
        text = outputs[0].outputs[0].text
        assert len(text) > 0, "Expected non-empty output from greedy decoding"


class TestTemperature:
    """Test inference with different temperature settings."""

    def test_temperature_zero(self, llm_instance):
        """Temperature=0 should produce deterministic output."""
        params = SamplingParams(max_tokens=10, temperature=0.0)
        outputs = llm_instance.generate(["Once upon a time"], params)
        assert len(outputs[0].outputs[0].text) > 0

    def test_temperature_moderate(self, llm_instance):
        """Temperature=0.5 should produce valid output."""
        params = SamplingParams(max_tokens=10, temperature=0.5, seed=42)
        outputs = llm_instance.generate(["Once upon a time"], params)
        assert len(outputs[0].outputs[0].text) > 0

    def test_temperature_high(self, llm_instance):
        """Temperature=1.0 should produce valid output."""
        params = SamplingParams(max_tokens=10, temperature=1.0, seed=42)
        outputs = llm_instance.generate(["Once upon a time"], params)
        assert len(outputs[0].outputs[0].text) > 0


class TestSamplingMethods:
    """Test top_k and top_p sampling."""

    def test_top_k_sampling(self, llm_instance):
        """Top-k sampling should produce valid output."""
        params = SamplingParams(
            max_tokens=10, temperature=0.8, top_k=10, seed=42
        )
        outputs = llm_instance.generate(["The weather today is"], params)
        assert len(outputs[0].outputs[0].text) > 0

    def test_top_p_sampling(self, llm_instance):
        """Top-p (nucleus) sampling should produce valid output."""
        params = SamplingParams(
            max_tokens=10, temperature=0.8, top_p=0.9, seed=42
        )
        outputs = llm_instance.generate(["The weather today is"], params)
        assert len(outputs[0].outputs[0].text) > 0

    def test_top_k_and_top_p_combined(self, llm_instance):
        """Combined top_k and top_p sampling should produce valid output."""
        params = SamplingParams(
            max_tokens=10, temperature=0.8, top_k=50, top_p=0.95, seed=42
        )
        outputs = llm_instance.generate(["The weather today is"], params)
        assert len(outputs[0].outputs[0].text) > 0


class TestBatchInference:
    """Test batch inference with multiple prompts."""

    def test_multiple_prompts(self, llm_instance):
        """Batch inference should handle multiple prompts."""
        prompts = [
            "Hello, world",
            "The capital of France is",
        ]
        params = SamplingParams(max_tokens=10, temperature=0.0)
        outputs = llm_instance.generate(prompts, params)

        assert len(outputs) == len(prompts), (
            f"Expected {len(prompts)} outputs, got {len(outputs)}"
        )
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            assert len(text) > 0, f"Prompt {i} produced empty output"

    def test_single_prompt_batch(self, llm_instance):
        """Single prompt in a batch should work."""
        params = SamplingParams(max_tokens=5, temperature=0.0)
        outputs = llm_instance.generate(["Test prompt"], params)
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].text) > 0


class TestMaxTokens:
    """Test that max_tokens limit is respected."""

    def test_max_tokens_limit(self, llm_instance):
        """Output should not exceed max_tokens."""
        for max_tokens in [1, 3, 5, 10]:
            params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
            outputs = llm_instance.generate(["Hello"], params)
            n_tokens = len(outputs[0].outputs[0].token_ids)
            assert n_tokens <= max_tokens, (
                f"Expected at most {max_tokens} tokens, got {n_tokens}"
            )

    def test_max_tokens_one(self, llm_instance):
        """max_tokens=1 should produce exactly 1 token."""
        params = SamplingParams(max_tokens=1, temperature=0.0)
        outputs = llm_instance.generate(["Hello"], params)
        n_tokens = len(outputs[0].outputs[0].token_ids)
        assert n_tokens == 1, f"Expected 1 token, got {n_tokens}"


class TestPromptLengths:
    """Test with various prompt lengths."""

    def test_short_prompt(self, llm_instance):
        """Very short prompt should work."""
        params = SamplingParams(max_tokens=5, temperature=0.0)
        outputs = llm_instance.generate(["Hi"], params)
        assert len(outputs[0].outputs[0].text) > 0

    def test_medium_prompt(self, llm_instance):
        """Medium-length prompt should work."""
        prompt = (
            "In a land far away, there lived a young wizard who spent "
            "most of his days reading ancient scrolls and practicing "
            "spells in the tower of knowledge."
        )
        params = SamplingParams(max_tokens=10, temperature=0.0)
        outputs = llm_instance.generate([prompt], params)
        assert len(outputs[0].outputs[0].text) > 0


class TestCompileModelError:
    """Test that COMPILE_MODEL=1 fails gracefully without RBLN hardware."""

    def test_compile_model_raises_error(self):
        """With COMPILE_MODEL=1, torch.compile(backend='rbln') should fail
        with BackendCompilerFailed since there is no RBLN hardware."""
        from torch._dynamo.exc import BackendCompilerFailed

        env_backup = os.environ.get("VLLM_RBLN_COMPILE_MODEL")
        warmup_backup = os.environ.get("VLLM_RBLN_ENABLE_WARM_UP")
        try:
            os.environ["VLLM_RBLN_COMPILE_MODEL"] = "1"
            os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "1"

            with pytest.raises((BackendCompilerFailed, RuntimeError)):
                llm = LLM(
                    model=MODEL_NAME,
                    max_model_len=MAX_MODEL_LEN,
                    max_num_seqs=MAX_NUM_SEQS,
                    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                )
                llm.generate(
                    ["test"], SamplingParams(max_tokens=1, temperature=0.0)
                )
                del llm
        finally:
            if env_backup is not None:
                os.environ["VLLM_RBLN_COMPILE_MODEL"] = env_backup
            else:
                os.environ.pop("VLLM_RBLN_COMPILE_MODEL", None)
            if warmup_backup is not None:
                os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = warmup_backup
            else:
                os.environ.pop("VLLM_RBLN_ENABLE_WARM_UP", None)
            time.sleep(2)
