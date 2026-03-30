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

"""End-to-end tests for logprobs functionality.

Tests that logprobs and prompt_logprobs are correctly returned and have
valid values (negative log probabilities, correct shapes, etc.).
"""

import math
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
    """Module-scoped LLM fixture for logprobs tests."""
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    yield llm
    del llm
    time.sleep(2)


class TestLogprobs:
    """Test logprobs=5 returns top-5 logprobs per token."""

    def test_logprobs_returned(self, llm_instance):
        """logprobs=5 should return logprobs in the output."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, logprobs=5
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        output = outputs[0].outputs[0]

        assert output.logprobs is not None, "logprobs should not be None"
        assert len(output.logprobs) > 0, "logprobs should not be empty"

    def test_logprobs_count(self, llm_instance):
        """Each token position should have up to 5+1 logprobs entries."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, logprobs=5
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        output = outputs[0].outputs[0]

        for token_logprobs in output.logprobs:
            # logprobs=5 means up to 5 top logprobs + the sampled token
            # so we may get up to 6 entries
            assert len(token_logprobs) <= 6, (
                f"Expected at most 6 logprob entries, got {len(token_logprobs)}"
            )
            assert len(token_logprobs) >= 1, (
                "Expected at least 1 logprob entry per token"
            )

    def test_logprobs_one(self, llm_instance):
        """logprobs=1 should return at least 1 logprob per token."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, logprobs=1
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        output = outputs[0].outputs[0]

        assert output.logprobs is not None
        for token_logprobs in output.logprobs:
            assert len(token_logprobs) >= 1


class TestLogprobValues:
    """Test that logprob values are valid."""

    def test_logprobs_are_negative(self, llm_instance):
        """Log probabilities should be negative (or zero for prob=1)."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, logprobs=5
        )
        outputs = llm_instance.generate(["The capital of France is"], params)
        output = outputs[0].outputs[0]

        for token_logprobs in output.logprobs:
            for token_id, logprob_obj in token_logprobs.items():
                lp = logprob_obj.logprob
                assert lp <= 0.0, (
                    f"Logprob should be <= 0, got {lp} for token {token_id}"
                )

    def test_logprobs_finite(self, llm_instance):
        """Log probabilities should be finite numbers."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, logprobs=5
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        output = outputs[0].outputs[0]

        for token_logprobs in output.logprobs:
            for token_id, logprob_obj in token_logprobs.items():
                lp = logprob_obj.logprob
                assert math.isfinite(lp), (
                    f"Logprob should be finite, got {lp} for token {token_id}"
                )

    def test_logprobs_approximate_sum(self, llm_instance):
        """Exponentiated logprobs should sum to approximately <= 1.

        Since we only get the top-k logprobs, the sum of exp(logprob)
        should be <= 1 (as probabilities sum to 1 over the full vocab).
        """
        params = SamplingParams(
            max_tokens=3, temperature=0.0, logprobs=5
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        output = outputs[0].outputs[0]

        for token_logprobs in output.logprobs:
            prob_sum = sum(
                math.exp(lp_obj.logprob)
                for lp_obj in token_logprobs.values()
            )
            # Sum of top-k probabilities should be <= 1 (with small tolerance)
            assert prob_sum <= 1.0 + 1e-5, (
                f"Sum of top-k probabilities exceeds 1: {prob_sum}"
            )


class TestLogprobTokenIds:
    """Verify logprob token IDs match vocabulary."""

    def test_logprob_token_ids_are_valid(self, llm_instance):
        """Token IDs in logprobs should be non-negative integers."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, logprobs=5
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        output = outputs[0].outputs[0]

        for token_logprobs in output.logprobs:
            for token_id in token_logprobs:
                assert isinstance(token_id, int), (
                    f"Token ID should be int, got {type(token_id)}"
                )
                assert token_id >= 0, (
                    f"Token ID should be non-negative, got {token_id}"
                )

    def test_sampled_token_in_logprobs(self, llm_instance):
        """The actually sampled token should appear in the logprobs."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, logprobs=5
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        output = outputs[0].outputs[0]

        sampled_tokens = output.token_ids
        assert output.logprobs is not None
        assert len(output.logprobs) == len(sampled_tokens)

        for i, (token_id, token_logprobs) in enumerate(
            zip(sampled_tokens, output.logprobs)
        ):
            assert token_id in token_logprobs, (
                f"Sampled token {token_id} at position {i} not found "
                f"in logprobs keys: {list(token_logprobs.keys())}"
            )


@pytest.mark.skip(
    reason="Known bug: _get_prompt_logprobs_dict raises RuntimeError "
    "due to tensor dimension mismatch in sampler.gather_logprobs"
)
class TestPromptLogprobs:
    """Test prompt_logprobs returns logprobs for prompt tokens.

    NOTE: These tests are skipped because prompt_logprobs triggers a known
    bug in RBLNModelRunner._get_prompt_logprobs_dict where
    sampler.gather_logprobs fails with 'Index tensor must have the same
    number of dimensions as input tensor'. This crashes the EngineCore
    process and kills the entire LLM instance.
    """

    def test_prompt_logprobs_returned(self, llm_instance):
        """prompt_logprobs should return logprobs for prompt tokens."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, prompt_logprobs=5
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        output = outputs[0]

        assert output.prompt_logprobs is not None, (
            "prompt_logprobs should not be None"
        )

    def test_prompt_logprobs_length(self, llm_instance):
        """prompt_logprobs length should match prompt token count."""
        prompt = "Hello, world"
        params = SamplingParams(
            max_tokens=5, temperature=0.0, prompt_logprobs=5
        )
        outputs = llm_instance.generate([prompt], params)
        output = outputs[0]

        prompt_lps = output.prompt_logprobs
        assert prompt_lps is not None

        # The number of prompt logprob entries should match the number
        # of prompt tokens. The first token has no logprob (None entry).
        prompt_token_count = len(output.prompt_token_ids)
        assert len(prompt_lps) == prompt_token_count, (
            f"Expected {prompt_token_count} prompt logprob entries, "
            f"got {len(prompt_lps)}"
        )

    def test_prompt_logprobs_first_token_none(self, llm_instance):
        """First prompt token should have None logprobs (no prior context)."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, prompt_logprobs=5
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        prompt_lps = outputs[0].prompt_logprobs

        assert prompt_lps is not None
        # First entry is typically None since there is no prior token
        assert prompt_lps[0] is None, (
            f"Expected None for first prompt logprob, got {prompt_lps[0]}"
        )

    def test_prompt_logprobs_values_valid(self, llm_instance):
        """Non-None prompt logprobs should be valid negative values."""
        params = SamplingParams(
            max_tokens=5, temperature=0.0, prompt_logprobs=5
        )
        outputs = llm_instance.generate(["Hello, world"], params)
        prompt_lps = outputs[0].prompt_logprobs

        assert prompt_lps is not None
        for i, token_logprobs in enumerate(prompt_lps):
            if token_logprobs is None:
                continue
            for token_id, logprob_obj in token_logprobs.items():
                lp = logprob_obj.logprob
                assert lp <= 0.0, (
                    f"Prompt logprob at position {i} should be <= 0, "
                    f"got {lp}"
                )
                assert math.isfinite(lp), (
                    f"Prompt logprob at position {i} should be finite, "
                    f"got {lp}"
                )
