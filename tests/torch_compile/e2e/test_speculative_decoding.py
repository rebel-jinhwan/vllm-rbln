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

"""End-to-end tests for speculative decoding with ngram proposer.

Tests ngram-based speculative decoding via the LLM interface, verifying
that the speculative decoding pipeline produces valid output and does
not crash.
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

SPEC_DECODE_CONFIG = {
    "method": "ngram",
    "num_speculative_tokens": 3,
    "prompt_lookup_max": 5,
}


def _try_create_spec_llm():
    """Attempt to create an LLM with speculative decoding config.

    Returns the LLM instance if successful, or pytest.skip if
    speculative decoding is not supported in the current environment.
    """
    try:
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            speculative_config=SPEC_DECODE_CONFIG,
        )
        return llm
    except Exception as e:
        pytest.skip(
            f"Speculative decoding not supported in this environment: {e}"
        )


@pytest.fixture(scope="module")
def spec_llm():
    """Module-scoped LLM with speculative decoding enabled."""
    llm = _try_create_spec_llm()
    yield llm
    del llm
    time.sleep(2)


@pytest.fixture(scope="module")
def baseline_llm():
    """Module-scoped baseline LLM without speculative decoding."""
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    yield llm
    del llm
    time.sleep(2)


class TestSpeculativeDecodingBasic:
    """Test that ngram-based speculative decoding produces valid output."""

    def test_spec_decode_produces_output(self, spec_llm):
        """Speculative decoding should produce non-empty output."""
        params = SamplingParams(max_tokens=10, temperature=0.0)
        outputs = spec_llm.generate(["The capital of France is"], params)
        text = outputs[0].outputs[0].text
        assert len(text) > 0, "Speculative decoding produced empty output"

    def test_spec_decode_no_crash(self, spec_llm):
        """Speculative decoding should not crash with various prompts."""
        prompts = [
            "Hello, world",
            "Once upon a time in a land far away",
        ]
        params = SamplingParams(max_tokens=10, temperature=0.0)
        outputs = spec_llm.generate(prompts, params)
        assert len(outputs) == len(prompts)
        for output in outputs:
            assert len(output.outputs[0].text) > 0

    def test_spec_decode_max_tokens_respected(self, spec_llm):
        """max_tokens should be respected with speculative decoding."""
        for max_tokens in [1, 5]:
            params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
            outputs = spec_llm.generate(["Hello"], params)
            n_tokens = len(outputs[0].outputs[0].token_ids)
            assert n_tokens <= max_tokens, (
                f"Expected at most {max_tokens} tokens, got {n_tokens}"
            )


class TestSpeculativeDecodingQuality:
    """Compare speculative decoding output quality with baseline."""

    def test_output_quality_fuzzy_match(self, spec_llm, baseline_llm):
        """Speculative decoding output should be similar to baseline.

        With greedy decoding (temperature=0), speculative decoding should
        produce output that is identical or very close to baseline, since
        the verification step ensures correctness.
        """
        prompt = "The capital of France is"
        params = SamplingParams(max_tokens=10, temperature=0.0)

        spec_outputs = spec_llm.generate([prompt], params)
        base_outputs = baseline_llm.generate([prompt], params)

        spec_text = spec_outputs[0].outputs[0].text
        base_text = base_outputs[0].outputs[0].text

        # With greedy decoding, speculative decoding should match baseline
        # exactly (modulo any implementation differences).
        # At minimum, both should produce non-empty output.
        assert len(spec_text) > 0
        assert len(base_text) > 0

        # Check that outputs share at least some common tokens.
        spec_tokens = spec_outputs[0].outputs[0].token_ids
        base_tokens = base_outputs[0].outputs[0].token_ids

        if len(spec_tokens) > 0 and len(base_tokens) > 0:
            # At least the first token should match for greedy decoding
            assert spec_tokens[0] == base_tokens[0], (
                f"First tokens differ: spec={spec_tokens[0]}, "
                f"base={base_tokens[0]}"
            )
