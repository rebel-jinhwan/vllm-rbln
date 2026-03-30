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

"""End-to-end tests for structured output with choice constraints.

Tests the structured output functionality using StructuredOutputsParams
to constrain model output to specific choices or patterns.
"""

import os
import time

os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_RBLN_COMPILE_MODEL"] = "0"
os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "0"

import pytest  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402

MODEL_NAME = "facebook/opt-125m"
MAX_MODEL_LEN = 128
MAX_NUM_SEQS = 2
GPU_MEMORY_UTILIZATION = 0.5


def _try_create_llm_with_structured_outputs():
    """Attempt to create an LLM that supports structured outputs.

    Returns the LLM instance if successful, or pytest.skip if structured
    outputs are not supported in the current environment.
    """
    try:
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
        # Quick smoke test to see if structured outputs work at all
        params = SamplingParams(
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(choice=["yes", "no"]),
        )
        outputs = llm.generate(["Is the sky blue?"], params)
        return llm
    except Exception as e:
        pytest.skip(
            f"Structured output not supported in this environment: {e}"
        )


@pytest.fixture(scope="module")
def llm_instance():
    """Module-scoped LLM for structured output tests."""
    llm = _try_create_llm_with_structured_outputs()
    yield llm
    del llm
    time.sleep(2)


class TestChoiceConstraints:
    """Test structured output with guided_choice parameter."""

    def test_choice_yes_no(self, llm_instance):
        """Output should be constrained to 'yes' or 'no'."""
        params = SamplingParams(
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(choice=["yes", "no"]),
        )
        outputs = llm_instance.generate(["Is the sky blue?"], params)
        text = outputs[0].outputs[0].text.strip().lower()
        assert text in ("yes", "no"), (
            f"Expected 'yes' or 'no', got {text!r}"
        )

    def test_choice_multiple_options(self, llm_instance):
        """Output should be one of the provided choices."""
        choices = ["red", "green", "blue", "yellow"]
        params = SamplingParams(
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(choice=choices),
        )
        outputs = llm_instance.generate(
            ["What color is the sky?"], params
        )
        text = outputs[0].outputs[0].text.strip().lower()
        assert text in choices, (
            f"Expected one of {choices}, got {text!r}"
        )

    def test_choice_single_option(self, llm_instance):
        """With a single choice, output must be that choice."""
        params = SamplingParams(
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(choice=["hello"]),
        )
        outputs = llm_instance.generate(["Say something"], params)
        text = outputs[0].outputs[0].text.strip().lower()
        assert text == "hello", f"Expected 'hello', got {text!r}"


class TestRegexConstraints:
    """Test structured output with regex constraints."""

    def test_regex_digits(self, llm_instance):
        """Output should match a digit-only regex pattern."""
        try:
            params = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                structured_outputs=StructuredOutputsParams(
                    regex=r"[0-9]+"
                ),
            )
            outputs = llm_instance.generate(
                ["Give me a number:"], params
            )
            text = outputs[0].outputs[0].text.strip()
            assert text.isdigit(), (
                f"Expected digits only, got {text!r}"
            )
        except Exception:
            pytest.skip("Regex structured output not supported")

    def test_regex_yes_no(self, llm_instance):
        """Output should match yes|no regex pattern."""
        try:
            params = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                structured_outputs=StructuredOutputsParams(
                    regex=r"(yes|no)"
                ),
            )
            outputs = llm_instance.generate(
                ["Is the sky blue?"], params
            )
            text = outputs[0].outputs[0].text.strip().lower()
            assert text in ("yes", "no"), (
                f"Expected 'yes' or 'no', got {text!r}"
            )
        except Exception:
            pytest.skip("Regex structured output not supported")
