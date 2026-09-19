# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RBLNModelRunnerV2 against RBLNModelRunner on a batched decode, plus the
torch sampler paths (seeded random sampling, penalties) it runs on the device."""

import pytest
import torch
from vllm import SamplingParams

from tests.vllm.runners import DPRequest
from tests.vllm.utils import check_logprobs_close, rbln_device_count

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "The capital of France is",
    "Write a short poem about the sea. " * 8,
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,",
]
ENGINE_KWARGS = dict(max_num_seqs=4, num_gpu_blocks_override=8)
MAX_TOKENS = 8


@pytest.mark.model_compile
def test_batched_decode_matches_v1(vllm_runner, monkeypatch) -> None:
    with vllm_runner(MODEL, **ENGINE_KWARGS) as v1:
        v1_outputs = v1.generate_greedy_logprobs(PROMPTS, MAX_TOKENS, 5)

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(MODEL, **ENGINE_KWARGS) as v2:
        v2_outputs = v2.generate_greedy_logprobs(PROMPTS, MAX_TOKENS, 5)

    check_logprobs_close(
        outputs_0_lst=v1_outputs,
        outputs_1_lst=v2_outputs,
        name_0="v1",
        name_1="v2",
    )


@pytest.mark.model_compile
def test_sampling_and_penalties(vllm_runner, monkeypatch) -> None:
    """The torch sampler paths on the device: temperature, top-k/top-p, a
    seed, and the three penalties. Reproducing a seed across two batched
    calls is not asserted: the batched decode graph's logits differ between
    engine calls by rounding on this hardware (RBLNModelRunner as well), and
    a sampled token flips on that. The seed itself is covered on CPU in
    tests/vllm/patches/test_model_runner_v2.py."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    greedy = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    sampled = SamplingParams(
        temperature=0.9, top_p=0.9, top_k=20, seed=7, max_tokens=MAX_TOKENS
    )
    penalized = SamplingParams(
        temperature=0.9,
        seed=7,
        repetition_penalty=1.3,
        frequency_penalty=0.5,
        presence_penalty=0.2,
        max_tokens=MAX_TOKENS,
    )
    with vllm_runner(MODEL, **ENGINE_KWARGS) as v2:

        def tokens(params: SamplingParams) -> list[list[int]]:
            outputs = v2.llm.generate(PROMPTS, params)
            return [list(o.outputs[0].token_ids) for o in outputs]

        greedy_tokens = tokens(greedy)
        sampled_tokens = tokens(sampled)
        penalized_tokens = tokens(penalized)

    assert all(len(t) == MAX_TOKENS for t in sampled_tokens + penalized_tokens)
    assert sampled_tokens != greedy_tokens, "temperature 0.9 never left the argmax"
    assert penalized_tokens != sampled_tokens, "penalties changed nothing"


def _prompt_logprobs(llm, num_logprobs: int):
    """Prompt logprobs in the shape check_logprobs_close reads: each
    position's top pick stands where the sampled token would."""
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=num_logprobs)
    results = []
    for output in llm.generate(PROMPTS, params):
        logprobs = [
            {tid: lp.logprob for tid, lp in step.items()}
            for step in output.prompt_logprobs[1:]
        ]
        picks = [max(step.items(), key=lambda kv: kv[1])[0] for step in logprobs]
        results.append((picks, "", logprobs))
    return results


@pytest.mark.model_compile
def test_prompt_logprobs_match_hf(hf_runner, vllm_runner, monkeypatch) -> None:
    """RBLNModelRunner cannot serve as the reference here: its prompt-logprob
    path fails in the sampler's gather on this configuration."""
    with hf_runner(MODEL) as hf:
        hf_outputs = hf.prompt_logprobs(PROMPTS, 5)

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(MODEL, **ENGINE_KWARGS) as v2:
        v2_outputs = _prompt_logprobs(v2.llm, 5)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=v2_outputs,
        name_0="hf",
        name_1="v2",
    )


@pytest.mark.model_compile
def test_pooling_matches_hf(hf_runner, vllm_runner, monkeypatch) -> None:
    """Upstream's PoolingRunner on the V2 runner: a decoder served as an
    embedding model. RBLNModelRunner cannot serve as the reference: its
    compiled wrapper still calls compute_logits, which the converted model
    lacks."""
    with hf_runner(MODEL) as hf:
        hf_embeddings = hf.embed(PROMPTS)

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(MODEL, runner="pooling", max_num_seqs=4) as v2:
        v2_embeddings = [
            torch.tensor(o.outputs.embedding) for o in v2.llm.embed(PROMPTS)
        ]

    for i, (hf_e, v2_e) in enumerate(zip(hf_embeddings, v2_embeddings)):
        cosine = torch.dot(hf_e, v2_e.float()).item()
        assert cosine > 0.99, f"prompt {i}: cosine {cosine:.4f} between hf and v2"


@pytest.mark.model_compile
def test_pipeline_parallel_matches_v1(vllm_runner, monkeypatch) -> None:
    """Two stages hand hidden states forward and sampled tokens back. Compared
    with RBLNModelRunner rather than HF: under PP both runners diverge from the
    HF reference in the same way from the first token, which is a property of
    the shared PP path and not of either runner."""
    if rbln_device_count() < 2:
        pytest.skip("pipeline parallelism needs 2 NPUs")
    # max_num_seqs is split across the stages: 4 leaves a decode batch of 2.
    with vllm_runner(MODEL, pipeline_parallel_size=2, max_num_seqs=4) as v1:
        v1_outputs = v1.generate_greedy_logprobs(PROMPTS, MAX_TOKENS, 5)

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(MODEL, pipeline_parallel_size=2, max_num_seqs=4) as v2:
        v2_outputs = v2.generate_greedy_logprobs(PROMPTS, MAX_TOKENS, 5)

    check_logprobs_close(
        outputs_0_lst=v1_outputs, outputs_1_lst=v2_outputs, name_0="v1", name_1="v2"
    )


@pytest.mark.model_compile
def test_data_parallel_matches_hf(hf_runner, async_vllm_runner, monkeypatch) -> None:
    """Two DP ranks of a dense model: every step agrees the padded batch across
    ranks, an idle rank runs the busy rank's shape, and a rank finishing early
    changes the group's shape under the survivor. Output must not move."""
    if rbln_device_count() < 2:
        pytest.skip("data parallelism needs 2 NPUs")
    with hf_runner(MODEL) as hf:
        hf_outputs = hf.generate_greedy_logprobs(PROMPTS, MAX_TOKENS, 5)

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with async_vllm_runner(MODEL, data_parallel_size=2, max_num_seqs=2) as v2:
        both_busy = v2.generate_greedy_logprobs(
            [
                DPRequest(PROMPTS[0], MAX_TOKENS, dp_rank=0),
                DPRequest(PROMPTS[1], MAX_TOKENS, dp_rank=1),
            ],
            5,
        )
        idle_peer = v2.generate_greedy_logprobs(
            [DPRequest(PROMPTS[2], MAX_TOKENS, dp_rank=0)], 5
        )
        early_finish = v2.generate_greedy_logprobs(
            [
                DPRequest(PROMPTS[0], 2, dp_rank=0),
                DPRequest(PROMPTS[2], MAX_TOKENS, dp_rank=1),
            ],
            5,
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs[:2], outputs_1_lst=both_busy, name_0="hf", name_1="v2"
    )
    check_logprobs_close(
        outputs_0_lst=hf_outputs[2:], outputs_1_lst=idle_peer, name_0="hf", name_1="v2"
    )
    check_logprobs_close(
        outputs_0_lst=hf_outputs[2:],
        outputs_1_lst=early_finish[1:],
        name_0="hf",
        name_1="v2 after its peer finished",
    )
