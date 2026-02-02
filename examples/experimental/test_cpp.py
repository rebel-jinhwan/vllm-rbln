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

# ruff: noqa
"""Chunked Pipeline Parallelism (CPP) — correctness & performance test.

This script validates that pipeline-parallel execution with chunked prefill
produces correct results, and measures the performance benefit of async P2P
(the CPP optimisation).

Prerequisites
-------------
* ``VLLM_RBLN_USE_VLLM_MODEL=1`` (V1 RBLN path, required for PP > 1)
* ``VLLM_USE_V1=1``
* ``VLLM_RBLN_BATCH_ATTN_OPT=1`` — required for the RBLN compiler to
  compile decode graphs. Without this the compiler receives per-partition
  seq_lens (shape ``[B, num_partition]``) instead of per-batch seq_idx
  (shape ``[B, 1]``) and fails with
  ``"Batch decode requires seq shape [B, 1], got (B, num_partition)"``.

Known limitations
-----------------
* The RBLN batch-decode kernel has a compiler bug at
  ``batch_bucket_size=4``. Avoid ``max_num_seqs`` values that produce a
  decode batch bucket of exactly 4 (e.g. use 2 or 8 instead of 4).

Usage examples
--------------
# Basic correctness check (PP=2, compare with PP=1 reference)
python test_cpp.py --model meta-llama/Llama-3.2-1B --pp 2 --correctness

# Performance benchmark
python test_cpp.py --model meta-llama/Llama-3.2-1B --pp 2 --benchmark

# Disable async send to measure its impact
VLLM_RBLN_ASYNC_PP_SEND=0 python test_cpp.py --model meta-llama/Llama-3.2-1B --pp 2 --benchmark

# Full test: correctness + benchmark
python test_cpp.py --model meta-llama/Llama-3.2-1B --pp 2 --correctness --benchmark
"""

import argparse
import os
import time

os.environ.setdefault("VLLM_RBLN_USE_VLLM_MODEL", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
# Required: batch attention opt provides (B, 1) seq_idx to the compiler
# instead of (B, num_partition) dyn_size_for_partitions which fails compilation.
os.environ.setdefault("VLLM_RBLN_BATCH_ATTN_OPT", "1")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser(
        description="CPP (Chunked Pipeline Parallelism) test & benchmark")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--pp", type=int, default=2, help="pipeline_parallel_size")
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=2,
        help="Avoid 4 (compiler bug at batch_bucket=4)",
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=512,
        help="Chunk size for chunked prefill",
    )
    p.add_argument("--block-size", type=int, default=1024)

    # Test modes
    p.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness test (PP=1 vs PP=N)",
    )
    p.add_argument("--benchmark",
                   action="store_true",
                   help="Run performance benchmark")

    # Benchmark parameters
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--warmup-requests", type=int, default=3)
    return p.parse_args()


def build_llm(args, pp_override=None):
    """Create an LLM instance with the given PP override."""
    pp = pp_override if pp_override is not None else args.pp
    return LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=pp,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=1.0,
        trust_remote_code=True,
    )


def make_prompts(tokenizer, num_requests, input_len):
    """Build deterministic prompts with exactly *input_len* tokens each."""
    # Use token IDs directly to guarantee exact length
    dummy_ids = [1] * input_len  # token ID 1 repeated
    prompt_text = tokenizer.decode(dummy_ids, skip_special_tokens=True)
    # Verify and adjust
    encoded = tokenizer.encode(prompt_text, add_special_tokens=True)
    if len(encoded) != input_len:
        # Fallback: repeat a single-char token
        vocab = iter(tokenizer.vocab)
        single_tok = next(vocab)
        while True:
            if len(tokenizer.encode(single_tok * 2,
                                    add_special_tokens=False)) == 2:
                break
            single_tok = next(vocab)
        prompt_text = single_tok * (input_len - 1)
    return [prompt_text] * num_requests


# -----------------------------------------------------------------------
# Correctness test
# -----------------------------------------------------------------------
def run_correctness_test(args):
    """Generate with PP=1 and PP=N, compare token-by-token."""
    sampling = SamplingParams(temperature=0.0, max_tokens=args.output_len)

    prompts = [
        "The capital of France is",
        "The president of the United States is",
        "Once upon a time in a faraway land,",
    ]

    print("=" * 60)
    print(f"Correctness test: PP=1 vs PP={args.pp}")
    print("=" * 60)

    # ---- PP=1 reference ----
    print("\n[1/2] Running PP=1 reference ...")
    llm_ref = build_llm(args, pp_override=1)
    ref_outputs = llm_ref.generate(prompts, sampling)
    ref_texts = [o.outputs[0].text for o in ref_outputs]
    del llm_ref  # free resources

    # ---- PP=N ----
    print(f"\n[2/2] Running PP={args.pp} ...")
    llm_pp = build_llm(args)
    pp_outputs = llm_pp.generate(prompts, sampling)
    pp_texts = [o.outputs[0].text for o in pp_outputs]
    del llm_pp

    # ---- compare ----
    all_match = True
    for i, (ref, pp) in enumerate(zip(ref_texts, pp_texts)):
        match = ref == pp
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"  Prompt {i}: {status}")
        if not match:
            all_match = False
            print(f"    PP=1:  {ref[:120]}...")
            print(f"    PP={args.pp}: {pp[:120]}...")

    print()
    if all_match:
        print("CORRECTNESS: PASSED ✓")
    else:
        print("CORRECTNESS: FAILED ✗")
    return all_match


# -----------------------------------------------------------------------
# Performance benchmark
# -----------------------------------------------------------------------
def run_benchmark(args):
    """Measure TTFT and decode throughput with chunked-prefill + PP."""
    print("=" * 60)
    print(f"Benchmark: PP={args.pp}  chunk={args.max_num_batched_tokens}  "
          f"requests={args.num_requests}  in={args.input_len}  "
          f"out={args.output_len}")
    print(
        f"  async_pp_send = {os.environ.get('VLLM_RBLN_ASYNC_PP_SEND', '1')}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = make_prompts(tokenizer, args.num_requests, args.input_len)

    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    llm = build_llm(args)

    # Warmup
    print(f"\nWarmup ({args.warmup_requests} requests) ...")
    _ = llm.generate(prompts[:min(len(prompts), args.warmup_requests)],
                     sampling)

    # Timed run
    print("Timed run ...")
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0

    # Validate output shapes
    input_lens = [
        len(o.prompt_token_ids) if o.prompt_token_ids else 0 for o in outputs
    ]
    output_lens = [len(o.outputs[0].token_ids) for o in outputs]
    if any(l != args.input_len for l in input_lens):
        print(f"  WARNING: Input length mismatch. Expected {args.input_len}, "
              f"got {set(input_lens)}")
    if any(l != args.output_len for l in output_lens):
        print(
            f"  WARNING: Output length mismatch. Expected {args.output_len}, "
            f"got {set(output_lens)}")

    total_input_tokens = args.num_requests * args.input_len
    total_output_tokens = args.num_requests * args.output_len

    print(f"\nResults:")
    print(f"  Total time:           {elapsed:.2f} s")
    print(f"  Prefill throughput:   {total_input_tokens / elapsed:.0f} tok/s")
    print(f"  Decode throughput:    {total_output_tokens / elapsed:.0f} tok/s")
    print(f"  Total throughput:     "
          f"{(total_input_tokens + total_output_tokens) / elapsed:.0f} tok/s")
    print(f"  Requests/sec:         {args.num_requests / elapsed:.2f}")

    del llm
    return elapsed


def main():
    args = parse_args()

    if not args.correctness and not args.benchmark:
        print("Specify --correctness and/or --benchmark. "
              "Running both by default.\n")
        args.correctness = True
        args.benchmark = True

    ok = True
    if args.correctness:
        ok = run_correctness_test(args)

    if args.benchmark:
        run_benchmark(args)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
