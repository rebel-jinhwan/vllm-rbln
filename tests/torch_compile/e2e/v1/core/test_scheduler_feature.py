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

"""Feature tests for RBLNScheduler.

Tests vllm compatibility and RBLN-specific scheduling constraints:
- No mixed batching (prefill and decode cannot coexist in the same batch)
- Prefill batch size = 1
- Decode batch size limited by max_num_seqs
- spec_decode_cap prevents spec tokens from crossing block boundaries
- undo_uncomputed_block_caching resets block hash for uncomputed blocks
- is_prefill helper edge cases
- Full request lifecycle
"""

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from tests.torch_compile.unit.v1.core.utils import (
    create_requests,
    create_runner_output,
    create_scheduler,
)
from vllm_rbln.v1.core.rbln_scheduler import (
    RBLNScheduler,
    is_prefill,
    undo_uncomputed_block_caching,
)


# ===========================================================================
# 1. Interface compliance
# ===========================================================================


class TestInterfaceCompliance:
    """Verify RBLNScheduler inherits from Scheduler and implements schedule()."""

    def test_inherits_from_scheduler(self):
        assert issubclass(RBLNScheduler, Scheduler)

    def test_has_schedule_method(self):
        assert hasattr(RBLNScheduler, "schedule")
        assert callable(getattr(RBLNScheduler, "schedule"))

    def test_instance_is_scheduler(self):
        scheduler = create_scheduler()
        assert isinstance(scheduler, Scheduler)
        assert isinstance(scheduler, RBLNScheduler)

    def test_schedule_returns_scheduler_output(self):
        from vllm.v1.core.sched.output import SchedulerOutput

        scheduler = create_scheduler()
        output = scheduler.schedule()
        assert isinstance(output, SchedulerOutput)


# ===========================================================================
# 2. No mixed batching
# ===========================================================================


class TestNoMixedBatching:
    """Prefill and decode requests must NOT be in the same batch."""

    def test_prefill_excludes_running_decode(self):
        """When a new prefill is scheduled, running decode requests are
        excluded from that step's output."""
        scheduler = create_scheduler()
        reqs = create_requests(num_requests=3, num_tokens=10)

        # Prefill first request
        scheduler.add_request(reqs[0])
        out = scheduler.schedule()
        scheduler.update_from_output(out, create_runner_output(out, 0))

        # Now reqs[0] is in decode state. Add reqs[1] (waiting -> prefill).
        scheduler.add_request(reqs[1])
        out = scheduler.schedule()

        # Only the new prefill request should be scheduled
        assert len(out.scheduled_new_reqs) == 1
        assert out.scheduled_new_reqs[0].req_id == reqs[1].request_id
        # reqs[0] (decode) should NOT be in this batch
        assert reqs[0].request_id not in out.num_scheduled_tokens

    def test_decode_excludes_waiting_prefill_when_running(self):
        """When decode requests are running and a waiting prefill exists,
        the prefill takes over and decode is excluded."""
        scheduler = create_scheduler()
        reqs = create_requests(num_requests=3, num_tokens=10)

        # Prefill reqs[0] and reqs[1]
        for r in reqs[:2]:
            scheduler.add_request(r)
        for _ in range(2):
            out = scheduler.schedule()
            scheduler.update_from_output(out, create_runner_output(out, 0))

        # Both are now in decode state. Add reqs[2] as waiting.
        scheduler.add_request(reqs[2])
        out = scheduler.schedule()

        # The new prefill should take over; decode requests excluded
        assert len(out.scheduled_new_reqs) == 1
        assert out.scheduled_new_reqs[0].req_id == reqs[2].request_id
        assert reqs[0].request_id not in out.num_scheduled_tokens
        assert reqs[1].request_id not in out.num_scheduled_tokens

    def test_all_running_are_prefill_schedules_exactly_one(self):
        """When all running requests are mid-prefill (chunked), only one
        is scheduled per step due to no-mixed-batching + batch=1."""
        scheduler = create_scheduler(max_num_batched_tokens=256)
        reqs = create_requests(num_requests=2, num_tokens=500)

        # Add both requests
        for r in reqs:
            scheduler.add_request(r)

        # First step: only one prefill scheduled
        out = scheduler.schedule()
        assert len(out.num_scheduled_tokens) == 1
        assert len(out.scheduled_new_reqs) == 1


# ===========================================================================
# 3. Prefill batch size = 1
# ===========================================================================


class TestPrefillBatchSizeOne:
    """Only one prefill request per step."""

    def test_single_prefill_per_step(self):
        """Submit multiple new requests; verify exactly 1 is prefilled per step."""
        scheduler = create_scheduler()
        reqs = create_requests(num_requests=5, num_tokens=10)
        for r in reqs:
            scheduler.add_request(r)

        for i in range(5):
            out = scheduler.schedule()
            assert len(out.scheduled_new_reqs) == 1
            assert out.scheduled_new_reqs[0].req_id == reqs[i].request_id
            # Only this one request gets tokens
            assert len(out.num_scheduled_tokens) == 1
            scheduler.update_from_output(out, create_runner_output(out, 0))

    def test_chunked_prefill_still_one_at_a_time(self):
        """With chunked prefill, a long request takes multiple steps but
        still only one prefill at a time."""
        scheduler = create_scheduler(max_num_batched_tokens=100)
        reqs = create_requests(num_requests=2, num_tokens=200)
        for r in reqs:
            scheduler.add_request(r)

        # Step 1: first chunk of reqs[0]
        out = scheduler.schedule()
        assert len(out.num_scheduled_tokens) == 1
        assert reqs[0].request_id in out.num_scheduled_tokens
        assert out.num_scheduled_tokens[reqs[0].request_id] == 100
        scheduler.update_from_output(out, create_runner_output(out))

        # Step 2: second chunk of reqs[0]
        out = scheduler.schedule()
        assert len(out.num_scheduled_tokens) == 1
        assert reqs[0].request_id in out.num_scheduled_tokens
        assert out.num_scheduled_tokens[reqs[0].request_id] == 100
        scheduler.update_from_output(out, create_runner_output(out, 0))

        # Step 3: now reqs[0] enters decode; reqs[1] starts prefill
        # With no-mixed-batching, only reqs[1] gets scheduled
        out = scheduler.schedule()
        assert len(out.scheduled_new_reqs) == 1
        assert out.scheduled_new_reqs[0].req_id == reqs[1].request_id


# ===========================================================================
# 4. Decode batch size limit
# ===========================================================================


class TestDecodeBatchLimit:
    """Decode batch should be limited by max_num_seqs."""

    def test_decode_limited_by_max_num_seqs(self):
        """With max_num_seqs=4, at most 4 requests can be in decode."""
        max_seqs = 4
        scheduler = create_scheduler(max_num_seqs=max_seqs)
        reqs = create_requests(num_requests=max_seqs + 2, num_tokens=10)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill max_seqs requests (one per step)
        for _ in range(max_seqs):
            out = scheduler.schedule()
            scheduler.update_from_output(out, create_runner_output(out, 0))

        assert len(scheduler.running) == max_seqs
        assert len(scheduler.waiting) == 2

        # Decode step: only max_seqs requests scheduled
        out = scheduler.schedule()
        # No new prefills because all slots are taken
        assert len(out.scheduled_new_reqs) == 0
        assert out.scheduled_cached_reqs.num_reqs == max_seqs

    def test_decode_batch_with_pp_size_simulation(self):
        """Simulate pp_size=2 by setting max_num_seqs=4 (from original 8).
        Verify at most 4 decode requests run."""
        # In practice, max_num_seqs is set to max_num_seqs // pp_size externally
        effective_max_seqs = 8 // 2  # pp_size=2
        scheduler = create_scheduler(
            max_num_seqs=effective_max_seqs,
            pipeline_parallel_size=2,
        )
        reqs = create_requests(num_requests=6, num_tokens=10)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill 4 requests
        for _ in range(effective_max_seqs):
            out = scheduler.schedule()
            scheduler.update_from_output(out, create_runner_output(out, 0))

        assert len(scheduler.running) == effective_max_seqs

        # Decode step
        out = scheduler.schedule()
        assert out.scheduled_cached_reqs.num_reqs == effective_max_seqs
        assert len(out.scheduled_new_reqs) == 0


# ===========================================================================
# 5. spec_decode_cap
# ===========================================================================


class TestSpecDecodeCap:
    """Verify spec tokens don't cross block boundaries."""

    _BLOCK_SIZE = 1024
    _NUM_BLOCKS = 100
    _MAX_NUM_SEQS = 10

    def _make_scheduler(self, **kwargs):
        return create_scheduler(
            block_size=self._BLOCK_SIZE,
            num_blocks=self._NUM_BLOCKS,
            max_num_seqs=self._MAX_NUM_SEQS,
            **kwargs,
        )

    def _make_request(self, num_tokens, req_id):
        return create_requests(
            num_requests=1,
            num_tokens=num_tokens,
            block_size=self._BLOCK_SIZE,
            max_tokens=2048,
            req_ids=[req_id],
        )[0]

    def _advance_to_decode(self, scheduler, request):
        scheduler.add_request(request)
        out = scheduler.schedule()
        scheduler.update_from_output(out, create_runner_output(out, 1))

    def test_spec_tokens_trimmed_near_block_boundary(self):
        """Request at block_size-1 tokens: remaining_in_block=1, so all
        spec tokens beyond 1 total scheduled token are trimmed."""
        scheduler = self._make_scheduler()
        req = self._make_request(1023, "A")
        self._advance_to_decode(scheduler, req)

        req.spec_token_ids = [1] * 4
        out = scheduler.schedule()

        # Only 1 token scheduled (decode token), all spec removed
        assert out.num_scheduled_tokens[req.request_id] == 1
        assert req.request_id not in out.scheduled_spec_decode_tokens

    def test_spec_tokens_not_trimmed_at_block_boundary(self):
        """Request at exact block_size: remaining_in_block = block_size,
        so spec tokens are not trimmed."""
        scheduler = self._make_scheduler()
        req = self._make_request(1024, "A")
        self._advance_to_decode(scheduler, req)

        req.spec_token_ids = [1] * 4
        out = scheduler.schedule()

        assert out.num_scheduled_tokens[req.request_id] == 5
        assert len(out.scheduled_spec_decode_tokens[req.request_id]) == 4

    def test_spec_tokens_partial_fit(self):
        """Request with remaining_in_block=4: only 3 spec tokens survive
        (1 decode + 3 spec = 4)."""
        scheduler = self._make_scheduler()
        req = self._make_request(1020, "A")
        self._advance_to_decode(scheduler, req)

        req.spec_token_ids = [1] * 6
        out = scheduler.schedule()

        assert out.num_scheduled_tokens[req.request_id] == 4
        assert len(out.scheduled_spec_decode_tokens[req.request_id]) == 3

    def test_retroactive_trim_across_requests(self):
        """Two requests: B tightens cap, A is retroactively trimmed."""
        scheduler = self._make_scheduler()
        req_a = self._make_request(1024, "A")
        req_b = self._make_request(1020, "B")
        self._advance_to_decode(scheduler, req_a)
        self._advance_to_decode(scheduler, req_b)

        req_a.spec_token_ids = [1] * 6
        req_b.spec_token_ids = [1] * 6
        out = scheduler.schedule()

        # Both trimmed to cap=4 (remaining_in_block for B)
        assert out.num_scheduled_tokens[req_a.request_id] == 4
        assert out.num_scheduled_tokens[req_b.request_id] == 4
        assert len(out.scheduled_spec_decode_tokens[req_a.request_id]) == 3
        assert len(out.scheduled_spec_decode_tokens[req_b.request_id]) == 3


# ===========================================================================
# 6. is_prefill helper
# ===========================================================================


class TestIsPrefillHelper:
    """Test edge cases of the is_prefill helper function."""

    def test_new_request_is_prefill(self):
        req = create_requests(num_requests=1, num_tokens=10)[0]
        assert is_prefill(req) is True

    def test_partially_computed_is_prefill(self):
        req = create_requests(num_requests=1, num_tokens=10)[0]
        req.num_computed_tokens = 5
        assert is_prefill(req) is True

    def test_one_token_remaining_is_prefill(self):
        """num_computed_tokens = num_tokens - 2 is still prefill."""
        req = create_requests(num_requests=1, num_tokens=10)[0]
        req.num_computed_tokens = 8  # 10 - 2 = 8
        assert is_prefill(req) is True

    def test_boundary_not_prefill(self):
        """num_computed_tokens = num_tokens - 1 is NOT prefill."""
        req = create_requests(num_requests=1, num_tokens=10)[0]
        req.num_computed_tokens = 9  # 10 - 1 = 9
        assert is_prefill(req) is False

    def test_fully_computed_not_prefill(self):
        """num_computed_tokens = num_tokens is NOT prefill."""
        req = create_requests(num_requests=1, num_tokens=10)[0]
        req.num_computed_tokens = 10
        assert is_prefill(req) is False

    def test_single_token_request(self):
        """A single-token request: num_computed_tokens=0, num_tokens=1.
        0 < 1-1=0 is False, so NOT considered prefill."""
        req = create_requests(num_requests=1, num_tokens=1)[0]
        assert is_prefill(req) is False

    def test_two_token_request(self):
        """Two-token request: num_computed_tokens=0, num_tokens=2.
        0 < 2-1=1 is True, so IS prefill."""
        req = create_requests(num_requests=1, num_tokens=2)[0]
        assert is_prefill(req) is True


# ===========================================================================
# 7. undo_uncomputed_block_caching
# ===========================================================================


class TestUndoUncomputedBlockCaching:
    """Test that uncomputed blocks are properly evicted."""

    def test_undo_called_during_prefill_scheduling(self):
        """When a new request is scheduled for prefill, blocks beyond the
        computed range should have their hash reset. We verify this
        indirectly by checking that the request can still be scheduled
        successfully in subsequent steps (no stale cache entries)."""
        scheduler = create_scheduler(
            max_num_batched_tokens=256,
            block_size=16,
            num_blocks=100,
            enable_prefix_caching=True,
        )
        req = create_requests(num_requests=1, num_tokens=100, block_size=16)[0]
        scheduler.add_request(req)

        # First chunk
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 100
        scheduler.update_from_output(out, create_runner_output(out, 0))

        # After prefill completes, request enters decode
        out = scheduler.schedule()
        assert req.request_id in out.num_scheduled_tokens
        assert out.num_scheduled_tokens[req.request_id] == 1

    def test_undo_during_chunked_prefill(self):
        """During chunked prefill, blocks allocated but not yet computed
        should be properly evicted so subsequent chunks work correctly."""
        scheduler = create_scheduler(
            max_num_batched_tokens=50,
            block_size=16,
            num_blocks=100,
            enable_prefix_caching=True,
        )
        req = create_requests(num_requests=1, num_tokens=100, block_size=16)[0]
        scheduler.add_request(req)

        # First chunk: 50 tokens
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 50
        scheduler.update_from_output(out, create_runner_output(out))

        # Second chunk: 50 tokens
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 50
        scheduler.update_from_output(out, create_runner_output(out, 0))

        # Decode
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 1


# ===========================================================================
# 8. Request lifecycle
# ===========================================================================


class TestRequestLifecycle:
    """Full lifecycle: add_request -> prefill -> update -> decode -> finish."""

    def test_full_lifecycle(self):
        scheduler = create_scheduler()
        req = create_requests(num_requests=1, num_tokens=10, max_tokens=2)[0]
        scheduler.add_request(req)

        # Step 1: Prefill
        out = scheduler.schedule()
        assert len(out.scheduled_new_reqs) == 1
        assert out.scheduled_new_reqs[0].req_id == req.request_id
        assert out.num_scheduled_tokens[req.request_id] == 10
        scheduler.update_from_output(out, create_runner_output(out, 100))

        assert req.status == RequestStatus.RUNNING
        assert len(scheduler.running) == 1
        assert len(scheduler.waiting) == 0

        # Step 2: First decode
        out = scheduler.schedule()
        assert out.scheduled_cached_reqs.num_reqs == 1
        assert out.num_scheduled_tokens[req.request_id] == 1
        scheduler.update_from_output(out, create_runner_output(out, 200))

        # Step 3: Second decode (should trigger finish since max_tokens=2)
        out = scheduler.schedule()
        scheduler.update_from_output(out, create_runner_output(out, 300))

        # Step 4: Request should now be finished
        out = scheduler.schedule()
        assert req.request_id in out.finished_req_ids or len(scheduler.running) == 0

    def test_multiple_requests_lifecycle(self):
        """Multiple requests go through prefill one at a time, then
        decode together."""
        scheduler = create_scheduler()
        reqs = create_requests(num_requests=3, num_tokens=10, max_tokens=5)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill each request one at a time
        for i in range(3):
            out = scheduler.schedule()
            assert len(out.scheduled_new_reqs) == 1
            scheduler.update_from_output(out, create_runner_output(out, 0))

        # All 3 in running, 0 in waiting
        assert len(scheduler.running) == 3
        assert len(scheduler.waiting) == 0

        # Decode step: all 3 scheduled together
        out = scheduler.schedule()
        assert out.scheduled_cached_reqs.num_reqs == 3
        assert len(out.num_scheduled_tokens) == 3
        assert all(n == 1 for n in out.num_scheduled_tokens.values())


# ===========================================================================
# Bug-catching tests
# ===========================================================================


class TestEdgeCases:
    """Edge cases and potential bug scenarios."""

    def test_empty_schedule_no_requests(self):
        """Schedule with no requests produces empty output."""
        scheduler = create_scheduler()
        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens == 0
        assert len(out.scheduled_new_reqs) == 0
        assert out.scheduled_cached_reqs.num_reqs == 0

    def test_preemption_respects_rbln_constraints(self):
        """Under memory pressure, preemption still respects RBLN constraints:
        no mixed batching, prefill batch = 1."""
        # 20 blocks available (block 0 is null = 19 usable).
        # Each 80-token request needs 5 blocks (80/16).
        # After prefilling 3 requests: 15 blocks used, 4 free.
        # A 4th request needing 5 blocks triggers preemption.
        scheduler = create_scheduler(
            max_num_batched_tokens=100,
            block_size=16,
            num_blocks=20,
            enable_prefix_caching=False,
        )
        reqs = create_requests(num_requests=4, num_tokens=80, block_size=16)

        # Prefill first three requests (one per step)
        for r in reqs[:3]:
            scheduler.add_request(r)
        for _ in range(3):
            out = scheduler.schedule()
            assert len(out.scheduled_new_reqs) == 1
            scheduler.update_from_output(out, create_runner_output(out, 0))

        assert len(scheduler.running) == 3

        # Add fourth request - memory is tight
        scheduler.add_request(reqs[3])
        out = scheduler.schedule()

        # The prefill should take over; no mixed batching
        scheduled_ids = set(out.num_scheduled_tokens.keys())
        if len(out.scheduled_new_reqs) > 0:
            # Prefill happened: no decode requests should be in the same batch
            for new_req in out.scheduled_new_reqs:
                assert new_req.req_id in scheduled_ids
            # Verify no decode requests are mixed in
            decode_req_ids = {r.request_id for r in reqs[:3]}
            assert scheduled_ids.isdisjoint(decode_req_ids)

    def test_no_mixed_batch_after_preemption(self):
        """After preemption, verify the batch still has no mixing."""
        scheduler = create_scheduler(
            max_num_batched_tokens=200,
            block_size=16,
            num_blocks=20,
            enable_prefix_caching=False,
        )
        reqs = create_requests(num_requests=2, num_tokens=50, block_size=16)

        # Prefill both requests
        for r in reqs:
            scheduler.add_request(r)
        for _ in range(2):
            out = scheduler.schedule()
            scheduler.update_from_output(out, create_runner_output(out, 0))

        # Both in decode now
        out = scheduler.schedule()
        # All scheduled requests should be decode (no new prefills)
        # or all should be prefill (if new request added)
        assert len(out.scheduled_new_reqs) == 0
        assert out.scheduled_cached_reqs.num_reqs == 2

    def test_waiting_request_does_not_start_during_decode(self):
        """When decode requests fill max_num_seqs, waiting requests do not
        get scheduled."""
        max_seqs = 2
        scheduler = create_scheduler(max_num_seqs=max_seqs)
        reqs = create_requests(num_requests=3, num_tokens=10)

        for r in reqs:
            scheduler.add_request(r)

        # Prefill 2 requests
        for _ in range(max_seqs):
            out = scheduler.schedule()
            scheduler.update_from_output(out, create_runner_output(out, 0))

        assert len(scheduler.running) == max_seqs
        assert len(scheduler.waiting) == 1

        # Decode step: waiting request cannot be added
        out = scheduler.schedule()
        assert len(out.scheduled_new_reqs) == 0
        assert out.scheduled_cached_reqs.num_reqs == max_seqs


# ===========================================================================
# 9. Long prefill threshold (chunked by threshold)
# ===========================================================================


class TestLongPrefillThreshold:
    """Test long_prefill_token_threshold limits tokens per prefill step."""

    def test_long_prefill_chunks_by_threshold(self):
        """A prompt longer than the threshold is chunked to that threshold."""
        threshold = 16
        scheduler = create_scheduler(
            long_prefill_token_threshold=threshold,
            max_num_batched_tokens=8192,
            block_size=16,
            num_blocks=1000,
        )
        req = create_requests(num_requests=1, num_tokens=32, block_size=16)[0]
        scheduler.add_request(req)

        # First step: should schedule only threshold tokens
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == threshold
        scheduler.update_from_output(out, create_runner_output(out))

        # Second step: remaining tokens
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 32 - threshold
        scheduler.update_from_output(out, create_runner_output(out, 0))

        # Third step: decode
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 1

    def test_threshold_zero_means_no_limit(self):
        """Threshold of 0 means no chunking by threshold."""
        scheduler = create_scheduler(
            long_prefill_token_threshold=0,
            max_num_batched_tokens=8192,
            block_size=16,
            num_blocks=1000,
        )
        req = create_requests(num_requests=1, num_tokens=64, block_size=16)[0]
        scheduler.add_request(req)

        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 64

    def test_threshold_larger_than_prompt_no_effect(self):
        """When threshold > prompt length, all tokens scheduled at once."""
        scheduler = create_scheduler(
            long_prefill_token_threshold=100,
            max_num_batched_tokens=8192,
            block_size=16,
            num_blocks=1000,
        )
        req = create_requests(num_requests=1, num_tokens=32, block_size=16)[0]
        scheduler.add_request(req)

        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 32


# ===========================================================================
# 10. Waiting queue with budget exhaustion
# ===========================================================================


class TestWaitingQueueBudgetExhaustion:
    """Submit many requests that exhaust token_budget; verify some stay waiting."""

    def test_requests_stay_in_waiting_when_slots_full(self):
        """With max_num_seqs=2, only 2 requests get scheduled; rest wait."""
        scheduler = create_scheduler(max_num_seqs=2)
        reqs = create_requests(num_requests=5, num_tokens=10)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill 2 requests (one per step)
        for _ in range(2):
            out = scheduler.schedule()
            assert len(out.scheduled_new_reqs) == 1
            scheduler.update_from_output(out, create_runner_output(out, 0))

        assert len(scheduler.running) == 2
        assert len(scheduler.waiting) == 3

        # Decode step: no new requests can be added
        out = scheduler.schedule()
        assert len(out.scheduled_new_reqs) == 0
        assert out.scheduled_cached_reqs.num_reqs == 2

    def test_waiting_requests_scheduled_after_finish(self):
        """After a running request finishes, a waiting request gets scheduled."""
        scheduler = create_scheduler(max_num_seqs=1)
        reqs = create_requests(num_requests=2, num_tokens=10, max_tokens=2)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill first request
        out = scheduler.schedule()
        assert out.scheduled_new_reqs[0].req_id == reqs[0].request_id
        scheduler.update_from_output(out, create_runner_output(out, 100))

        # Decode until first request finishes and second gets scheduled
        found_second = False
        for _ in range(10):
            out = scheduler.schedule()
            if len(out.scheduled_new_reqs) > 0:
                assert out.scheduled_new_reqs[0].req_id == reqs[1].request_id
                found_second = True
                break
            if out.total_num_scheduled_tokens == 0:
                # Nothing scheduled, nothing to update
                continue
            scheduler.update_from_output(out, create_runner_output(out, 100))
        assert found_second, "Second request was never scheduled"


# ===========================================================================
# 11. New request block allocation failure
# ===========================================================================


class TestNewRequestBlockAllocationFailure:
    """Create scheduler with very limited blocks; verify allocation failure."""

    def test_block_allocation_fails_with_limited_blocks(self):
        """With very few blocks, a large request cannot be allocated."""
        # 3 blocks total, block 0 is null -> 2 usable blocks = 32 tokens.
        # A request with 48 tokens needs 3 blocks -> fails.
        scheduler = create_scheduler(
            max_num_batched_tokens=8192,
            block_size=16,
            num_blocks=3,
            enable_prefix_caching=False,
        )
        req = create_requests(num_requests=1, num_tokens=48, block_size=16)[0]
        scheduler.add_request(req)

        out = scheduler.schedule()
        # The request should not be scheduled since there aren't enough blocks
        assert req.request_id not in out.num_scheduled_tokens
        assert out.total_num_scheduled_tokens == 0
        # Request stays in waiting
        assert len(scheduler.waiting) == 1

    def test_second_request_fails_when_blocks_exhausted(self):
        """First request succeeds but second fails due to block exhaustion."""
        # 5 blocks: 1 null + 4 usable = 64 tokens.
        # First request: 48 tokens = 3 blocks. OK.
        # Second request: 48 tokens = 3 blocks. Only 1 left -> fail.
        scheduler = create_scheduler(
            max_num_batched_tokens=8192,
            block_size=16,
            num_blocks=5,
            enable_prefix_caching=False,
        )
        reqs = create_requests(num_requests=2, num_tokens=48, block_size=16)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill first request
        out = scheduler.schedule()
        assert reqs[0].request_id in out.num_scheduled_tokens
        scheduler.update_from_output(out, create_runner_output(out, 0))

        # Try to prefill second request - should fail
        out = scheduler.schedule()
        assert reqs[1].request_id not in out.num_scheduled_tokens
        # Second request stays in waiting
        assert len(scheduler.waiting) == 1


# ===========================================================================
# 12. Spec decode tokens
# ===========================================================================


class TestSpecDecodeTokens:
    """Test spec_token_ids scheduling with block boundary constraints."""

    _BLOCK_SIZE = 16
    _NUM_BLOCKS = 1000

    def _make_scheduler(self, **kwargs):
        return create_scheduler(
            block_size=self._BLOCK_SIZE,
            num_blocks=self._NUM_BLOCKS,
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            **kwargs,
        )

    def test_spec_tokens_scheduled(self):
        """Spec tokens that fit within the block are scheduled."""
        scheduler = self._make_scheduler()
        req = create_requests(
            num_requests=1, num_tokens=16, block_size=self._BLOCK_SIZE,
            max_tokens=100,
        )[0]
        scheduler.add_request(req)

        # Prefill
        out = scheduler.schedule()
        scheduler.update_from_output(out, create_runner_output(out, 1))

        # Now in decode. Set spec tokens. Position is at block boundary (16),
        # so full block of space remains.
        req.spec_token_ids = [10, 20, 30]
        out = scheduler.schedule()

        assert out.num_scheduled_tokens[req.request_id] == 4  # 1 decode + 3 spec
        assert req.request_id in out.scheduled_spec_decode_tokens
        assert out.scheduled_spec_decode_tokens[req.request_id] == [10, 20, 30]

    def test_spec_tokens_trimmed_at_block_boundary(self):
        """Spec tokens that would cross block boundary are trimmed."""
        scheduler = self._make_scheduler()
        # Prompt of 15 tokens. After prefill, num_computed_tokens = 15.
        # After 1 output token appended, num_tokens = 16.
        # At next schedule: remaining_in_block = 16 - (15 % 16) = 1.
        # With 3 spec tokens: 4 new tokens requested but cap=1, so only 1 scheduled.
        req = create_requests(
            num_requests=1, num_tokens=15, block_size=self._BLOCK_SIZE,
            max_tokens=100,
        )[0]
        scheduler.add_request(req)

        # Prefill: num_computed_tokens set to 15 after _update_after_schedule
        out = scheduler.schedule()
        scheduler.update_from_output(out, create_runner_output(out, 1))

        # Now: num_computed_tokens=15, num_tokens=16
        # Set spec tokens before next schedule
        req.spec_token_ids = [10, 20, 30]
        out = scheduler.schedule()

        # remaining_in_block = 16 - 15 = 1, so only 1 token scheduled
        assert out.num_scheduled_tokens[req.request_id] == 1
        assert req.request_id not in out.scheduled_spec_decode_tokens

    def test_spec_tokens_cleared_after_scheduling(self):
        """After scheduling, spec_token_ids on the request is cleared."""
        scheduler = self._make_scheduler()
        req = create_requests(
            num_requests=1, num_tokens=16, block_size=self._BLOCK_SIZE,
            max_tokens=100,
        )[0]
        scheduler.add_request(req)

        out = scheduler.schedule()
        scheduler.update_from_output(out, create_runner_output(out, 1))

        req.spec_token_ids = [10, 20]
        out = scheduler.schedule()

        # spec_token_ids should be cleared on the request after scheduling
        assert req.spec_token_ids == []


# ===========================================================================
# 13. Multiple decode steps
# ===========================================================================


class TestMultipleDecodeSteps:
    """Run several decode cycles to exercise the running request loop."""

    def test_multiple_decode_cycles(self):
        """Run 10 decode steps with multiple requests."""
        scheduler = create_scheduler(max_num_seqs=4)
        reqs = create_requests(num_requests=4, num_tokens=10, max_tokens=20)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill all 4 (one per step)
        for _ in range(4):
            out = scheduler.schedule()
            assert len(out.scheduled_new_reqs) == 1
            scheduler.update_from_output(out, create_runner_output(out, 0))

        assert len(scheduler.running) == 4

        # Run 10 decode steps
        for step in range(10):
            out = scheduler.schedule()
            assert out.scheduled_cached_reqs.num_reqs == 4
            assert all(n == 1 for n in out.num_scheduled_tokens.values())
            scheduler.update_from_output(out, create_runner_output(out, step + 100))

    def test_decode_until_completion(self):
        """Run decode until requests complete (max_tokens reached)."""
        scheduler = create_scheduler(max_num_seqs=2)
        reqs = create_requests(num_requests=2, num_tokens=10, max_tokens=3)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill both
        for _ in range(2):
            out = scheduler.schedule()
            scheduler.update_from_output(out, create_runner_output(out, 0))

        # Decode until finished
        finished_count = 0
        for _ in range(10):
            out = scheduler.schedule()
            if out.total_num_scheduled_tokens == 0:
                break
            scheduler.update_from_output(out, create_runner_output(out, 0))
            finished_count += len(out.finished_req_ids)

        # All requests should eventually finish
        assert len(scheduler.running) == 0


# ===========================================================================
# 14. Paused state
# ===========================================================================


class TestPausedState:
    """Set pause state and verify scheduling behavior."""

    def test_paused_all_schedules_nothing(self):
        """When PAUSED_ALL, token_budget=0 so nothing is scheduled."""
        from vllm.v1.core.sched.interface import PauseState

        scheduler = create_scheduler()
        reqs = create_requests(num_requests=3, num_tokens=10)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill one request first
        out = scheduler.schedule()
        scheduler.update_from_output(out, create_runner_output(out, 0))
        assert len(scheduler.running) == 1

        # Pause
        scheduler.set_pause_state(PauseState.PAUSED_ALL)

        # Schedule should produce nothing
        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens == 0
        assert len(out.scheduled_new_reqs) == 0
        assert out.scheduled_cached_reqs.num_reqs == 0

    def test_unpause_resumes_scheduling(self):
        """After unpausing, scheduling resumes normally."""
        from vllm.v1.core.sched.interface import PauseState

        scheduler = create_scheduler()
        reqs = create_requests(num_requests=2, num_tokens=10)
        for r in reqs:
            scheduler.add_request(r)

        # Pause before any scheduling
        scheduler.set_pause_state(PauseState.PAUSED_ALL)
        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens == 0

        # Unpause
        scheduler.set_pause_state(PauseState.UNPAUSED)
        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens > 0
        assert len(out.scheduled_new_reqs) == 1

    def test_paused_all_with_running_requests(self):
        """PAUSED_ALL prevents even running decode requests from being scheduled."""
        from vllm.v1.core.sched.interface import PauseState

        scheduler = create_scheduler()
        reqs = create_requests(num_requests=2, num_tokens=10)
        for r in reqs:
            scheduler.add_request(r)

        # Prefill both
        for _ in range(2):
            out = scheduler.schedule()
            scheduler.update_from_output(out, create_runner_output(out, 0))

        assert len(scheduler.running) == 2

        # Pause
        scheduler.set_pause_state(PauseState.PAUSED_ALL)
        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens == 0
        assert out.scheduled_cached_reqs.num_reqs == 0


# ===========================================================================
# 15. SchedulerOutput construction fields
# ===========================================================================


class TestSchedulerOutputFields:
    """Verify SchedulerOutput has all expected fields populated."""

    def test_output_fields_on_prefill(self):
        """Check all fields of SchedulerOutput during prefill."""
        scheduler = create_scheduler()
        req = create_requests(num_requests=1, num_tokens=10)[0]
        scheduler.add_request(req)

        out = scheduler.schedule()

        # Basic fields
        assert out.total_num_scheduled_tokens == 10
        assert len(out.scheduled_new_reqs) == 1
        assert out.scheduled_new_reqs[0].req_id == req.request_id
        assert out.scheduled_cached_reqs.num_reqs == 0
        assert out.num_scheduled_tokens[req.request_id] == 10
        assert isinstance(out.num_common_prefix_blocks, list)
        assert isinstance(out.preempted_req_ids, set)
        assert isinstance(out.finished_req_ids, set)
        assert out.scheduled_spec_decode_tokens == {}
        assert out.scheduled_encoder_inputs == {}

    def test_output_fields_on_decode(self):
        """Check all fields of SchedulerOutput during decode."""
        scheduler = create_scheduler()
        req = create_requests(num_requests=1, num_tokens=10)[0]
        scheduler.add_request(req)

        out = scheduler.schedule()
        scheduler.update_from_output(out, create_runner_output(out, 0))

        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens == 1
        assert len(out.scheduled_new_reqs) == 0
        assert out.scheduled_cached_reqs.num_reqs == 1
        assert out.num_scheduled_tokens[req.request_id] == 1

    def test_output_finished_req_ids(self):
        """Finished request IDs appear in the output after completion."""
        scheduler = create_scheduler()
        req = create_requests(num_requests=1, num_tokens=10, max_tokens=3)[0]
        scheduler.add_request(req)

        # Prefill
        out = scheduler.schedule()
        scheduler.update_from_output(out, create_runner_output(out, 100))

        # Decode until finished
        finished_seen = False
        for _ in range(10):
            out = scheduler.schedule()
            if req.request_id in out.finished_req_ids:
                finished_seen = True
                break
            if out.total_num_scheduled_tokens == 0:
                break
            scheduler.update_from_output(out, create_runner_output(out, 100))

        assert finished_seen, "Request was never reported as finished"


# ===========================================================================
# 16. Chunked prefill token alignment
# ===========================================================================


class TestChunkedPrefillTokenAlignment:
    """Test chunked prefill with token budget constraints."""

    def test_chunked_prefill_respects_budget(self):
        """Chunked prefill allocates exactly the token budget per step."""
        budget = 32
        scheduler = create_scheduler(
            max_num_batched_tokens=budget,
            block_size=16,
            num_blocks=1000,
        )
        # Use 64 tokens so it divides evenly into 2 chunks of 32
        req = create_requests(num_requests=1, num_tokens=64, block_size=16)[0]
        scheduler.add_request(req)

        # First chunk
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == budget
        scheduler.update_from_output(out, create_runner_output(out))

        # Second chunk
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == budget
        scheduler.update_from_output(out, create_runner_output(out, 0))

        # After full prefill, decode step
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 1

    def test_chunked_prefill_with_threshold_and_budget(self):
        """When both threshold and budget limit tokens, the smaller wins."""
        scheduler = create_scheduler(
            max_num_batched_tokens=64,
            long_prefill_token_threshold=20,
            block_size=16,
            num_blocks=1000,
        )
        req = create_requests(num_requests=1, num_tokens=100, block_size=16)[0]
        scheduler.add_request(req)

        out = scheduler.schedule()
        # threshold=20 < budget=64, so threshold wins
        assert out.num_scheduled_tokens[req.request_id] == 20


# ===========================================================================
# 17. Preemption with FCFS policy under memory pressure
# ===========================================================================


class TestPreemptionFCFS:
    """Test FCFS preemption when running requests can't allocate blocks."""

    def test_fcfs_preemption_under_memory_pressure(self):
        """Under FCFS with tight blocks, decode triggers preemption when a new
        block is needed but none are free."""
        # 8 blocks: 1 null + 7 usable.
        # 2 requests of 48 tokens = 3 blocks each = 6 blocks used. 1 free.
        # After prefill + 1 output token, num_computed=48+1=49 -> needs 4th block.
        # First request gets the 1 free block. Second needs a block -> preemption.
        scheduler = create_scheduler(
            max_num_batched_tokens=8192,
            block_size=16,
            num_blocks=8,
            enable_prefix_caching=False,
            max_num_seqs=4,
        )
        reqs = create_requests(
            num_requests=2, num_tokens=48, block_size=16, max_tokens=50
        )
        for r in reqs:
            scheduler.add_request(r)

        # Prefill both (one per step)
        for _ in range(2):
            out = scheduler.schedule()
            scheduler.update_from_output(out, create_runner_output(out, 0))

        assert len(scheduler.running) == 2

        # Decode step: each request at position 48, needs block 4.
        # Only 1 free block -> first request gets it, second is preempted.
        out = scheduler.schedule()
        assert len(out.preempted_req_ids) == 1
        assert len(scheduler.running) == 1
