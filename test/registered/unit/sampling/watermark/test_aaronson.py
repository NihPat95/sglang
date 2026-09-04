import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.watermark.aaronson import (
    AaronsonWatermarkState,
    build_aaronson_batch_config,
    hash_contexts,
)
from sglang.srt.sampling.watermark.request import WatermarkRequestConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAaronsonWatermarkState(CustomTestCase):
    def test_startup_key_only_enables_header_selected_requests(self):
        requests = [
            SimpleNamespace(watermark=None),
            SimpleNamespace(watermark=WatermarkRequestConfig(provider="aaronson")),
            SimpleNamespace(watermark=WatermarkRequestConfig(provider="textseal")),
        ]
        keys, context_windows, enabled = build_aaronson_batch_config(
            requests,
            default_key="0123456789abcdef",
            default_context_window=4,
            device="cpu",
        )

        self.assertEqual(keys.tolist(), [0, 0x0123456789ABCDEF, 0])
        self.assertEqual(context_windows.tolist(), [4, 4, 4])
        self.assertEqual(enabled.tolist(), [False, True, False])

    def test_speculative_state_commits_only_accept_path(self):
        """Branch-local contexts and rejected tokens must not enter request state."""
        state = AaronsonWatermarkState(
            max_num_reqs=4,
            context_window=3,
            max_contexts_per_req=8,
            default_key=None,
            device="cpu",
        )
        req_pool_indices = torch.tensor([1], dtype=torch.int64)
        state.init_from_prompt(req_pool_indices, [[10, 11, 12]])

        contexts, context_lengths = state.speculative_contexts(
            req_pool_indices=req_pool_indices,
            draft_tokens=torch.tensor([12, 20, 30, 40]),
            custom_mask=torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, False, True],
                ]
            ).flatten(),
            positions=torch.tensor([3, 4, 5, 5]),
            draft_token_num=4,
            full_mask=False,
            context_windows=torch.tensor([3], dtype=torch.int32),
        )

        self.assertEqual(context_lengths.tolist(), [3, 3, 3, 3])
        self.assertEqual(
            contexts.tolist(),
            [[10, 11, 12], [11, 12, 20], [12, 20, 30], [12, 20, 40]],
        )

        context_hashes = hash_contexts(contexts, context_lengths)
        state.record_speculative(
            req_pool_indices=req_pool_indices,
            context_hashes=context_hashes,
            selected=torch.ones(4, dtype=torch.bool),
            accept_index=torch.tensor([[0, 1, -1, -1]], dtype=torch.int32),
            accept_lens=torch.tensor([2], dtype=torch.int32),
        )
        state.append_speculative(
            req_pool_indices=req_pool_indices,
            accept_tokens=torch.tensor([[101, 102, -1, -1]], dtype=torch.int32),
            accept_lens=torch.tensor([2], dtype=torch.int32),
        )

        self.assertEqual(state.num_watermarked_contexts[1].item(), 2)
        self.assertEqual(
            state.watermarked_context_hashes[1, :2].tolist(),
            context_hashes[:2].to(torch.int32).tolist(),
        )
        tail, lengths = state.contexts_tail(req_pool_indices)
        self.assertEqual(lengths.tolist(), [3])
        self.assertEqual(tail.tolist(), [[12, 101, 102]])


if __name__ == "__main__":
    unittest.main()
