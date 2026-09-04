from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.sampling.aaronson_selector import (
    select_aaronson_tokens_triton,
)
from sglang.srt.sampling.watermark.aaronson import (
    AaronsonWatermarkState,
    select_aaronson_tokens_torch,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestAaronsonSelector(CustomTestCase):
    def test_matches_reference_across_split_boundaries(self) -> None:
        """Blockwise selection must preserve detector-compatible score argmax."""
        generator = torch.Generator().manual_seed(7)
        batch_size = 5
        vocab_size = 16_397
        probabilities = torch.rand(batch_size, vocab_size, generator=generator)
        support_lengths = torch.tensor([1, 8_191, 8_192, 8_193, vocab_size])
        is_candidate = torch.arange(vocab_size).view(1, -1) < support_lengths.view(
            -1, 1
        )
        probabilities = torch.where(is_candidate, probabilities, 0.0)
        probabilities /= probabilities.sum(dim=-1, keepdim=True)
        context_hashes = torch.tensor(
            [0, 1, 2**31 - 1, 2**31, 2**32 - 1], dtype=torch.int64
        )
        keys = torch.tensor([0, 741852963, -(2**40), 2**39, -17])

        expected = select_aaronson_tokens_torch(probabilities, context_hashes, keys).to(
            torch.int32
        )
        actual = select_aaronson_tokens_triton(
            probabilities.cuda(), context_hashes.cuda(), keys.cuda()
        ).cpu()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fused_dense_state_matches_cpu_reference(self) -> None:
        """Fused hashing and repeat history must match the CPU state transition."""
        req_pool_indices = torch.tensor([1, 3], dtype=torch.int64)
        prompts = [[3, 4, 5], [7, 8]]

        def make_state(device: str) -> AaronsonWatermarkState:
            state = AaronsonWatermarkState(
                max_num_reqs=4,
                context_window=4,
                max_contexts_per_req=16,
                default_key=None,
                device=device,
            )
            state.init_from_prompt(req_pool_indices.to(device), prompts)
            return state

        def make_sampling_info(device: str) -> SimpleNamespace:
            return SimpleNamespace(
                aaronson_keys=torch.tensor(
                    [1234, 5678], dtype=torch.int64, device=device
                ),
                aaronson_context_windows=torch.tensor(
                    [3, 2], dtype=torch.int32, device=device
                ),
                aaronson_enabled=torch.tensor(
                    [True, False], dtype=torch.bool, device=device
                ),
                temperatures=torch.tensor([[0.8], [1.0]], device=device),
                top_ks=torch.tensor([8, 8], dtype=torch.int32, device=device),
                top_ps=torch.tensor([0.9, 1.0], device=device),
                min_ps=torch.tensor([0.05, 0.0], device=device),
            )

        cpu_state = make_state("cpu")
        cuda_state = make_state("cuda")
        logits = torch.randn(2, 257, generator=torch.Generator().manual_seed(11))
        cpu_logits = logits.clone()
        cuda_logits = logits.cuda()

        cpu_state.force(
            logits=cpu_logits,
            req_pool_indices=req_pool_indices,
            sampling_info=make_sampling_info("cpu"),
        )
        cuda_state.force(
            logits=cuda_logits,
            req_pool_indices=req_pool_indices.cuda(),
            sampling_info=make_sampling_info("cuda"),
        )

        torch.testing.assert_close(cuda_logits.cpu(), cpu_logits, rtol=0, atol=0)
        torch.testing.assert_close(
            cuda_state.num_watermarked_contexts.cpu(),
            cpu_state.num_watermarked_contexts,
        )
        torch.testing.assert_close(
            cuda_state.watermarked_context_hashes[1, :1].cpu(),
            cpu_state.watermarked_context_hashes[1, :1],
        )


if __name__ == "__main__":
    unittest.main()
