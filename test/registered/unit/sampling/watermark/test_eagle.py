import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.sampling.watermark import WatermarkBatchInfo
from sglang.srt.speculative.eagle_utils import _apply_verifier_only_watermark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEagleWatermark(CustomTestCase):
    def test_verifier_replaces_only_bonus_and_advances_overlap_context(self):
        """Verifier watermarking must not alter native draft acceptance."""
        watermark = WatermarkBatchInfo(
            enabled=torch.tensor([True, True]),
            all_enabled=True,
            key_a=torch.tensor([11, 12]),
            key_b=torch.tensor([21, 22]),
            mixing_probabilities=torch.tensor([0.5, 0.5]),
            ngrams=torch.tensor([2, 2], dtype=torch.int32),
            contexts=torch.tensor([[1, 2], [3, 4]]),
            nonces=torch.tensor([31, 32]),
        )
        sampling_info = SimpleNamespace(
            watermark=watermark,
            temperatures=torch.ones((2, 1)),
            top_ks=torch.tensor([4, 4]),
            top_ps=torch.ones(2),
            min_ps=torch.zeros(2),
            watermark_max_top_k=4,
        )
        batch = SimpleNamespace(
            sampling_info=sampling_info,
            seq_lens=torch.tensor([20, 30]),
            enable_overlap=True,
        )
        verify_input = SimpleNamespace(
            draft_token=torch.tensor([7, 70, 71, 72, 8, 80, 81, 82]),
            draft_token_num=4,
            positions=torch.arange(100, 108),
        )
        logits = torch.arange(32, dtype=torch.float32).view(8, 4)
        predict = torch.tensor([11, 12, 90, -1, 21, 91, -1, -1], dtype=torch.int32)
        num_correct_drafts = torch.tensor([2, 1], dtype=torch.int32)
        accept_index = torch.tensor([[0, 1, 2], [4, 5, -1]], dtype=torch.int32)
        original_num_correct_drafts = num_correct_drafts.clone()
        original_accept_index = accept_index.clone()

        with patch(
            "sglang.srt.layers.sampler.sample_textseal_from_probs",
            return_value=torch.tensor([3, 2], dtype=torch.int32),
        ) as selector:
            watermarked_predict = _apply_verifier_only_watermark(
                verify_input,
                batch,
                logits,
                predict,
                num_correct_drafts,
                accept_index,
            )

        self.assertEqual(watermarked_predict.tolist(), [11, 12, 3, -1, 21, 2, -1, -1])
        torch.testing.assert_close(num_correct_drafts, original_num_correct_drafts)
        torch.testing.assert_close(accept_index, original_accept_index)
        self.assertEqual(watermark.contexts.tolist(), [[11, 12], [8, 21]])
        self.assertEqual(
            selector.call_args.args[4].contexts.tolist(), [[11, 12], [8, 21]]
        )
        self.assertEqual(selector.call_args.args[5].tolist(), [102, 105])


if __name__ == "__main__":
    unittest.main()
