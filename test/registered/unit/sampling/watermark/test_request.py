import unittest
from types import SimpleNamespace

from starlette.datastructures import Headers

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.request_headers import apply_watermark_request
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.sampling.watermark import (
    TextSealConfig,
    WatermarkRegistry,
    WatermarkRequestConfig,
    WatermarkRequestError,
    normalize_watermark_request,
    parse_watermark_header,
)
from sglang.srt.sampling.watermark.detector import (
    AaronsonDetectorConfig,
    TextSealDetectorConfig,
    WatermarkDetectorError,
    detect_watermark_token_ids,
    parse_detector_headers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestWatermarkRequest(CustomTestCase):
    def test_detector_header_contract(self):
        secret = "741852963"
        textseal = parse_detector_headers(
            Headers(
                {
                    "X-SGLang-TextSeal-Detector": (
                        f"detect;key_a={secret};key_b=963852741;"
                        "ngram=3;mixing_probability=0.5"
                    )
                }
            )
        )
        self.assertEqual(
            textseal,
            TextSealDetectorConfig(
                key_a=741852963,
                key_b=963852741,
                ngram=3,
                mixing_probability=0.5,
            ),
        )
        self.assertNotIn(secret, repr(textseal))
        self.assertEqual(
            parse_detector_headers(
                Headers(
                    {
                        "X-SGLang-Aaronson-Detector": (
                            "detect;key=0123456789abcdef;context_window=4"
                        )
                    }
                )
            ),
            AaronsonDetectorConfig(key=0x0123456789ABCDEF, context_window=4),
        )

    def test_detector_scores_match_reference_vectors(self):
        token_ids = list(range(32))
        textseal = detect_watermark_token_ids(
            token_ids,
            TextSealDetectorConfig(
                key_a=741852963,
                key_b=963852741,
                ngram=3,
                mixing_probability=0.5,
            ),
        )
        aaronson = detect_watermark_token_ids(
            token_ids,
            AaronsonDetectorConfig(
                key=0x0123456789ABCDEF,
                context_window=4,
            ),
        )

        self.assertEqual(textseal["n_tokens"], 28)
        self.assertAlmostEqual(textseal["p_value"], 0.5931171696424633)
        self.assertEqual(aaronson["n_tokens"], 28)
        self.assertAlmostEqual(aaronson["p_value"], 0.2070135124329177)

    def test_detector_header_errors_do_not_echo_keys(self):
        secret = "do-not-log"
        cases = [
            f"detect;key_a={secret};key_b=2;ngram=3;mixing_probability=0.5",
            "detect;key_a=1;key_b=2;ngram=3",
        ]
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(WatermarkDetectorError) as context:
                    parse_detector_headers(
                        Headers({"X-SGLang-TextSeal-Detector": value})
                    )
                self.assertNotIn(secret, str(context.exception))

    def test_header_contract(self):
        self.assertIsNone(parse_watermark_header(None))
        self.assertIsNone(parse_watermark_header(" off "))
        self.assertEqual(
            parse_watermark_header("textseal"),
            WatermarkRequestConfig(provider="textseal"),
        )
        self.assertEqual(
            parse_watermark_header("aaronson"),
            WatermarkRequestConfig(provider="aaronson"),
        )

    def test_rejects_unknown_provider_and_malformed_header(self):
        cases = [
            ("unknown", "watermark_provider_unknown"),
            ("textseal;key_a=741852963", "watermark_invalid_request"),
            ("textseal;profile=default", "watermark_invalid_request"),
        ]
        for value, code in cases:
            with self.subTest(value=value):
                with self.assertRaises(WatermarkRequestError) as context:
                    parse_watermark_header(value)
                self.assertEqual(context.exception.code, code)
                self.assertNotIn("741852963", str(context.exception))

    def test_header_exclusively_controls_generation_watermark(self):
        request = SimpleNamespace(
            watermark={"provider": "textseal", "key_a": "741852963"}
        )
        apply_watermark_request(request, Headers())
        self.assertIsNone(request.watermark)

        apply_watermark_request(request, Headers({"x-sglang-watermark": "textseal"}))
        self.assertEqual(request.watermark.provider, "textseal")

    def test_openai_request_schemas_do_not_expose_watermark(self):
        self.assertNotIn("watermark", CompletionRequest.model_fields)
        self.assertNotIn("watermark", ChatCompletionRequest.model_fields)

    def test_body_watermark_config_is_rejected_without_echoing_key(self):
        secret = "fedcba9876543210"
        with self.assertRaises(WatermarkRequestError) as context:
            normalize_watermark_request({"key": secret, "context_window": 4})
        self.assertNotIn(secret, str(context.exception))

    def test_batch_normalization_and_splitting(self):
        request = GenerateReqInput(
            text=["first", "second"],
            watermark=[
                WatermarkRequestConfig(provider="textseal"),
                None,
            ],
        )
        request.normalize_batch_and_arguments()

        self.assertEqual(request[0].watermark.provider, "textseal")
        self.assertIsNone(request[1].watermark)

    def test_registry_resolves_request_errors_without_secrets(self):
        config = TextSealConfig(key_a=741852963, key_b=963852741)
        registry = WatermarkRegistry(textseal=config)
        request = WatermarkRequestConfig(provider="textseal")
        self.assertIs(registry.resolve_request(request), config)

        cases = [
            (
                WatermarkRegistry(),
                request,
                "watermark_disabled",
            ),
            (
                registry,
                WatermarkRequestConfig(provider="unknown"),
                "watermark_provider_unknown",
            ),
        ]
        for candidate_registry, candidate_request, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(WatermarkRequestError) as context:
                    candidate_registry.resolve_request(candidate_request)
                self.assertEqual(context.exception.code, code)
                self.assertNotIn("741852963", str(context.exception))


if __name__ == "__main__":
    unittest.main()
