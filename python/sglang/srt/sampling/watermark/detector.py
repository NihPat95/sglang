from __future__ import annotations

import math
from typing import Any, Optional

import msgspec
import torch

from sglang.srt.sampling.watermark.aaronson import (
    hash_contexts,
    watermark_hash32_pairs_torch,
)
from sglang.srt.sampling.watermark.request import parse_watermark_key
from sglang.srt.sampling.watermark.textseal import prf_dual

TEXTSEAL_DETECTOR_HEADER = "x-sglang-textseal-detector"
AARONSON_DETECTOR_HEADER = "x-sglang-aaronson-detector"


class WatermarkDetectorError(ValueError):
    pass


class TextSealDetectorConfig(msgspec.Struct, frozen=True, kw_only=True):
    key_a: int
    key_b: int
    ngram: int
    mixing_probability: float

    def __repr__(self) -> str:
        return (
            "TextSealDetectorConfig(key_a=<redacted>, key_b=<redacted>, "
            f"ngram={self.ngram!r}, "
            f"mixing_probability={self.mixing_probability!r})"
        )


class AaronsonDetectorConfig(msgspec.Struct, frozen=True, kw_only=True):
    key: int
    context_window: int

    def __repr__(self) -> str:
        return (
            "AaronsonDetectorConfig(key=<redacted>, "
            f"context_window={self.context_window!r})"
        )


DetectorConfig = TextSealDetectorConfig | AaronsonDetectorConfig


def parse_detector_headers(headers: Any) -> Optional[DetectorConfig]:
    textseal_value = headers.get(TEXTSEAL_DETECTOR_HEADER)
    aaronson_value = headers.get(AARONSON_DETECTOR_HEADER)
    if textseal_value is not None and aaronson_value is not None:
        raise WatermarkDetectorError("only one watermark detector header is allowed")
    if textseal_value is not None:
        return _parse_textseal_detector_header(textseal_value)
    if aaronson_value is not None:
        return _parse_aaronson_detector_header(aaronson_value)
    return None


def _parse_header_fields(value: str, allowed: set[str]) -> dict[str, str]:
    parts = [part.strip() for part in value.split(";")]
    if not parts or parts[0].lower() != "detect":
        raise WatermarkDetectorError("watermark detector header must start with detect")

    fields = {}
    for part in parts[1:]:
        if "=" not in part:
            raise WatermarkDetectorError("invalid watermark detector header")
        name, field_value = (item.strip() for item in part.split("=", 1))
        if name not in allowed or name in fields or not field_value:
            raise WatermarkDetectorError("invalid watermark detector header")
        fields[name] = field_value
    if set(fields) != allowed:
        raise WatermarkDetectorError("watermark detector header is missing fields")
    return fields


def _parse_textseal_detector_header(value: str) -> TextSealDetectorConfig:
    fields = _parse_header_fields(
        value, {"key_a", "key_b", "ngram", "mixing_probability"}
    )
    key_a = _parse_decimal_key(fields["key_a"])
    key_b = _parse_decimal_key(fields["key_b"])
    if key_a == key_b:
        raise WatermarkDetectorError("TextSeal detector keys must be distinct")
    ngram = _parse_positive_integer(fields["ngram"], maximum=10)
    try:
        mixing_probability = float(fields["mixing_probability"])
    except ValueError as error:
        raise WatermarkDetectorError(
            "TextSeal detector mixing_probability must be in [0, 1]"
        ) from error
    if not math.isfinite(mixing_probability) or not 0 <= mixing_probability <= 1:
        raise WatermarkDetectorError(
            "TextSeal detector mixing_probability must be in [0, 1]"
        )
    return TextSealDetectorConfig(
        key_a=key_a,
        key_b=key_b,
        ngram=ngram,
        mixing_probability=mixing_probability,
    )


def _parse_aaronson_detector_header(value: str) -> AaronsonDetectorConfig:
    fields = _parse_header_fields(value, {"key", "context_window"})
    try:
        key = parse_watermark_key(fields["key"])
    except ValueError as error:
        raise WatermarkDetectorError("invalid Aaronson detector key") from error
    return AaronsonDetectorConfig(
        key=key,
        context_window=_parse_positive_integer(fields["context_window"]),
    )


def _parse_decimal_key(value: str) -> int:
    try:
        key = int(value, 10)
    except ValueError as error:
        raise WatermarkDetectorError("invalid TextSeal detector key") from error
    if not -(1 << 63) <= key <= (1 << 63) - 1:
        raise WatermarkDetectorError("invalid TextSeal detector key")
    return key


def _parse_positive_integer(value: str, maximum: Optional[int] = None) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise WatermarkDetectorError(
            "watermark detector window must be a positive integer"
        ) from error
    if parsed < 1 or maximum is not None and parsed > maximum:
        raise WatermarkDetectorError(
            "watermark detector window must be a positive integer"
        )
    return parsed


def detect_watermark_token_ids(
    token_ids: list[int], config: DetectorConfig
) -> dict[str, Any]:
    if isinstance(config, TextSealDetectorConfig):
        return _detect_textseal_token_ids(token_ids, config)
    return _detect_aaronson_token_ids(token_ids, config)


def _detect_textseal_token_ids(
    token_ids: list[int], config: TextSealDetectorConfig
) -> dict[str, Any]:
    contexts = []
    observed_token_ids = []
    seen = set()
    for position in range(config.ngram + 1, len(token_ids)):
        context = tuple(token_ids[position - config.ngram : position])
        observed_token_id = token_ids[position]
        unique_key = context + (observed_token_id,)
        if unique_key in seen:
            continue
        seen.add(unique_key)
        contexts.append(context)
        observed_token_ids.append(observed_token_id)

    if not contexts:
        return _detection_result(
            provider="textseal", score=0.0, p_value=1.0, n_tokens=0, threshold=0.01
        )

    context_tensor = torch.tensor(contexts, dtype=torch.int64)
    token_tensor = torch.tensor(observed_token_ids, dtype=torch.int64).view(-1, 1)
    uniform_a, uniform_b = prf_dual(
        context_tensor,
        token_tensor,
        config.key_a,
        config.key_b,
    )
    mixing_probability = config.mixing_probability
    scores = mixing_probability * -torch.log1p(-uniform_a.double()) + (
        1 - mixing_probability
    ) * -torch.log1p(-uniform_b.double())
    score = scores.sum().item()
    base_variance = mixing_probability**2 + (1 - mixing_probability) ** 2
    p_value = _gamma_survival(
        shape=len(contexts) / base_variance,
        value=score / base_variance,
    )
    return _detection_result(
        provider="textseal",
        score=score,
        p_value=p_value,
        n_tokens=len(contexts),
        threshold=0.01,
    )


def _detect_aaronson_token_ids(
    token_ids: list[int], config: AaronsonDetectorConfig
) -> dict[str, Any]:
    contexts = []
    observed_token_ids = []
    seen = set()
    for position in range(config.context_window, len(token_ids)):
        context = tuple(token_ids[position - config.context_window : position])
        if context in seen:
            continue
        seen.add(context)
        contexts.append(context)
        observed_token_ids.append(token_ids[position])

    if not contexts:
        return _detection_result(
            provider="aaronson",
            score=0.0,
            p_value=1.0,
            n_tokens=0,
            threshold=1e-4,
        )

    context_tensor = torch.tensor(contexts, dtype=torch.int64)
    context_lengths = torch.full(
        (len(contexts),), config.context_window, dtype=torch.int32
    )
    context_hashes = hash_contexts(context_tensor, context_lengths)
    keys = torch.full((len(contexts),), config.key, dtype=torch.int64)
    observed = torch.tensor(observed_token_ids, dtype=torch.int64)
    hashed = watermark_hash32_pairs_torch(keys, context_hashes, observed)
    uniform = (hashed.double() + 0.5) / float(1 << 32)
    score = (-torch.log1p(-uniform)).sum().item()
    p_value = _gamma_survival(shape=len(contexts), value=score)
    return _detection_result(
        provider="aaronson",
        score=score,
        p_value=p_value,
        n_tokens=len(contexts),
        threshold=1e-4,
    )


def _gamma_survival(*, shape: float, value: float) -> float:
    shape_tensor = torch.tensor(shape, dtype=torch.float64)
    value_tensor = torch.tensor(value, dtype=torch.float64)
    return torch.special.gammaincc(shape_tensor, value_tensor).item()


def _detection_result(
    *,
    provider: str,
    score: float,
    p_value: float,
    n_tokens: int,
    threshold: float,
) -> dict[str, Any]:
    return {
        "object": "watermark_detection",
        "provider": provider,
        "detected": p_value < threshold,
        "p_value": p_value,
        "score": score,
        "n_tokens": n_tokens,
        "threshold": threshold,
    }
