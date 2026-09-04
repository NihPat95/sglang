from typing import Any, Optional

import msgspec


class WatermarkRequestConfig(msgspec.Struct, frozen=True, kw_only=True):
    provider: str


class WatermarkRequestError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalize_watermark_request(value: Any) -> Optional[WatermarkRequestConfig]:
    if value is None or isinstance(value, WatermarkRequestConfig):
        return value
    raise WatermarkRequestError(
        "watermark_invalid_request",
        "watermark generation must be selected with the X-SGLang-Watermark header",
    )


def parse_watermark_key(value: Any) -> int:
    if not isinstance(value, str):
        raise WatermarkRequestError(
            "watermark_invalid_request", "watermark key must be a hex string"
        )
    digits = value[2:] if value.lower().startswith("0x") else value
    if not 1 <= len(digits) <= 16:
        raise WatermarkRequestError(
            "watermark_invalid_request",
            "watermark key must contain 1 to 16 hex digits",
        )
    if any(character not in "0123456789abcdefABCDEF" for character in digits):
        raise WatermarkRequestError(
            "watermark_invalid_request", "watermark key must contain only hex digits"
        )
    key = int(digits, 16)
    return key if key < (1 << 63) else key - (1 << 64)


def parse_watermark_header(value: Optional[str]) -> Optional[WatermarkRequestConfig]:
    if value is None:
        return None

    value = value.strip()
    if value == "off":
        return None

    if value not in {"textseal", "aaronson"}:
        if value.split(";", 1)[0] not in {"off", "textseal", "aaronson"}:
            raise WatermarkRequestError(
                "watermark_provider_unknown", "unknown watermark provider"
            )
        raise WatermarkRequestError(
            "watermark_invalid_request", "invalid watermark request header"
        )

    return WatermarkRequestConfig(provider=value)
