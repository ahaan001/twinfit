"""Meta Model API endpoint, model ids and list prices.

Prices copied from docs/meta_api_notes.md (standard tier) on PRICES_COPIED. Verify against
https://dev.meta.ai/docs/pricing-rate-limits before relying on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

BASE_URL = "https://api.meta.ai/v1"
PRICES_COPIED = "2026-10-31"

MUSE_WONDER = "muse-wonder-1.0"
MUSE_SPARK = "muse-spark-1.3"
MUSE_SPARK_CONTRIBUTOR = "muse-spark-1.3-contributor"
SAM = "sam-3.1"

TOKENS_PER_MILLION = 1_000_000
CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True)
class TokenPrice:
    """USD per million tokens."""

    input: float
    cached_input: float
    output: float


TOKEN_PRICES: dict[str, TokenPrice] = {
    MUSE_WONDER: TokenPrice(input=4.00, cached_input=0.20, output=20.00),
    MUSE_SPARK: TokenPrice(input=1.25, cached_input=0.15, output=4.25),
    MUSE_SPARK_CONTRIBUTOR: TokenPrice(input=0.10, cached_input=0.002, output=0.20),
}

SAM_USD_PER_1K_IMAGES = 2.50
SAM_USD_PER_1K_VIDEO_FRAMES = 0.20
SAM_MIN_FRAMES_PER_VIDEO_REQUEST = 50


def token_price(model: str) -> TokenPrice:
    try:
        return TOKEN_PRICES[model]
    except KeyError:
        known = ", ".join(sorted(TOKEN_PRICES))
        raise ValueError(f"no token prices for model {model!r}; known: {known}") from None


def usage_dict(usage: Any) -> dict[str, int]:
    """Normalize SDK usage (pydantic model, dict or None) to plain token counts.

    `cached_tokens` counts the part of `input_tokens` served from the prompt cache.
    """
    if usage is None:
        return {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    if not isinstance(usage, dict):
        usage = usage.model_dump()
    details = usage.get("input_tokens_details") or {}
    cached = usage.get("cached_tokens", details.get("cached_tokens")) or 0
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_tokens": int(cached),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def price_usd(model: str, usage: Any) -> float:
    """Cost of a Responses call from its returned usage at list price."""
    price = token_price(model)
    counts = usage_dict(usage)
    uncached = max(counts["input_tokens"] - counts["cached_tokens"], 0)
    total = (
        uncached * price.input
        + counts["cached_tokens"] * price.cached_input
        + counts["output_tokens"] * price.output
    )
    return total / TOKENS_PER_MILLION


def estimate_usd(model: str, input_tokens: int, max_output_tokens: int) -> float:
    """Worst-case cost before the call: every input token uncached, output at the maximum."""
    price = token_price(model)
    return (input_tokens * price.input + max_output_tokens * price.output) / TOKENS_PER_MILLION


def estimate_input_tokens(text_length: int) -> int:
    """Rough token count from character count (about four characters per token)."""
    return math.ceil(text_length / CHARS_PER_TOKEN_ESTIMATE)


def sam_image_usd(n_images: int = 1) -> float:
    return n_images * SAM_USD_PER_1K_IMAGES / 1000


def sam_video_usd(n_frames: int, n_requests: int = 1) -> float:
    """Video frame cost with the 50-frame minimum applied per request."""
    billed = max(n_frames, SAM_MIN_FRAMES_PER_VIDEO_REQUEST * n_requests)
    return billed * SAM_USD_PER_1K_VIDEO_FRAMES / 1000
