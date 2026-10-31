"""Budget-guarded Meta Model API client built on the OpenAI SDK.

Every paid call goes: cache lookup, budget check, call with retries, ledger record, cache
store. Code that needs the SDK directly (file uploads, streaming) uses `raw_client()`,
`with_retries()` and `record_cost()` so the ledger stays accurate.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import openai
import pydantic
from openai import OpenAI
from openai.types.responses import Response

from twinfit import session
from twinfit.meta import budget, config, ledger
from twinfit.meta import cache as response_cache
from twinfit.meta.errors import BILLING_MESSAGE, BillingError, MissingApiKey

T = TypeVar("T")

MAX_ATTEMPTS = 6
BASE_DELAY_S = 1.0
BACKOFF_FACTOR = 2.0
BILLING_STATUSES = frozenset({401, 402, 403})
_BILLING_WORDS = re.compile(r"payment|billing", re.IGNORECASE)

# Injectable for tests so backoff runs instantly and deterministically.
_sleep: Callable[[float], None] = time.sleep
_uniform: Callable[[float, float], float] = random.uniform


def raw_client() -> OpenAI:
    """OpenAI SDK client pointed at Meta with SDK retries off (this module does its own)."""
    key = session.model_api_key()
    if not key:
        raise MissingApiKey(
            "MODEL_API_KEY is not set; put it in ~/.config/twinfit/.env "
            "(create a key at https://dev.meta.ai/platform)"
        )
    return OpenAI(base_url=config.BASE_URL, api_key=key, max_retries=0)


def _is_billing(exc: openai.APIError) -> bool:
    status = getattr(exc, "status_code", None)
    return status in BILLING_STATUSES or bool(_BILLING_WORDS.search(str(exc)))


def _is_retryable(exc: openai.APIError) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or (status is not None and 500 <= status < 600)


def retry_after_seconds(exc: openai.APIError) -> float | None:
    """Parse Retry-After as seconds or an HTTP date; None when absent or unparseable."""
    response = getattr(exc, "response", None)
    raw = response.headers.get("retry-after") if response is not None else None
    if not raw:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


def backoff_delay(attempt: int, retry_after: float | None) -> float:
    """Full jitter: uniform in [0, base * factor**attempt], unless the server set Retry-After."""
    if retry_after is not None:
        return retry_after
    return _uniform(0.0, BASE_DELAY_S * BACKOFF_FACTOR**attempt)


def with_retries(call: Callable[[], T]) -> T:
    """Run `call`, retrying 429 and 5xx up to MAX_ATTEMPTS; billing failures raise at once."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call()
        except openai.APIError as exc:
            if _is_billing(exc):
                raise BillingError(BILLING_MESSAGE) from exc
            if not _is_retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                raise
            _sleep(backoff_delay(attempt, retry_after_seconds(exc)))
    raise AssertionError("unreachable")


def record_cost(
    model: str,
    kind: str,
    usage: Any,
    cost_usd: float,
    cache_hit: bool,
    request_id: str | None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Append one ledger record; `tag` defaults to the session tag."""
    return ledger.append_record(
        model=model,
        kind=kind,
        usage=usage,
        cost_usd=cost_usd,
        cache_hit=cache_hit,
        request_id=request_id,
        tag=tag,
    )


def _response_from_json(data: dict[str, Any]) -> Response:
    # The server may omit fields the SDK schema marks required; the SDK itself parses
    # leniently with construct(), so fall back to that.
    try:
        return Response.model_validate(data)
    except pydantic.ValidationError:
        return Response.construct(**data)


def _estimate_input_tokens(input: Any) -> int:
    text = input if isinstance(input, str) else json.dumps(response_cache.canonicalize(input))
    return config.estimate_input_tokens(len(text))


def responses_create(
    model: str,
    input: Any,
    *,
    max_output_tokens: int,
    estimate_usd: float | None = None,
    cache: bool = True,
    tag: str | None = None,
    **kwargs: Any,
) -> Response:
    """Cached, budget-checked, retried `responses.create`.

    The default estimate prices every input token uncached and the full max_output_tokens.
    Pass `estimate_usd` for image or video inputs. Streaming is not supported here; use
    `raw_client()` with `record_cost()` instead. Cache hits are reconstructed as Response
    objects and recorded at zero cost.
    """
    if kwargs.get("stream"):
        raise ValueError("responses_create does not stream; use raw_client() and record_cost()")
    request = {"input": input, "max_output_tokens": max_output_tokens, **kwargs}
    tag = tag or session.load_session().tag
    cache_key = response_cache.key(model, request)
    if cache:
        cached = response_cache.get(cache_key)
        if cached is not None:
            response = _response_from_json(cached)
            record_cost(model, "responses", response.usage, 0.0, True, response.id, tag)
            return response
    if estimate_usd is None:
        estimate_usd = config.estimate_usd(model, _estimate_input_tokens(input), max_output_tokens)
    budget.check(estimate_usd, tag)
    client = raw_client()
    response = with_retries(lambda: client.responses.create(model=model, **request))
    cost = config.price_usd(model, response.usage)
    record_cost(model, "responses", response.usage, cost, False, response.id, tag)
    if cache:
        response_cache.put(cache_key, response.model_dump(mode="json"))
    return response


def smoke(tag: str | None = None) -> dict[str, Any]:
    """List models, then ask muse-spark-1.3 for the single word OK (uncached, recorded)."""
    client = raw_client()
    models = [m.id for m in with_retries(client.models.list)]
    response = responses_create(
        config.MUSE_SPARK,
        "Reply with the single word OK and nothing else.",
        max_output_tokens=256,
        cache=False,
        tag=tag,
    )
    return {
        "models": models,
        "output_text": response.output_text,
        "cost_usd": config.price_usd(config.MUSE_SPARK, response.usage),
    }
