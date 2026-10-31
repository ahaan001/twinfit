"""Unit tests for twinfit.meta with the OpenAI SDK mocked. No network."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx2
import openai
import pytest
from openai.types.responses import Response

from twinfit import session
from twinfit.meta import (
    BillingError,
    BudgetExceeded,
    MissingApiKey,
    SpendNotAllowed,
    budget,
    cache,
    client,
    config,
    ledger,
)

MODEL = config.MUSE_SPARK


def make_response(text: str = "OK", input_tokens: int = 1000, output_tokens: int = 200) -> Response:
    return Response.model_validate(
        {
            "id": "resp_test",
            "created_at": 1.0,
            "model": MODEL,
            "object": "response",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


def status_error(status: int, message: str = "error", headers: dict | None = None):
    response = httpx2.Response(
        status, headers=headers or {}, request=httpx2.Request("POST", "https://test/responses")
    )
    classes = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
        503: openai.InternalServerError,
    }
    return classes.get(status, openai.APIStatusError)(message, response=response, body=None)


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.responses = FakeResponses(outcomes)


@pytest.fixture
def spend_session(monkeypatch, isolated_data_dir):
    """Spending allowed, fake key, no sleeping, deterministic jitter at the cap."""
    state = {"session": session.Session(tag="test", tag_cap_usd=None, allow_spend=True)}
    monkeypatch.setattr(session, "load_session", lambda root=None: state["session"])
    monkeypatch.setattr(session, "model_api_key", lambda: "LLM_test_fake")
    monkeypatch.setattr(session, "budget_usd", lambda: 120.0)
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", sleeps.append)
    monkeypatch.setattr(client, "_uniform", lambda lo, hi: hi)
    state["sleeps"] = sleeps
    return state


def install_client(monkeypatch, outcomes) -> FakeClient:
    fake = FakeClient(outcomes)
    monkeypatch.setattr(client, "raw_client", lambda: fake)
    return fake


# --- config ---------------------------------------------------------------------------


def test_price_usd_splits_cached_input():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "input_tokens_details": {"cached_tokens": 400_000},
    }
    expected = 600_000 * 1.25 / 1e6 + 400_000 * 0.15 / 1e6 + 4.25
    assert config.price_usd(MODEL, usage) == pytest.approx(expected)
    assert config.price_usd(config.MUSE_WONDER, {"input_tokens": 1e6, "output_tokens": 0}) == 4.0
    with pytest.raises(ValueError):
        config.price_usd("unknown-model", usage)


def test_sam_pricing_has_frame_minimum():
    assert config.sam_image_usd(4) == pytest.approx(0.01)
    assert config.sam_video_usd(10) == pytest.approx(50 * 0.20 / 1000)
    assert config.sam_video_usd(300) == pytest.approx(300 * 0.20 / 1000)
    assert config.sam_video_usd(30, n_requests=3) == pytest.approx(150 * 0.20 / 1000)


# --- retry and billing ------------------------------------------------------------------


def test_retry_backoff_and_retry_after(monkeypatch, spend_session):
    retry_date = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    fake = install_client(
        monkeypatch,
        [
            status_error(429, headers={"Retry-After": "3"}),
            status_error(429, headers={"Retry-After": retry_date}),
            status_error(503),
            status_error(500),
            make_response("OK"),
        ],
    )
    response = client.responses_create(MODEL, "say OK", max_output_tokens=16)
    assert response.output_text == "OK"
    assert len(fake.responses.calls) == 5
    sleeps = spend_session["sleeps"]
    assert sleeps[0] == 3.0
    assert 28.0 <= sleeps[1] <= 30.0
    # Full jitter with the random draw pinned to the cap: base 1 s, factor 2.
    assert sleeps[2:] == [4.0, 8.0]


def test_retry_gives_up_after_six_attempts(monkeypatch, spend_session):
    fake = install_client(monkeypatch, [status_error(503) for _ in range(6)])
    with pytest.raises(openai.InternalServerError):
        client.responses_create(MODEL, "say OK", max_output_tokens=16)
    assert len(fake.responses.calls) == 6
    assert spend_session["sleeps"] == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_non_retryable_status_raises_immediately(monkeypatch, spend_session):
    fake = install_client(monkeypatch, [status_error(400, "bad input")])
    with pytest.raises(openai.BadRequestError):
        client.responses_create(MODEL, "say OK", max_output_tokens=16)
    assert len(fake.responses.calls) == 1
    assert spend_session["sleeps"] == []


@pytest.mark.parametrize(
    "error",
    [
        status_error(401, "invalid key"),
        status_error(402, "payment required"),
        status_error(403, "forbidden"),
        status_error(400, "Your billing account is not active"),
    ],
)
def test_billing_errors_never_retry(monkeypatch, spend_session, error):
    fake = install_client(monkeypatch, [error, make_response()])
    with pytest.raises(BillingError, match="https://dev.meta.ai/platform"):
        client.responses_create(MODEL, "say OK", max_output_tokens=16)
    assert len(fake.responses.calls) == 1
    assert spend_session["sleeps"] == []
    assert ledger.read_records() == []


def test_raw_client_requires_key(monkeypatch, isolated_data_dir):
    monkeypatch.setattr(session, "model_api_key", lambda: None)
    with pytest.raises(MissingApiKey):
        client.raw_client()


def test_raw_client_configuration(monkeypatch, isolated_data_dir):
    monkeypatch.setattr(session, "model_api_key", lambda: "LLM_test_fake")
    c = client.raw_client()
    assert str(c.base_url) == config.BASE_URL + "/"
    assert c.max_retries == 0


# --- budget ----------------------------------------------------------------------------


def test_budget_refuses_when_spend_not_allowed(monkeypatch, isolated_data_dir):
    monkeypatch.setattr(
        session, "load_session", lambda root=None: session.Session(tag="t", allow_spend=False)
    )
    with pytest.raises(SpendNotAllowed, match="TWINFIT_ALLOW_SPEND"):
        budget.check(0.01, "t")


def test_budget_global_cap(monkeypatch, spend_session):
    monkeypatch.setattr(session, "budget_usd", lambda: 1.0)
    ledger.append_record(
        model=MODEL, kind="other", usage=None, cost_usd=0.95, cache_hit=False, request_id=None
    )
    budget.check(0.04, "test")
    with pytest.raises(BudgetExceeded, match="global budget"):
        budget.check(0.06, "test")


def test_budget_tag_cap(monkeypatch, spend_session):
    spend_session["session"] = session.Session(tag="m1", tag_cap_usd=0.5, allow_spend=True)
    for tag, cost in (("m1", 0.45), ("other", 10.0)):
        ledger.append_record(
            model=MODEL,
            kind="other",
            usage=None,
            cost_usd=cost,
            cache_hit=False,
            request_id=None,
            tag=tag,
        )
    budget.check(0.04, "m1")
    with pytest.raises(BudgetExceeded, match="tag 'm1'"):
        budget.check(0.06, "m1")


def test_responses_create_checks_budget_before_calling(monkeypatch, spend_session):
    spend_session["session"] = session.Session(tag="test", allow_spend=False)
    fake = install_client(monkeypatch, [make_response()])
    with pytest.raises(SpendNotAllowed):
        client.responses_create(MODEL, "say OK", max_output_tokens=16)
    assert fake.responses.calls == []


# --- cache and ledger -------------------------------------------------------------------


def test_cache_hit_records_zero_cost(monkeypatch, spend_session, isolated_data_dir):
    fake = install_client(monkeypatch, [make_response("OK", 1000, 200)])
    first = client.responses_create(MODEL, "say OK", max_output_tokens=16)
    second = client.responses_create(MODEL, "say OK", max_output_tokens=16)
    assert len(fake.responses.calls) == 1
    assert second.output_text == first.output_text == "OK"
    assert second.usage.input_tokens == 1000
    records = ledger.read_records()
    assert len(records) == 2
    paid, hit = records
    assert paid["cost_usd"] == pytest.approx((1000 * 1.25 + 200 * 4.25) / 1e6)
    assert paid["cache_hit"] is False and paid["kind"] == "responses"
    assert hit["cost_usd"] == 0.0 and hit["cache_hit"] is True
    assert hit["request_id"] == "resp_test" and hit["tag"] == "test"
    assert hit["usage"] == {"input_tokens": 1000, "cached_tokens": 0, "output_tokens": 200}
    assert len(list((isolated_data_dir / "cache" / "meta").glob("*.json"))) == 1
    assert ledger.total() == pytest.approx(paid["cost_usd"])


def test_cache_key_replaces_file_content():
    blob = "data:image/png;base64," + "A" * 5000
    request_a = {"input": [{"type": "input_image", "image_url": blob}], "max_output_tokens": 8}
    request_b = {
        "input": [{"type": "input_image", "image_url": blob + "B"}],
        "max_output_tokens": 8,
    }
    assert cache.key(MODEL, request_a) == cache.key(MODEL, dict(reversed(list(request_a.items()))))
    assert cache.key(MODEL, request_a) != cache.key(MODEL, request_b)
    assert cache.key(MODEL, request_a) != cache.key(config.MUSE_WONDER, request_a)
    assert len(cache.canonical_json(request_a)) < 200


def test_cache_miss_bypassed_when_disabled(monkeypatch, spend_session):
    fake = install_client(monkeypatch, [make_response(), make_response()])
    client.responses_create(MODEL, "say OK", max_output_tokens=16, cache=False)
    client.responses_create(MODEL, "say OK", max_output_tokens=16, cache=False)
    assert len(fake.responses.calls) == 2
    assert all(r["cache_hit"] is False for r in ledger.read_records())


def test_ledger_summary(spend_session):
    ledger.append_record(
        model=MODEL,
        kind="responses",
        usage=None,
        cost_usd=0.5,
        cache_hit=False,
        request_id="a",
        tag="m1",
    )
    ledger.append_record(
        model=config.SAM,
        kind="sam_image",
        usage=None,
        cost_usd=0.25,
        cache_hit=False,
        request_id="b",
        tag="m2",
    )
    s = ledger.summary()
    assert s["total_usd"] == pytest.approx(0.75)
    assert s["by_tag"] == {"m1": 0.5, "m2": 0.25}
    assert s["by_model"] == {MODEL: 0.5, config.SAM: 0.25}
    today = datetime.now(UTC).date().isoformat()
    assert s["by_day"] == {today: pytest.approx(0.75)}
    assert ledger.total("m1") == 0.5
    with pytest.raises(ValueError):
        ledger.append_record(
            model=MODEL, kind="bogus", usage=None, cost_usd=0, cache_hit=False, request_id=None
        )


WRITER = """
import sys
from twinfit.meta import ledger
for i in range(50):
    ledger.append_record(model="m", kind="other", usage={"input_tokens": i},
                         cost_usd=0.001, cache_hit=False, request_id=f"{sys.argv[1]}-{i}",
                         tag="concurrency")
"""


def test_ledger_concurrent_appends_from_two_processes(isolated_data_dir, monkeypatch):
    monkeypatch.setenv("TWINFIT_TAG", "concurrency")
    procs = [
        subprocess.Popen([sys.executable, "-c", WRITER, name], stderr=subprocess.PIPE)
        for name in ("p1", "p2")
    ]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode()
    lines = ledger_lines(isolated_data_dir)
    assert len(lines) == 100
    records = [json.loads(line) for line in lines]
    ids = {r["request_id"] for r in records}
    assert len(ids) == 100
    assert ledger.total("concurrency") == pytest.approx(0.1)


def ledger_lines(data_dir) -> list[str]:
    text = (data_dir / "cost_ledger.jsonl").read_text()
    assert text.endswith("\n")
    return text.splitlines()
