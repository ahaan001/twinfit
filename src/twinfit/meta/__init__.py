"""Budget-guarded Meta Model API client. All Meta calls in twinfit go through here."""

from twinfit.meta import budget, cache, config, ledger
from twinfit.meta.client import (
    raw_client,
    record_cost,
    responses_create,
    smoke,
    with_retries,
)
from twinfit.meta.errors import (
    BillingError,
    BudgetExceeded,
    MetaClientError,
    MissingApiKey,
    SpendNotAllowed,
)

__all__ = [
    "BillingError",
    "BudgetExceeded",
    "MetaClientError",
    "MissingApiKey",
    "SpendNotAllowed",
    "budget",
    "cache",
    "config",
    "ledger",
    "raw_client",
    "record_cost",
    "responses_create",
    "smoke",
    "with_retries",
]
