"""Errors raised by the budget-guarded Meta API client."""

from __future__ import annotations

BILLING_MESSAGE = "Add a payment method or check the key at https://dev.meta.ai/platform"


class MetaClientError(RuntimeError):
    """Base class for twinfit.meta errors."""


class MissingApiKey(MetaClientError):
    """MODEL_API_KEY is not set in the environment, ~/.config/twinfit/.env or the repo .env."""


class BillingError(MetaClientError):
    """Authentication or billing failure (401, 402, 403 or a payment/billing message)."""


class SpendNotAllowed(MetaClientError):
    """The session has allow_spend false."""


class BudgetExceeded(MetaClientError):
    """The estimate would push the global or tag total past its cap."""
