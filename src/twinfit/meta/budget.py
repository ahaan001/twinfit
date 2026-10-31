"""Spend gate consulted before every paid call."""

from __future__ import annotations

from twinfit import session
from twinfit.meta import ledger
from twinfit.meta.errors import BudgetExceeded, SpendNotAllowed

ENABLE_HINT = (
    "set allow_spend: true in .twinfit-session.yaml at the worktree root, "
    "or export TWINFIT_ALLOW_SPEND=1"
)


def check(estimate_usd: float, tag: str | None = None) -> None:
    """Raise unless `estimate_usd` fits under the global cap and the session's tag cap.

    `tag` defaults to the session tag. The session's `tag_cap_usd` applies to whichever
    tag is checked.
    """
    current = session.load_session()
    tag = tag or current.tag
    if not current.allow_spend:
        raise SpendNotAllowed(f"spending is disabled for tag {tag!r}; to enable it {ENABLE_HINT}")
    cap = session.budget_usd()
    spent = ledger.total()
    if spent + estimate_usd > cap:
        raise BudgetExceeded(
            f"global budget: spent ${spent:.4f} + estimate ${estimate_usd:.4f} "
            f"exceeds TWINFIT_BUDGET_USD ${cap:.2f}"
        )
    if current.tag_cap_usd is not None:
        tag_spent = ledger.total(tag)
        if tag_spent + estimate_usd > current.tag_cap_usd:
            raise BudgetExceeded(
                f"tag {tag!r}: spent ${tag_spent:.4f} + estimate ${estimate_usd:.4f} "
                f"exceeds tag_cap_usd ${current.tag_cap_usd:.2f}"
            )
