"""One paid call. Runs only with TWINFIT_LIVE=1, a key and allow_spend true."""

import pytest

from twinfit.meta import smoke


@pytest.mark.live
def test_smoke_returns_ok():
    result = smoke()
    assert result["models"]
    assert "OK" in result["output_text"]
    assert result["cost_usd"] > 0
