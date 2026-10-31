import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("TWINFIT_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live Meta API test; set TWINFIT_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """A throwaway data dir so ledger and cache tests never touch ~/twinfit-data."""
    monkeypatch.setenv("TWINFIT_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"
