from twinfit import paths, session


def test_data_dir_env_and_subfolders(isolated_data_dir):
    d = paths.data_dir()
    assert d == isolated_data_dir
    for sub in paths.DATA_SUBDIRS:
        assert (d / sub).is_dir()
    assert paths.ledger_path() == d / "cost_ledger.jsonl"


def test_session_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("TWINFIT_TAG", raising=False)
    monkeypatch.delenv("TWINFIT_ALLOW_SPEND", raising=False)
    s = session.load_session(tmp_path)
    assert s == session.Session(tag="untagged", tag_cap_usd=None, allow_spend=False)


def test_session_file_and_env_override(tmp_path, monkeypatch):
    (tmp_path / ".twinfit-session.yaml").write_text(
        "tag: m1-dev\ntag_cap_usd: 45\nallow_spend: true\n"
    )
    monkeypatch.delenv("TWINFIT_TAG", raising=False)
    monkeypatch.delenv("TWINFIT_ALLOW_SPEND", raising=False)
    s = session.load_session(tmp_path)
    assert s.tag == "m1-dev" and s.tag_cap_usd == 45.0 and s.allow_spend is True
    monkeypatch.setenv("TWINFIT_TAG", "other")
    monkeypatch.setenv("TWINFIT_ALLOW_SPEND", "0")
    s = session.load_session(tmp_path)
    assert s.tag == "other" and s.allow_spend is False


def test_repo_root_is_git_toplevel():
    root = paths.repo_root()
    assert (root / "pyproject.toml").exists()
