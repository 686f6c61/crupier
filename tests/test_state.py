import os

import pytest

from crupier.errors import CrupierError
from crupier.state import SQLiteStateStore


def test_sqlite_state_store_creates_transitions_lists_and_events(tmp_path):
    path = tmp_path / "private" / "state.sqlite3"
    store = SQLiteStateStore(path)
    assert not path.exists()

    created = store.create(
        kind="session",
        record_id="ses_1",
        status="active",
        payload={"turns": 0},
    )
    updated = store.transition(
        kind="session",
        record_id="ses_1",
        expected_statuses={"active"},
        status="closed",
        payload={"turns": 2},
        expires_at=None,
        event="closed",
        actor="test",
    )

    assert created.version == 1
    assert updated.version == 2
    assert updated.payload == {"turns": 2}
    assert store.list("session", status="closed")[0].id == "ses_1"
    assert [item["event"] for item in store.events("session", "ses_1")] == [
        "created",
        "closed",
    ]
    if os.name == "posix":
        assert path.stat().st_mode & 0o077 == 0


def test_sqlite_state_store_rejects_duplicates_missing_and_invalid_transition(tmp_path):
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    store.create(
        kind="experiment",
        record_id="exp_1",
        status="active",
        payload={},
    )

    with pytest.raises(CrupierError, match="already exists"):
        store.create(
            kind="experiment",
            record_id="exp_1",
            status="active",
            payload={},
        )
    with pytest.raises(CrupierError, match="was not found"):
        store.get("experiment", "missing")
    with pytest.raises(CrupierError, match="expected one of"):
        store.transition(
            kind="experiment",
            record_id="exp_1",
            expected_statuses={"paused"},
            status="closed",
            payload={},
            expires_at=None,
            event="closed",
        )


def test_sqlite_state_store_rejects_symbolic_link_paths(tmp_path):
    if os.name == "nt":
        pytest.skip("Creating symbolic links may require elevated Windows privileges.")
    target = tmp_path / "target.sqlite3"
    link = tmp_path / "state.sqlite3"
    link.symlink_to(target)
    store = SQLiteStateStore(link)

    with pytest.raises(CrupierError, match="symbolic links"):
        store.list("session")


def test_state_store_fails_loudly_when_chmod_cannot_be_applied(tmp_path, monkeypatch):
    path = tmp_path / "state.sqlite3"
    real_chmod = os.chmod

    def fail_state_chmod(target, mode):
        if os.fspath(target) == os.fspath(path):
            raise OSError("chmod unavailable")
        real_chmod(target, mode)

    monkeypatch.setattr(os, "chmod", fail_state_chmod)

    with pytest.raises(CrupierError, match=r"state\.sqlite3.*mode 0600.*chmod unavailable"):
        SQLiteStateStore(path).list("session")


def test_state_store_verifies_resulting_mode_is_private(tmp_path, monkeypatch):
    path = tmp_path / "state.sqlite3"
    real_chmod = os.chmod

    def ignore_state_chmod(target, mode):
        if os.fspath(target) == os.fspath(path):
            real_chmod(target, 0o644)
            return
        real_chmod(target, mode)

    monkeypatch.setattr(os, "chmod", ignore_state_chmod)

    with pytest.raises(CrupierError, match=r"state\.sqlite3.*effective mode is 0644"):
        SQLiteStateStore(path).list("session")


def test_sqlite_state_store_rolls_back_record_when_audit_event_fails(
    tmp_path,
    monkeypatch,
):
    store = SQLiteStateStore(tmp_path / "state.sqlite3")

    def fail_event(*args, **kwargs):
        raise RuntimeError("event write failed")

    monkeypatch.setattr(SQLiteStateStore, "_append_event", fail_event)
    with pytest.raises(RuntimeError, match="event write failed"):
        store.create(
            kind="session",
            record_id="ses_atomic",
            status="active",
            payload={},
        )

    with pytest.raises(CrupierError, match="was not found"):
        store.get("session", "ses_atomic")
