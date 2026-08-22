import os
from types import SimpleNamespace

import pytest

import crupier.state as state_module
from crupier.errors import CrupierError
from crupier.state import (
    SQLiteStateStore,
    ensure_private_directory,
    private_append_text,
    private_write_text,
)


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


def test_private_paths_reject_symlink_and_non_directory_or_file_targets(tmp_path, monkeypatch):
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(tmp_path / "missing-directory")
    with pytest.raises(CrupierError, match="cannot be a symbolic link"):
        ensure_private_directory(directory_link)

    missing_directory = tmp_path / "missing-directory"
    with monkeypatch.context() as patcher:
        patcher.setattr(state_module.Path, "mkdir", lambda self, **kwargs: None)
        with pytest.raises(CrupierError, match="must be a regular directory"):
            ensure_private_directory(missing_directory)

    parent = tmp_path / "private"
    parent.mkdir()
    target_link = parent / "target-link"
    target_link.symlink_to(parent / "missing-target")
    with pytest.raises(CrupierError, match="target.*symbolic link"):
        private_write_text(target_link, "secret")

    target_directory = parent / "target-directory"
    target_directory.mkdir()
    with pytest.raises(CrupierError, match="must be a regular file"):
        private_write_text(target_directory, "secret")


def test_private_write_closes_descriptor_when_wrapping_it_fails(tmp_path, monkeypatch):
    target = tmp_path / "private" / "artifact.json"
    real_close = os.close
    closed = []

    def fail_fdopen(descriptor, *args, **kwargs):
        del args, kwargs
        raise OSError(f"cannot wrap {descriptor}")

    def record_close(descriptor):
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(state_module.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(state_module.os, "close", record_close)

    with pytest.raises(OSError, match="cannot wrap"):
        private_write_text(target, "secret")
    assert len(closed) == 1


def test_private_append_rejects_non_regular_existing_target(tmp_path, monkeypatch):
    target = tmp_path / "private" / "events.jsonl"
    private_write_text(target, "existing\n")
    monkeypatch.setattr(state_module.stat, "S_ISREG", lambda mode: False)

    with pytest.raises(CrupierError, match="must be a regular file"):
        private_append_text(target, "new\n")


def test_state_transition_rejects_missing_and_lost_update(tmp_path, monkeypatch):
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    with pytest.raises(CrupierError, match="was not found"):
        store.transition(
            kind="session",
            record_id="missing",
            expected_statuses={"active"},
            status="closed",
            payload={},
            expires_at=None,
            event="closed",
        )

    store.create(kind="session", record_id="ses_1", status="active", payload={})
    real_connection = store._raw_connect()

    class LostUpdateConnection:
        def execute(self, sql, parameters=()):
            if sql.lstrip().startswith("UPDATE records"):
                return SimpleNamespace(rowcount=0)
            return real_connection.execute(sql, parameters)

        @property
        def in_transaction(self):
            return real_connection.in_transaction

        def rollback(self):
            return real_connection.rollback()

        def commit(self):
            return real_connection.commit()

        def close(self):
            return real_connection.close()

    monkeypatch.setattr(store, "_raw_connect", lambda: LostUpdateConnection())
    with pytest.raises(CrupierError, match="Concurrent update"):
        store.transition(
            kind="session",
            record_id="ses_1",
            expected_statuses={"active"},
            status="closed",
            payload={},
            expires_at=None,
            event="closed",
        )


def test_state_schema_double_check_and_directory_path(tmp_path):
    store = SQLiteStateStore(tmp_path / "state.sqlite3")

    class FlipInitializedLock:
        def __enter__(self):
            store._initialized = True

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

    store._schema_lock = FlipInitializedLock()
    store._ensure_schema()
    assert store._initialized is True

    directory_path = tmp_path / "directory.sqlite3"
    directory_path.mkdir()
    with pytest.raises(CrupierError, match="must be a regular file"):
        SQLiteStateStore(directory_path).list("session")
