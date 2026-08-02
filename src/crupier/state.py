"""Small atomic state store shared by approvals, sessions, and experiments."""

from __future__ import annotations

import builtins
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from .errors import CrupierError


@dataclass(slots=True)
class StateRecord:
    kind: str
    id: str
    status: str
    version: int
    created_at: str
    updated_at: str
    expires_at: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "status": self.status,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "payload": self.payload,
        }


class SQLiteStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._initialized = False
        self._schema_lock = RLock()

    def create(
        self,
        *,
        kind: str,
        record_id: str,
        status: str,
        payload: dict[str, Any],
        expires_at: str | None = None,
        event: str = "created",
        actor: str | None = None,
    ) -> StateRecord:
        now = _now()
        encoded = _encode(payload)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO records
                        (kind, id, status, version, created_at, updated_at, expires_at, payload)
                    VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                    """,
                    (kind, record_id, status, now, now, expires_at, encoded),
                )
            except sqlite3.IntegrityError as exc:
                raise CrupierError(f"State record {kind}:{record_id} already exists.") from exc
            self._append_event(
                connection,
                kind=kind,
                record_id=record_id,
                event=event,
                actor=actor,
                payload={"status": status},
                created_at=now,
            )
        return self.get(kind, record_id)

    def get(self, kind: str, record_id: str) -> StateRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM records WHERE kind = ? AND id = ?",
                (kind, record_id),
            ).fetchone()
        if row is None:
            raise CrupierError(f"State record {kind}:{record_id} was not found.")
        return _record(row)

    def list(
        self,
        kind: str,
        *,
        status: str | None = None,
        limit: int = 200,
    ) -> list[StateRecord]:
        query = "SELECT * FROM records WHERE kind = ?"
        params: list[Any] = [kind]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 10_000)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_record(row) for row in rows]

    def transition(
        self,
        *,
        kind: str,
        record_id: str,
        expected_statuses: set[str],
        status: str,
        payload: dict[str, Any],
        expires_at: str | None,
        event: str,
        actor: str | None = None,
        expected_version: int | None = None,
    ) -> StateRecord:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM records WHERE kind = ? AND id = ?",
                (kind, record_id),
            ).fetchone()
            if row is None:
                raise CrupierError(f"State record {kind}:{record_id} was not found.")
            current = _record(row)
            if expected_version is not None and current.version != expected_version:
                raise CrupierError(
                    f"Concurrent update detected for state record {kind}:{record_id}; "
                    f"expected version {expected_version}, found {current.version}."
                )
            if current.status not in expected_statuses:
                expected = ", ".join(sorted(expected_statuses))
                raise CrupierError(
                    f"State record {kind}:{record_id} is {current.status!r}; expected one of: {expected}."
                )
            cursor = connection.execute(
                """
                UPDATE records
                SET status = ?, version = version + 1, updated_at = ?, expires_at = ?, payload = ?
                WHERE kind = ? AND id = ? AND version = ?
                """,
                (
                    status,
                    now,
                    expires_at,
                    _encode(payload),
                    kind,
                    record_id,
                    current.version,
                ),
            )
            if cursor.rowcount != 1:
                raise CrupierError(
                    f"Concurrent update detected for state record {kind}:{record_id}."
                )
            self._append_event(
                connection,
                kind=kind,
                record_id=record_id,
                event=event,
                actor=actor,
                payload={"from": current.status, "to": status},
                created_at=now,
            )
        return self.get(kind, record_id)

    def events(self, kind: str, record_id: str) -> builtins.list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT seq, event, actor, created_at, payload
                FROM events
                WHERE kind = ? AND record_id = ?
                ORDER BY seq
                """,
                (kind, record_id),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "event": row["event"],
                "actor": row["actor"],
                "created_at": row["created_at"],
                "payload": _decode(row["payload"]),
            }
            for row in rows
        ]

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._schema_lock:
            if self._initialized:
                return
            if self.path.parent.is_symlink() or self.path.is_symlink():
                raise CrupierError(
                    f"State path {self.path} cannot use symbolic links."
                )
            if self.path.exists() and not self.path.is_file():
                raise CrupierError(f"State path {self.path} must be a regular file.")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.path.parent.chmod(0o700)
            except OSError:
                pass
            with closing(self._raw_connect()) as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS records (
                        kind TEXT NOT NULL,
                        id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        expires_at TEXT,
                        payload TEXT NOT NULL,
                        PRIMARY KEY (kind, id)
                    );
                    CREATE INDEX IF NOT EXISTS records_kind_status_updated
                        ON records(kind, status, updated_at DESC);
                    CREATE TABLE IF NOT EXISTS events (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind TEXT NOT NULL,
                        record_id TEXT NOT NULL,
                        event TEXT NOT NULL,
                        actor TEXT,
                        created_at TEXT NOT NULL,
                        payload TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS events_record
                        ON events(kind, record_id, seq);
                    """
                )
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            self._initialized = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._ensure_schema()
        connection = self._raw_connect()
        try:
            yield connection
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        else:
            if connection.in_transaction:
                connection.commit()
        finally:
            connection.close()

    def _raw_connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        *,
        kind: str,
        record_id: str,
        event: str,
        actor: str | None,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(kind, record_id, event, actor, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (kind, record_id, event, actor, created_at, _encode(payload)),
        )


def _record(row: sqlite3.Row) -> StateRecord:
    return StateRecord(
        kind=str(row["kind"]),
        id=str(row["id"]),
        status=str(row["status"]),
        version=int(row["version"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        expires_at=str(row["expires_at"]) if row["expires_at"] is not None else None,
        payload=_decode(str(row["payload"])),
    )


def _encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode(payload: str) -> dict[str, Any]:
    value = json.loads(payload)
    return dict(value) if isinstance(value, dict) else {}


def _now() -> str:
    return datetime.now(UTC).isoformat()
