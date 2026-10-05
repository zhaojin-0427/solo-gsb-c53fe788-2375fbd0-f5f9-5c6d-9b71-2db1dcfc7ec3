"""SQLite 存储层。

* 单文件数据库，WAL 模式；
* 所有写操作经全局锁串行化，事务内最后做一次唯一键复查，杜绝并发重复插入；
* 计算失败时不写入任何结果（“拒绝且不保存部分结果”）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Optional

from .config import settings
from .errors import ConflictError, NotFoundError

_WRITE_LOCK = threading.RLock()


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or settings.db_path
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


_CONN: Optional[sqlite3.Connection] = None
_CONN_LOCK = threading.Lock()


def get_conn() -> sqlite3.Connection:
    global _CONN
    with _CONN_LOCK:
        if _CONN is None:
            _CONN = _connect()
        return _CONN


SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    version            TEXT PRIMARY KEY,
    name               TEXT,
    description        TEXT,
    expression         TEXT NOT NULL,
    expression_canonical TEXT NOT NULL,
    inputs_json        TEXT NOT NULL,
    output_unit        TEXT NOT NULL,
    correlations_json  TEXT NOT NULL,
    summary_json       TEXT NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS evaluations (
    id                  TEXT PRIMARY KEY,
    request_hash        TEXT NOT NULL,
    idempotency_key     TEXT NOT NULL,
    model_version       TEXT NOT NULL REFERENCES models(version),
    inputs_json         TEXT NOT NULL,
    n                   INTEGER NOT NULL,
    seed                INTEGER NOT NULL,
    convergence_tol     REAL NOT NULL,
    result_json         TEXT NOT NULL,
    cached              INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_request_hash ON evaluations(request_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_idempotency ON evaluations(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluations(model_version);
"""


def init_db() -> None:
    conn = get_conn()
    with _WRITE_LOCK:
        conn.executescript(SCHEMA)
        conn.commit()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def insert_model(row: dict[str, Any]) -> None:
    conn = get_conn()
    with _WRITE_LOCK:
        exists = conn.execute("SELECT 1 FROM models WHERE version = ?", (row["version"],)).fetchone()
        if exists:
            raise ConflictError(f"模型版本 {row['version']!r} 已存在且不可变")
        try:
            conn.execute(
                """INSERT INTO models
                   (version, name, description, expression, expression_canonical,
                    inputs_json, output_unit, correlations_json, summary_json)
                   VALUES (:version, :name, :description, :expression, :expression_canonical,
                           :inputs_json, :output_unit, :correlations_json, :summary_json)""",
                {
                    "version": row["version"],
                    "name": row.get("name"),
                    "description": row.get("description"),
                    "expression": row["expression"],
                    "expression_canonical": row["expression_canonical"],
                    "inputs_json": json.dumps(row["inputs"], ensure_ascii=False, sort_keys=True),
                    "output_unit": row["output_unit"],
                    "correlations_json": json.dumps(row["correlations"], ensure_ascii=False),
                    "summary_json": json.dumps(row["summary"], ensure_ascii=False, sort_keys=True),
                },
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise ConflictError(f"模型版本 {row['version']!r} 已存在且不可变") from exc


def get_model(version: str) -> dict[str, Any]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM models WHERE version = ?", (version,)).fetchone()
    if row is None:
        raise NotFoundError(f"模型版本 {version!r} 不存在")
    return _model_row_to_dict(row)


def list_models(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM models ORDER BY created_at DESC, version LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [_model_row_to_dict(r) for r in rows]


def _model_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "version": row["version"],
        "name": row["name"],
        "description": row["description"],
        "expression": row["expression"],
        "expression_canonical": row["expression_canonical"],
        "inputs": json.loads(row["inputs_json"]),
        "output_unit": row["output_unit"],
        "correlations": json.loads(row["correlations_json"]),
        "summary": json.loads(row["summary_json"]),
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# evaluations
# ---------------------------------------------------------------------------

def insert_evaluation(
    eval_id: str,
    request_hash: str,
    idempotency_key: str,
    model_version: str,
    inputs_json: str,
    n: int,
    seed: int,
    tol: float,
    result_json: str,
) -> dict[str, Any]:
    """幂等插入。事务内复查 request_hash / idempotency_key，命中则返回既有结果。"""
    conn = get_conn()
    with _WRITE_LOCK:
        existing = conn.execute(
            "SELECT * FROM evaluations WHERE request_hash = ? OR idempotency_key = ?",
            (request_hash, idempotency_key),
        ).fetchone()
        if existing is not None:
            stored = _eval_row_to_dict(existing)
            if stored["request_hash"] != request_hash:
                raise ConflictError(
                    f"幂等键 {idempotency_key!r} 已绑定到另一个计算请求（"
                    f"evaluation {stored['id']}）"
                )
            return {"cached": True, "evaluation": stored}
        try:
            conn.execute(
                """INSERT INTO evaluations
                   (id, request_hash, idempotency_key, model_version, inputs_json,
                    n, seed, convergence_tol, result_json, cached)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (eval_id, request_hash, idempotency_key, model_version, inputs_json,
                 n, seed, tol, result_json),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise ConflictError("重复的幂等键或请求哈希") from exc
        row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,)).fetchone()
        return {"cached": False, "evaluation": _eval_row_to_dict(row)}


def get_evaluation(eval_id: str) -> dict[str, Any]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"计算结果 {eval_id!r} 不存在")
    return _eval_row_to_dict(row)


def find_evaluation_by_idempotency_key(key: str) -> Optional[dict[str, Any]]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM evaluations WHERE idempotency_key = ?", (key,)).fetchone()
    return _eval_row_to_dict(row) if row is not None else None


def list_evaluations(model_version: Optional[str] = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    conn = get_conn()
    if model_version is not None:
        rows = conn.execute(
            "SELECT * FROM evaluations WHERE model_version = ? ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            (model_version, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM evaluations ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_eval_row_to_dict(r) for r in rows]


def _eval_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "request_hash": row["request_hash"],
        "idempotency_key": row["idempotency_key"],
        "model_version": row["model_version"],
        "inputs": json.loads(row["inputs_json"]),
        "n": row["n"],
        "seed": row["seed"],
        "convergence_tolerance": row["convergence_tol"],
        "result": json.loads(row["result_json"]),
        "cached": bool(row["cached"]),
        "created_at": row["created_at"],
    }
