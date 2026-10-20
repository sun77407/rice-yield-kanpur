"""GEE EECU quota ledger (Community tier: 150 EECU-h total, 100 EECU-h usable ceiling).

Every export logs an *estimate* before it runs and the *actual* EECU after it
finishes, both to ``logs/gee_quota.jsonl``. If the projected cumulative cost
would exceed the ceiling, ``QuotaExceeded`` is raised and the caller must stop
and ask the user.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

QUOTA_LOG = Path("logs/gee_quota.jsonl")
CEILING_EECU_HOURS = 100.0


class QuotaExceeded(RuntimeError):  # noqa: N818
    """Projected cumulative EECU would exceed the agreed ceiling."""


def _read(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [
        json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _append(record: dict[str, Any], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), **record}
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def cumulative_hours(log: Path = QUOTA_LOG) -> float:
    """Sum EECU-hours committed: actual where an export finished, else its estimate."""
    est: dict[str, float] = {}
    act: dict[str, float] = {}
    for r in _read(log):
        if r["event"] == "estimate":
            est[r["export_id"]] = float(r["estimated_eecu_hours"])
        elif r["event"] == "actual" and r.get("actual_eecu_hours") is not None:
            act[r["export_id"]] = float(r["actual_eecu_hours"])
    return sum(act.get(k, v) for k, v in est.items())


def log_estimate(
    export_id: str,
    estimated_eecu_hours: float,
    basis: str,
    details: dict[str, Any],
    log: Path = QUOTA_LOG,
    ceiling: float = CEILING_EECU_HOURS,
) -> None:
    """Record an estimate *before* running an export; raise if it breaks the ceiling."""
    projected = cumulative_hours(log) + estimated_eecu_hours
    if projected > ceiling:
        raise QuotaExceeded(
            f"{export_id}: projected {projected:.2f} EECU-h > ceiling {ceiling:.0f}. Ask the user."
        )
    _append(
        {
            "event": "estimate",
            "export_id": export_id,
            "estimated_eecu_hours": estimated_eecu_hours,
            "basis": basis,
            "projected_cumulative_hours": projected,
            "details": details,
        },
        log,
    )


def log_actual(
    export_id: str,
    task_id: str,
    actual_eecu_seconds: float | None,
    state: str,
    log: Path = QUOTA_LOG,
) -> None:
    """Record the metered cost after a task finishes (``None`` if GEE did not report it)."""
    hours = None if actual_eecu_seconds is None else actual_eecu_seconds / 3600.0
    rec: dict[str, Any] = {
        "event": "actual",
        "export_id": export_id,
        "task_id": task_id,
        "state": state,
        "actual_eecu_seconds": actual_eecu_seconds,
        "actual_eecu_hours": hours,
    }
    if hours is None:
        # keep the estimate in force when the metered value is unavailable
        rec.pop("actual_eecu_hours")
    _append(rec, log)
