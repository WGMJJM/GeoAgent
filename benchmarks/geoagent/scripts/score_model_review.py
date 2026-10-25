"""Score saved GIS runs using their in-run model review, without numeric validation.

This is a separate reporting policy. Frozen requests, raw execution records, and
the original deterministic validator remain available for historical audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

DEFAULT_SOURCE = ROOT / "reports/request-v5-task-association/evaluator_original_gis_results.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/request-v5-task-association/model_review_results.jsonl"


def score(row: dict) -> dict:
    terminal = row.get("terminal") or {}
    metadata = row.get("run_metadata") or {}
    review = metadata.get("completion_review") or {}
    run_status = terminal.get("run_status")
    failures = row.get("gis_tool_failures") or []

    if metadata.get("task_association_error") or terminal.get("error") == "TASK_ASSOCIATION_BLOCKED":
        status, reason = "SYSTEM_BLOCKED", "TASK_ASSOCIATION_ERROR"
    elif run_status == "COMPLETED" and review.get("decision") == "accept":
        status, reason = "PASS", None
    elif run_status == "COMPLETED":
        status, reason = "REVIEW_NOT_ACCEPTED", review.get("decision") or "REVIEW_MISSING"
    elif run_status == "WAITING_USER":
        status, reason = "BLOCKED_USER_INPUT", "WAITING_USER"
    elif run_status == "PARTIAL_COMPLETED":
        status, reason = "PARTIAL", review.get("decision") or "PARTIAL_COMPLETED"
    elif failures:
        status, reason = "GIS_EXECUTION_FAILURE", "GIS_TOOL_FAILURE"
    else:
        status, reason = "SYSTEM_BLOCKED", row.get("error_type") or terminal.get("error") or run_status or "NO_TERMINAL_RECORD"

    return {
        "task_id": row["task_id"],
        "run_id": row.get("run_id") or terminal.get("run_id"),
        "conversation_id": row.get("conversation_id"),
        "status": status,
        "reason": reason,
        "run_status": run_status,
        "model_review_decision": review.get("decision"),
        "gis_tool_failures": failures,
        "original_program_score": {
            "status": row.get("status"),
            "error_type": row.get("error_type"),
        },
        "score_basis": "run_status_and_in_run_model_completion_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("Source and output must differ; raw records cannot be overwritten.")
    rows = read_jsonl(source)
    if len(rows) != 20 or len({row["task_id"] for row in rows}) != 20:
        raise ValueError("Expected exactly 20 unique frozen GIS task records.")
    scored = [score(row) for row in rows]
    counts = Counter(row["status"] for row in scored)
    summary = {
        "scoring_policy": "model_review_only_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source": source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source),
        "source_sha256": sha256(source),
        "task_count": len(scored),
        "status_counts": dict(counts),
        "model_review_passes": counts["PASS"],
        "model_review_pass_rate": counts["PASS"] / len(scored),
        "interpretation": "Model-reviewed completion rate, not independently verified GIS correctness.",
        "program_validator_used": False,
        "original_records_preserved": True,
    }
    write_jsonl(output, scored)
    write_json(output.with_name("model_review_summary.json"), summary)
    print(f"Model-reviewed completion: {counts['PASS']}/{len(scored)} ({counts['PASS'] / len(scored):.1%})")


if __name__ == "__main__":
    main()
