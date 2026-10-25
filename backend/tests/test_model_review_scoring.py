"""The model-review report must not inherit deterministic benchmark verdicts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/geoagent/scripts/score_model_review.py"
SPEC = importlib.util.spec_from_file_location("score_model_review", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def _row(run_status: str, review: str | None, *, old_status: str = "GIS_EXECUTION_FAILURE") -> dict:
    return {
        "task_id": "gis-test",
        "run_id": "run-test",
        "terminal": {"run_status": run_status},
        "run_metadata": {"completion_review": {"decision": review}} if review else {},
        "status": old_status,
        "error_type": "VALIDATION_FAILURE",
    }


def test_completed_model_accept_is_not_failed_by_old_numeric_verdict() -> None:
    result = scorer.score(_row("COMPLETED", "accept"))
    assert result["status"] == "PASS"
    assert result["original_program_score"]["status"] == "GIS_EXECUTION_FAILURE"
    assert result["score_basis"] == "run_status_and_in_run_model_completion_review"


def test_completion_without_accepted_review_is_not_auto_passed() -> None:
    assert scorer.score(_row("COMPLETED", None))["status"] == "REVIEW_NOT_ACCEPTED"
    assert scorer.score(_row("PARTIAL_COMPLETED", "continue"))["status"] == "PARTIAL"
    assert scorer.score(_row("WAITING_USER", None))["status"] == "BLOCKED_USER_INPUT"


def test_task_association_block_remains_visible() -> None:
    row = _row("COMPLETED", "accept")
    row["run_metadata"]["task_association_error"] = "conflict"
    assert scorer.score(row)["status"] == "SYSTEM_BLOCKED"


def test_recovered_tool_error_does_not_hide_final_model_failure() -> None:
    row = _row("FAILED", None)
    row["terminal"]["error"] = "MODEL_EMPTY_RESPONSE"
    row["gis_tool_failures"] = [{"tool": "python.execute", "error": "PYTHON_EXECUTION_FAILED"}]
    result = scorer.score(row)
    assert result["status"] == "SYSTEM_BLOCKED"
    assert result["reason"] == "MODEL_EMPTY_RESPONSE"
    assert result["gis_tool_failures"] == row["gis_tool_failures"]
