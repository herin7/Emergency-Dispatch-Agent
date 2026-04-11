from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


TASK_NAME_BY_ID = {
    "easy": "EasyDispatchTask",
    "medium": "MediumDispatchTask",
    "hard": "HardDispatchTask",
}

TASK_GRADER_IMPORTS = {
    task_id: f"emergency_dispatch.grader:grade_{task_id}"
    for task_id in TASK_NAME_BY_ID
}


class GradeBreakdown(BaseModel):
    critical_response_rate: float = Field(..., ge=0.0, le=1.0)
    mean_response_time_score: float = Field(..., ge=0.0, le=1.0)
    coverage_efficiency: float = Field(..., ge=0.0, le=1.0)
    zero_timeout_score: float = Field(..., ge=0.0, le=1.0)
    task_objective_score: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)


class DispatchEpisodeGrader:
    def grade(self, final_state: dict[str, Any] | None = None, task_name: str = "HardDispatchTask") -> GradeBreakdown:
        state = final_state if isinstance(final_state, dict) else {}
        grid_size = self._extract_grid_size(state)
        metrics_data = state.get("metrics", {})

        critical_total = max(metrics_data.get("critical_calls", 0), 1)
        resolved_critical = metrics_data.get("resolved_critical_calls", 0)
        critical_response_rate = min(resolved_critical / critical_total, 1.0)

        resolved_critical_count = max(resolved_critical, 1)
        total_response_time_critical = metrics_data.get("total_response_time_critical", 0)
        if resolved_critical == 0:
            mean_response_time_score = 0.0
        else:
            mean_response_time_critical = total_response_time_critical / resolved_critical_count
            max_reasonable_response_time = grid_size * 2
            mean_response_time_score = max(0.0, 1.0 - (mean_response_time_critical / max_reasonable_response_time))

        total_calls_generated = max(metrics_data.get("total_calls", 0), 1)
        resolved_calls = metrics_data.get("resolved_calls", 0)
        coverage_efficiency = min(resolved_calls / total_calls_generated, 1.0)

        critical_timeouts = metrics_data.get("critical_timeouts", 0)
        high_timeouts = metrics_data.get("high_timeouts", 0)
        total_timeouts = critical_timeouts + high_timeouts
        zero_timeout_score = max(0.0, 1.0 - (total_timeouts * 0.33))

        task_objective_score = self._task_objective_score(metrics_data, task_name, grid_size)

        final_score = (
            critical_response_rate * 0.30
            + mean_response_time_score * 0.20
            + coverage_efficiency * 0.15
            + zero_timeout_score * 0.15
            + task_objective_score * 0.20
        )
        final_score = min(max(final_score, 0.0), 1.0)
        return GradeBreakdown(
            critical_response_rate=critical_response_rate,
            mean_response_time_score=mean_response_time_score,
            coverage_efficiency=coverage_efficiency,
            zero_timeout_score=zero_timeout_score,
            task_objective_score=task_objective_score,
            final_score=final_score,
        )

    @staticmethod
    def _extract_grid_size(state: dict[str, Any]) -> int:
        if "grid_size" in state:
            return state["grid_size"]
        grid = state.get("grid", {})
        if isinstance(grid, dict):
            return grid.get("size", 10)
        if hasattr(grid, "size"):
            return grid.size  # type: ignore[union-attr]
        return 10

    @staticmethod
    def _task_objective_score(metrics: dict[str, Any], task_name: str, grid_size: int) -> float:
        normalized_task_name = DispatchEpisodeGrader._normalize_task_name(task_name)
        if normalized_task_name == "easy":
            resolution_rate = metrics.get("resolved_calls", 0) / max(metrics.get("total_calls", 1), 1)
            no_critical_timeout = 1.0 if metrics.get("critical_timeouts", 0) == 0 else 0.0
            return min((resolution_rate / 0.8) * 0.7 + no_critical_timeout * 0.3, 1.0)

        if normalized_task_name == "medium":
            crit_rate = metrics.get("resolved_critical_calls", 0) / max(metrics.get("critical_calls", 1), 1)
            fuel_efficiency = max(0.0, 1.0 - (metrics.get("fuel_out_events", 0) * 0.2))
            return min((crit_rate / 0.7) * 0.6 + fuel_efficiency * 0.4, 1.0)

        no_critical_timeout = 1.0 if metrics.get("critical_timeouts", 0) == 0 else 0.0
        coverage = metrics.get("resolved_calls", 0) / max(metrics.get("total_calls", 1), 1)
        return min(no_critical_timeout * 0.6 + (coverage / 0.8) * 0.4, 1.0)

    @staticmethod
    def _normalize_task_name(task_name: str) -> str:
        normalized = (task_name or "hard").strip().lower()
        if normalized in TASK_NAME_BY_ID:
            return normalized
        if normalized == "easydispatchtask":
            return "easy"
        if normalized == "mediumdispatchtask":
            return "medium"
        return "hard"


def _extract_state_from_candidate(candidate: Any) -> dict[str, Any] | None:
    if isinstance(candidate, dict):
        for key in ("final_state", "state", "observation"):
            nested = candidate.get(key)
            if isinstance(nested, dict):
                return nested
        return candidate
    if isinstance(candidate, list):
        for item in reversed(candidate):
            nested = _extract_state_from_candidate(item)
            if nested is not None:
                return nested
    return None


def _resolve_grader_state(final_state: Any = None, **kwargs: Any) -> dict[str, Any]:
    candidates = (
        final_state,
        kwargs.get("state"),
        kwargs.get("observation"),
        kwargs.get("trajectory"),
        kwargs.get("result"),
        kwargs.get("episode"),
    )
    for candidate in candidates:
        resolved = _extract_state_from_candidate(candidate)
        if resolved is not None:
            return resolved
    return {}


def _grade_task(task_id: str, final_state: Any = None, **kwargs: Any) -> float:
    state = _resolve_grader_state(final_state, **kwargs)
    return DispatchEpisodeGrader().grade(state, task_name=task_id).final_score


def grade_easy(final_state: Any = None, **kwargs: Any) -> float:
    return _grade_task("easy", final_state=final_state, **kwargs)


def grade_medium(final_state: Any = None, **kwargs: Any) -> float:
    return _grade_task("medium", final_state=final_state, **kwargs)


def grade_hard(final_state: Any = None, **kwargs: Any) -> float:
    return _grade_task("hard", final_state=final_state, **kwargs)
