from __future__ import annotations

from pydantic import BaseModel, Field

from .models import EnvironmentState


class GradeBreakdown(BaseModel):
    critical_response_rate: float = Field(..., ge=0.0, le=1.0)
    mean_response_time_score: float = Field(..., ge=0.0, le=1.0)
    coverage_efficiency: float = Field(..., ge=0.0, le=1.0)
    zero_timeout_score: float = Field(..., ge=0.0, le=1.0)
    task_objective_score: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)


class DispatchEpisodeGrader:
    def grade(self, final_state: dict, task_name: str = "HardDispatchTask") -> GradeBreakdown:
        grid_size = self._extract_grid_size(final_state)
        metrics_data = final_state.get("metrics", {})

        critical_total = max(metrics_data.get("critical_calls", 0), 1)
        resolved_critical = metrics_data.get("resolved_critical_calls", 0)
        critical_response_rate = min(resolved_critical / critical_total, 1.0)

        resolved_critical_count = max(resolved_critical, 1)
        total_response_time_critical = metrics_data.get("total_response_time_critical", 0)
        # If no critical calls resolved, mean_response_time_score = 0.0
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

        task_objective_score = self._task_objective_score(
            metrics_data, task_name, grid_size,
        )

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
    def _extract_grid_size(state: dict) -> int:
        """Extract grid size from either old (grid.size) or new (grid_size) format."""
        if "grid_size" in state:
            return state["grid_size"]
        grid = state.get("grid", {})
        if isinstance(grid, dict):
            return grid.get("size", 10)
        if hasattr(grid, "size"):
            return grid.size  # type: ignore[union-attr]
        return 10

    @staticmethod
    def _task_objective_score(metrics: dict, task_name: str, grid_size: int) -> float:
        """Task-specific win conditions. Returns 0.0–1.0."""
        if task_name == "EasyDispatchTask":
            resolution_rate = metrics.get("resolved_calls", 0) / max(metrics.get("total_calls", 1), 1)
            no_critical_timeout = 1.0 if metrics.get("critical_timeouts", 0) == 0 else 0.0
            return min((resolution_rate / 0.8) * 0.7 + no_critical_timeout * 0.3, 1.0)

        elif task_name == "MediumDispatchTask":
            crit_rate = metrics.get("resolved_critical_calls", 0) / max(metrics.get("critical_calls", 1), 1)
            fuel_efficiency = max(0.0, 1.0 - (metrics.get("fuel_out_events", 0) * 0.2))
            return min((crit_rate / 0.7) * 0.6 + fuel_efficiency * 0.4, 1.0)

        else:  # HardDispatchTask
            no_critical_timeout = 1.0 if metrics.get("critical_timeouts", 0) == 0 else 0.0
            coverage = metrics.get("resolved_calls", 0) / max(metrics.get("total_calls", 1), 1)
            return min(no_critical_timeout * 0.6 + (coverage / 0.8) * 0.4, 1.0)
