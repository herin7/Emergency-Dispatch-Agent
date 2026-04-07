from __future__ import annotations

from pydantic import BaseModel, Field

from .models import EnvironmentState


class GradeBreakdown(BaseModel):
    critical_response_rate: float = Field(..., ge=0.0, le=1.0)
    mean_response_time_score: float = Field(..., ge=0.0, le=1.0)
    coverage_efficiency: float = Field(..., ge=0.0, le=1.0)
    zero_timeout_score: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)


class DispatchEpisodeGrader:
    def grade(self, final_state: dict) -> GradeBreakdown:
        state = EnvironmentState.model_validate(final_state)
        metrics = state.metrics

        critical_total = max(metrics.critical_calls, 1)
        resolved_total = max(metrics.resolved_calls, 1)
        mean_response_time = metrics.total_response_time / resolved_total
        max_reasonable_response_time = max(state.grid.size * 2, 1)
        total_fuel_consumed = max(metrics.total_fuel_consumed, 1.0)

        critical_response_rate = min(metrics.resolved_critical_calls / critical_total, 1.0)
        mean_response_time_score = max(0.0, 1.0 - (mean_response_time / max_reasonable_response_time))
        coverage_efficiency = min(metrics.resolved_calls / total_fuel_consumed, 1.0)
        zero_timeout_score = 1.0 if metrics.critical_timeouts == 0 else 0.0

        final_score = (
            critical_response_rate * 0.40
            + mean_response_time_score * 0.25
            + coverage_efficiency * 0.15
            + zero_timeout_score * 0.20
        )
        final_score = min(max(final_score, 0.0), 1.0)
        return GradeBreakdown(
            critical_response_rate=critical_response_rate,
            mean_response_time_score=mean_response_time_score,
            coverage_efficiency=coverage_efficiency,
            zero_timeout_score=zero_timeout_score,
            final_score=final_score,
        )
