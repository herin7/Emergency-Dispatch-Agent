from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AmbulanceStatus(str, Enum):
    IDLE = "idle"
    DISPATCHED = "dispatched"
    RETURNING = "returning"
    HOLDING = "holding"
    OUT_OF_FUEL = "out_of_fuel"


class UrgencyLevel(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ActionType(str, Enum):
    DISPATCH = "dispatch"
    RETURN_TO_BASE = "return_to_base"
    HOLD = "hold"
    REASSIGN = "reassign"


class CityGrid(BaseModel):
    """City grid size (Manhattan distance metric, no terrain)."""
    size: int = Field(..., ge=1, le=50)


class Ambulance(BaseModel):
    id: str
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    fuel_level: float = Field(..., ge=0, le=100)
    status: AmbulanceStatus
    base_x: int = Field(..., ge=0)
    base_y: int = Field(..., ge=0)
    target_x: int | None = Field(default=None, ge=0)
    target_y: int | None = Field(default=None, ge=0)
    assigned_call_id: str | None = None
    dispatch_start_step: int | None = Field(default=None, ge=0)


class EmergencyCall(BaseModel):
    id: str
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    urgency: UrgencyLevel
    arrival_time: int = Field(..., ge=0)
    assigned_ambulance_id: str | None = None
    resolved: bool = False
    timeout_penalty_applied: bool = False
    resolved_time: int | None = Field(default=None, ge=0)


# ── OpenEnv Spec: Typed Action ──
class Action(BaseModel):
    action_type: ActionType
    ambulance_id: str | None = None
    call_id: str | None = None
    target_x: int | None = Field(default=None, ge=0)
    target_y: int | None = Field(default=None, ge=0)

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: ActionType) -> ActionType:
        return v


# ── OpenEnv Spec: Typed Observation ──
class Observation(BaseModel):
    """What the agent sees each step."""
    step_count: int
    grid_size: int
    ambulances: list[Ambulance]
    active_calls: list[EmergencyCall]
    completed_calls: list[EmergencyCall]
    cumulative_reward: float
    max_steps: int
    metrics: StepMetrics
    distance_matrix: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Precomputed Manhattan distances: {amb_id: {call_id: distance}}"
    )
    mode: Literal["running", "done"] = "running"


# ── OpenEnv Spec: Typed Reward ──
class Reward(BaseModel):
    """Reward breakdown for the current step."""
    step_reward: float = Field(..., description="Total reward for this step")
    per_step_cost: float = 0.0
    arrival_bonus: float = 0.0
    invalid_action_penalty: float = 0.0
    fuel_penalty: float = 0.0
    timeout_penalty: float = 0.0


# ── OpenEnv Spec: StepResult ──
class StepResult(BaseModel):
    """Full result of a step() call."""
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class StepMetrics(BaseModel):
    total_calls: int = 0
    resolved_calls: int = 0
    critical_calls: int = 0
    high_calls: int = 0
    resolved_critical_calls: int = 0
    critical_timeouts: int = 0
    high_timeouts: int = 0
    total_response_time: int = 0
    total_response_time_critical: int = 0
    movement_steps: int = 0
    useful_steps: int = 0
    fuel_out_events: int = 0
    total_fuel_consumed: float = 0.0


class EnvironmentState(BaseModel):
    step_count: int
    grid: CityGrid
    ambulances: list[Ambulance]
    active_calls: list[EmergencyCall]
    completed_calls: list[EmergencyCall]
    cumulative_reward: float
    max_steps: int
    metrics: StepMetrics
    task_name: str | None = None
    mode: Literal["running", "done"] = "running"
