from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    size: int = Field(..., ge=1)
    cells: list[list[int]]

    @field_validator("cells")
    @classmethod
    def validate_square_matrix(cls, cells: list[list[int]]) -> list[list[int]]:
        if not cells:
            raise ValueError("CityGrid cells must not be empty.")
        row_length = len(cells[0])
        if row_length == 0:
            raise ValueError("CityGrid rows must not be empty.")
        if any(len(row) != row_length for row in cells):
            raise ValueError("CityGrid cells must form a rectangular matrix.")
        return cells

    @model_validator(mode="after")
    def validate_size_matches_matrix(self) -> "CityGrid":
        if len(self.cells) != self.size or any(len(row) != self.size for row in self.cells):
            raise ValueError("CityGrid size must match an NxN matrix.")
        return self


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


class Action(BaseModel):
    action_type: ActionType
    ambulance_id: str | None = None
    call_id: str | None = None
    target_x: int | None = Field(default=None, ge=0)
    target_y: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "Action":
        if self.action_type in {ActionType.DISPATCH, ActionType.REASSIGN}:
            if not self.ambulance_id:
                raise ValueError("ambulance_id is required for dispatch and reassign.")
            if not self.call_id and (self.target_x is None or self.target_y is None):
                raise ValueError("dispatch and reassign require call_id or explicit target coordinates.")
        if self.action_type == ActionType.RETURN_TO_BASE and not self.ambulance_id:
            raise ValueError("ambulance_id is required for return_to_base.")
        if self.action_type == ActionType.HOLD and self.ambulance_id is None:
            raise ValueError("ambulance_id is required for hold.")
        return self


class StepMetrics(BaseModel):
    total_calls: int = 0
    resolved_calls: int = 0
    critical_calls: int = 0
    high_calls: int = 0
    resolved_critical_calls: int = 0
    critical_timeouts: int = 0
    high_timeouts: int = 0
    total_response_time: int = 0
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
