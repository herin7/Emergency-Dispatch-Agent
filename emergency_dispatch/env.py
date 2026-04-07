from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import (
    Action,
    ActionType,
    Ambulance,
    AmbulanceStatus,
    CityGrid,
    EmergencyCall,
    EnvironmentState,
    StepMetrics,
    UrgencyLevel,
)


@dataclass(slots=True)
class EnvironmentConfig:
    grid_size: int = 10
    ambulance_bases: tuple[tuple[int, int], ...] = ((0, 0), (0, 9), (9, 0))
    poisson_lambda: float = 0.3
    max_steps: int = 200
    task_name: str = "EmergencyDispatch"
    urgency_weights: dict[UrgencyLevel, float] | None = None
    per_step_cost: float = -0.5
    invalid_dispatch_penalty: float = -5.0
    critical_timeout_penalty: float = -100.0
    high_timeout_penalty: float = -40.0
    fuel_empty_penalty: float = -30.0
    minimum_dispatch_fuel: float = 10.0
    minimum_dispatch_distance_limit: int = 5
    end_on_critical_timeout: bool = True
    end_on_all_fuel_depleted: bool = True

    def resolved_urgency_weights(self) -> dict[UrgencyLevel, float]:
        return self.urgency_weights or {
            UrgencyLevel.CRITICAL: 0.1,
            UrgencyLevel.HIGH: 0.2,
            UrgencyLevel.MEDIUM: 0.4,
            UrgencyLevel.LOW: 0.3,
        }


class EmergencyDispatchEnv:
    def __init__(self, config: EnvironmentConfig | None = None, seed: int | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.grid = self._build_grid(self.config.grid_size)
        self.ambulances: list[Ambulance] = []
        self.active_calls: list[EmergencyCall] = []
        self.completed_calls: list[EmergencyCall] = []
        self.metrics = StepMetrics()
        self._call_counter = 0
        self.reset()

    def _build_grid(self, size: int) -> CityGrid:
        return CityGrid(size=size, cells=[[0 for _ in range(size)] for _ in range(size)])

    def reset(self) -> dict[str, Any]:
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.grid = self._build_grid(self.config.grid_size)
        self.active_calls = []
        self.completed_calls = []
        self.metrics = StepMetrics()
        self._call_counter = 0
        self.ambulances = [
            Ambulance(
                id=f"amb_{index}",
                x=base_x,
                y=base_y,
                fuel_level=100,
                status=AmbulanceStatus.IDLE,
                base_x=base_x,
                base_y=base_y,
            )
            for index, (base_x, base_y) in enumerate(self.config.ambulance_bases)
        ]
        return self.state()

    def state(self) -> dict[str, Any]:
        return EnvironmentState(
            step_count=self.step_count,
            grid=self.grid,
            ambulances=self.ambulances,
            active_calls=self.active_calls,
            completed_calls=self.completed_calls,
            cumulative_reward=self.cumulative_reward,
            max_steps=self.config.max_steps,
            metrics=self.metrics,
            task_name=self.config.task_name,
            mode="done" if self._is_done() else "running",
        ).model_dump(mode="json")

    def step(self, action: Action | dict[str, Any] | None) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        parsed_action = self._coerce_action(action)
        reward = self.config.per_step_cost

        if parsed_action is not None:
            reward += self._apply_action(parsed_action)

        movement_reward, movement_info = self._advance_ambulances()
        reward += movement_reward

        timeout_penalty = self._apply_timeouts()
        reward += timeout_penalty

        self._generate_calls()

        self.step_count += 1
        self.cumulative_reward += reward
        done = self._is_done()
        state = self.state()
        info = {
            "applied_action": parsed_action.model_dump(mode="json") if parsed_action else None,
            "movement": movement_info,
            "active_call_count": len(self.active_calls),
            "completed_call_count": len(self.completed_calls),
            "metrics": self.metrics.model_dump(mode="json"),
        }
        return state, reward, done, info

    def _coerce_action(self, action: Action | dict[str, Any] | None) -> Action | None:
        if action is None:
            return None
        if isinstance(action, Action):
            return action
        return Action.model_validate(action)

    def _apply_action(self, action: Action) -> float:
        ambulance = self._find_ambulance(action.ambulance_id) if action.ambulance_id else None
        if action.action_type == ActionType.HOLD:
            if ambulance:
                ambulance.status = AmbulanceStatus.HOLDING
                ambulance.target_x = ambulance.x
                ambulance.target_y = ambulance.y
                ambulance.assigned_call_id = None
            return 0.0

        if ambulance is None:
            return self.config.invalid_dispatch_penalty

        if action.action_type == ActionType.RETURN_TO_BASE:
            if ambulance.assigned_call_id and self._find_call(ambulance.assigned_call_id) is not None:
                return self.config.invalid_dispatch_penalty
            ambulance.status = AmbulanceStatus.RETURNING
            ambulance.target_x = ambulance.base_x
            ambulance.target_y = ambulance.base_y
            ambulance.assigned_call_id = None
            ambulance.dispatch_start_step = None
            return 0.0

        call = self._find_call(action.call_id) if action.call_id else None
        target_x = action.target_x if action.target_x is not None else (call.x if call else ambulance.x)
        target_y = action.target_y if action.target_y is not None else (call.y if call else ambulance.y)
        distance = abs(ambulance.x - target_x) + abs(ambulance.y - target_y)

        if action.action_type in {ActionType.DISPATCH, ActionType.REASSIGN}:
            if call is None:
                return self.config.invalid_dispatch_penalty
            if action.action_type == ActionType.DISPATCH and ambulance.status not in {
                AmbulanceStatus.IDLE,
                AmbulanceStatus.HOLDING,
            }:
                return self.config.invalid_dispatch_penalty
            if ambulance.fuel_level < self.config.minimum_dispatch_fuel and distance > self.config.minimum_dispatch_distance_limit:
                return self.config.invalid_dispatch_penalty
            ambulance.status = AmbulanceStatus.DISPATCHED
            ambulance.target_x = target_x
            ambulance.target_y = target_y
            ambulance.assigned_call_id = call.id if call else ambulance.assigned_call_id
            ambulance.dispatch_start_step = self.step_count
            if call:
                call.assigned_ambulance_id = ambulance.id
            return 0.0

        return 0.0

    def _advance_ambulances(self) -> tuple[float, list[dict[str, Any]]]:
        reward = 0.0
        movement_info: list[dict[str, Any]] = []

        for ambulance in self.ambulances:
            moved = False
            previous_position = (ambulance.x, ambulance.y)

            if ambulance.status in {AmbulanceStatus.DISPATCHED, AmbulanceStatus.RETURNING}:
                if ambulance.fuel_level <= 0:
                    if ambulance.status == AmbulanceStatus.DISPATCHED:
                        reward += self.config.fuel_empty_penalty
                        self.metrics.fuel_out_events += 1
                    ambulance.status = AmbulanceStatus.OUT_OF_FUEL
                    ambulance.target_x = ambulance.x
                    ambulance.target_y = ambulance.y
                else:
                    moved = self._move_towards_target(ambulance)
                    ambulance.fuel_level = max(0.0, ambulance.fuel_level - 1.0)
                    self.metrics.movement_steps += 1
                    self.metrics.total_fuel_consumed += 1.0
                    if ambulance.status == AmbulanceStatus.DISPATCHED:
                        self.metrics.useful_steps += 1
                    if ambulance.fuel_level == 0 and ambulance.status == AmbulanceStatus.DISPATCHED:
                        reward += self.config.fuel_empty_penalty
                        self.metrics.fuel_out_events += 1
                        ambulance.status = AmbulanceStatus.OUT_OF_FUEL

            resolved_call_reward = self._resolve_arrival_if_needed(ambulance)
            reward += resolved_call_reward

            if moved:
                movement_info.append(
                    {
                        "ambulance_id": ambulance.id,
                        "from": previous_position,
                        "to": (ambulance.x, ambulance.y),
                        "fuel_level": ambulance.fuel_level,
                    }
                )

        return reward, movement_info

    def _move_towards_target(self, ambulance: Ambulance) -> bool:
        if ambulance.target_x is None or ambulance.target_y is None:
            return False
        if ambulance.x == ambulance.target_x and ambulance.y == ambulance.target_y:
            return False
        if ambulance.x < ambulance.target_x:
            ambulance.x += 1
        elif ambulance.x > ambulance.target_x:
            ambulance.x -= 1
        elif ambulance.y < ambulance.target_y:
            ambulance.y += 1
        elif ambulance.y > ambulance.target_y:
            ambulance.y -= 1
        return True

    def _resolve_arrival_if_needed(self, ambulance: Ambulance) -> float:
        if ambulance.status != AmbulanceStatus.DISPATCHED or ambulance.assigned_call_id is None:
            if (
                ambulance.status == AmbulanceStatus.RETURNING
                and ambulance.x == ambulance.base_x
                and ambulance.y == ambulance.base_y
            ):
                ambulance.status = AmbulanceStatus.IDLE
                ambulance.target_x = None
                ambulance.target_y = None
            return 0.0

        call = self._find_call(ambulance.assigned_call_id)
        if call is None:
            ambulance.status = AmbulanceStatus.IDLE
            ambulance.assigned_call_id = None
            ambulance.target_x = None
            ambulance.target_y = None
            ambulance.dispatch_start_step = None
            return 0.0

        if ambulance.x != call.x or ambulance.y != call.y:
            return 0.0

        response_time = self.step_count - call.arrival_time + 1
        call.resolved = True
        call.resolved_time = self.step_count + 1
        self.active_calls = [active for active in self.active_calls if active.id != call.id]
        self.completed_calls.append(call)
        self.metrics.resolved_calls += 1
        if call.urgency == UrgencyLevel.CRITICAL:
            self.metrics.resolved_critical_calls += 1
        self.metrics.total_response_time += response_time

        ambulance.status = AmbulanceStatus.RETURNING
        ambulance.assigned_call_id = None
        ambulance.target_x = ambulance.base_x
        ambulance.target_y = ambulance.base_y
        ambulance.dispatch_start_step = None
        return self._arrival_reward(call.urgency, response_time)

    def _arrival_reward(self, urgency: UrgencyLevel, response_time: int) -> float:
        routing_bonus = 0.0
        if response_time > 0:
            routing_bonus = min(15.0, 15.0 / response_time)
        if urgency == UrgencyLevel.CRITICAL and response_time <= 5:
            return 50.0 + routing_bonus
        if urgency == UrgencyLevel.HIGH and response_time <= 10:
            return 25.0 + routing_bonus
        if urgency == UrgencyLevel.MEDIUM:
            return 10.0 + routing_bonus
        if urgency == UrgencyLevel.LOW:
            return 5.0 + routing_bonus
        return routing_bonus

    def _apply_timeouts(self) -> float:
        penalty = 0.0
        for call in self.active_calls:
            elapsed = self.step_count - call.arrival_time
            if call.urgency == UrgencyLevel.CRITICAL and elapsed > 15 and not call.timeout_penalty_applied:
                penalty += self.config.critical_timeout_penalty
                call.timeout_penalty_applied = True
                self.metrics.critical_timeouts += 1
            if call.urgency == UrgencyLevel.HIGH and elapsed > 25 and not call.timeout_penalty_applied:
                penalty += self.config.high_timeout_penalty
                call.timeout_penalty_applied = True
                self.metrics.high_timeouts += 1
        return penalty

    def _generate_calls(self) -> None:
        new_call_count = int(self.rng.poisson(self.config.poisson_lambda))
        for _ in range(new_call_count):
            urgency = self._sample_urgency()
            call = EmergencyCall(
                id=f"call_{self._call_counter}",
                x=int(self.rng.integers(0, self.config.grid_size)),
                y=int(self.rng.integers(0, self.config.grid_size)),
                urgency=urgency,
                arrival_time=self.step_count + 1,
            )
            self._call_counter += 1
            self.active_calls.append(call)
            self.metrics.total_calls += 1
            if urgency == UrgencyLevel.CRITICAL:
                self.metrics.critical_calls += 1
            if urgency == UrgencyLevel.HIGH:
                self.metrics.high_calls += 1

    def _sample_urgency(self) -> UrgencyLevel:
        levels = list(self.config.resolved_urgency_weights().keys())
        probabilities = list(self.config.resolved_urgency_weights().values())
        selected_index = int(self.rng.choice(len(levels), p=probabilities))
        return levels[selected_index]

    def _find_ambulance(self, ambulance_id: str | None) -> Ambulance | None:
        if ambulance_id is None:
            return None
        return next((ambulance for ambulance in self.ambulances if ambulance.id == ambulance_id), None)

    def _find_call(self, call_id: str | None) -> EmergencyCall | None:
        if call_id is None:
            return None
        return next((call for call in self.active_calls if call.id == call_id), None)

    def json_state(self) -> str:
        return json.dumps(self.state(), separators=(",", ":"))

    def available_actions(self) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        unresolved_calls = [call for call in self.active_calls if not call.resolved]
        for ambulance in self.ambulances:
            actions.append(Action(action_type=ActionType.HOLD, ambulance_id=ambulance.id).model_dump(mode="json"))
            actions.append(
                Action(action_type=ActionType.RETURN_TO_BASE, ambulance_id=ambulance.id).model_dump(mode="json")
            )
            for call in unresolved_calls:
                actions.append(
                    Action(
                        action_type=ActionType.DISPATCH,
                        ambulance_id=ambulance.id,
                        call_id=call.id,
                    ).model_dump(mode="json")
                )
                actions.append(
                    Action(
                        action_type=ActionType.REASSIGN,
                        ambulance_id=ambulance.id,
                        call_id=call.id,
                    ).model_dump(mode="json")
                )
        return actions

    def heuristic_action(self) -> dict[str, Any]:
        available_calls = [call for call in self.active_calls if not call.resolved]
        if not available_calls:
            idle_ambulance = next((amb for amb in self.ambulances if amb.status == AmbulanceStatus.IDLE), None)
            if idle_ambulance:
                return Action(action_type=ActionType.HOLD, ambulance_id=idle_ambulance.id).model_dump(mode="json")
            return Action(action_type=ActionType.HOLD, ambulance_id=self.ambulances[0].id).model_dump(mode="json")

        ranked_calls = sorted(
            available_calls,
            key=lambda call: (
                {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}[call.urgency.value],
                call.arrival_time,
            ),
        )
        for call in ranked_calls:
            best_ambulance = min(
                self.ambulances,
                key=lambda amb: abs(amb.x - call.x) + abs(amb.y - call.y) + (0 if amb.status == AmbulanceStatus.IDLE else 3),
            )
            return Action(
                action_type=ActionType.DISPATCH if best_ambulance.status == AmbulanceStatus.IDLE else ActionType.REASSIGN,
                ambulance_id=best_ambulance.id,
                call_id=call.id,
            ).model_dump(mode="json")
        return Action(action_type=ActionType.HOLD, ambulance_id=self.ambulances[0].id).model_dump(mode="json")

    def seed_rng(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def render(self) -> str:
        canvas = [["." for _ in range(self.config.grid_size)] for _ in range(self.config.grid_size)]
        for ambulance in self.ambulances:
            canvas[ambulance.x][ambulance.y] = "A"
        for call in self.active_calls:
            canvas[call.x][call.y] = call.urgency.value[0]
        return "\n".join(" ".join(row) for row in canvas)

    def _is_done(self) -> bool:
        if self.step_count >= self.config.max_steps:
            return True
        if self.config.end_on_critical_timeout and self.metrics.critical_timeouts > 0:
            return True
        if self.config.end_on_all_fuel_depleted and all(ambulance.fuel_level <= 0 for ambulance in self.ambulances):
            return True
        return False
