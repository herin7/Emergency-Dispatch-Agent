from __future__ import annotations

from dataclasses import dataclass

from .env import EmergencyDispatchEnv, EnvironmentConfig
from .grader import DispatchEpisodeGrader
from .models import UrgencyLevel


@dataclass(slots=True)
class BaseDispatchTask:
    id: str
    name: str
    description: str
    config: EnvironmentConfig
    grader_class: type[DispatchEpisodeGrader] = DispatchEpisodeGrader

    def create_env(self, seed: int | None = None) -> EmergencyDispatchEnv:
        return EmergencyDispatchEnv(config=self.config, seed=seed)

    def create_grader(self) -> DispatchEpisodeGrader:
        return self.grader_class()

    def grade(self, final_state: dict) -> dict:
        return self.create_grader().grade(final_state, task_name=self.name).model_dump(mode="json")

    def task_spec(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "class": f"{self.__class__.__module__}:{self.__class__.__name__}",
            "grader": {
                "class": f"{self.grader_class.__module__}:{self.grader_class.__name__}",
            },
        }


class EasyDispatchTask(BaseDispatchTask):
    def __init__(self) -> None:
        super().__init__(
            id="easy",
            name="EasyDispatchTask",
            description="10x10 grid, 3 ambulances, mostly Low and Medium calls.",
            config=EnvironmentConfig(
                grid_size=10,
                ambulance_bases=((0, 0), (0, 9), (9, 0)),
                poisson_lambda=0.25,
                max_steps=200,
                task_name="EasyDispatchTask",
                urgency_weights={
                    UrgencyLevel.CRITICAL: 0.05,
                    UrgencyLevel.HIGH: 0.15,
                    UrgencyLevel.MEDIUM: 0.35,
                    UrgencyLevel.LOW: 0.45,
                },
            ),
        )


class MediumDispatchTask(BaseDispatchTask):
    def __init__(self) -> None:
        super().__init__(
            id="medium",
            name="MediumDispatchTask",
            description="15x15 grid, 5 ambulances, mixed calls with active fuel constraints.",
            config=EnvironmentConfig(
                grid_size=15,
                ambulance_bases=((0, 0), (0, 14), (7, 7), (14, 0), (14, 14)),
                poisson_lambda=0.45,
                max_steps=200,
                task_name="MediumDispatchTask",
                urgency_weights={
                    UrgencyLevel.CRITICAL: 0.15,
                    UrgencyLevel.HIGH: 0.25,
                    UrgencyLevel.MEDIUM: 0.30,
                    UrgencyLevel.LOW: 0.30,
                },
            ),
        )


class HardDispatchTask(BaseDispatchTask):
    def __init__(self) -> None:
        super().__init__(
            id="hard",
            name="HardDispatchTask",
            description="20x20 grid, 5 ambulances, frequent Critical calls.",
            config=EnvironmentConfig(
                grid_size=20,
                ambulance_bases=((0, 0), (0, 19), (10, 10), (19, 0), (19, 19)),
                poisson_lambda=0.8,
                max_steps=200,
                task_name="HardDispatchTask",
                urgency_weights={
                    UrgencyLevel.CRITICAL: 0.40,
                    UrgencyLevel.HIGH: 0.25,
                    UrgencyLevel.MEDIUM: 0.20,
                    UrgencyLevel.LOW: 0.15,
                },
            ),
        )
