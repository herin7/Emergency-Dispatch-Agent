from __future__ import annotations

from dataclasses import dataclass

from .env import EmergencyDispatchEnv, EnvironmentConfig
from .grader import DispatchEpisodeGrader
from .models import UrgencyLevel


@dataclass(slots=True)
class BaseDispatchTask:
    name: str
    description: str
    config: EnvironmentConfig

    def create_env(self, seed: int | None = None) -> EmergencyDispatchEnv:
        return EmergencyDispatchEnv(config=self.config, seed=seed)

    def create_grader(self) -> DispatchEpisodeGrader:
        return DispatchEpisodeGrader()


class EasyDispatchTask(BaseDispatchTask):
    def __init__(self) -> None:
        super().__init__(
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
