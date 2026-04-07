from __future__ import annotations

import json
import os
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import ValidationError

from emergency_dispatch.models import Action
from emergency_dispatch.tasks import EasyDispatchTask, HardDispatchTask, MediumDispatchTask


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("EMERGENCY_DISPATCH_TASK", "hard").lower()
BENCHMARK = "EmergencyDispatch"
# MAX_COMPLETION_TOKENS = 120
MAX_COMPLETION_TOKENS = 60
TEMPERATURE = 0.1

SYSTEM_PROMPT = """You are an expert emergency dispatch planner.

Goal:
Maximize total reward by efficiently assigning ambulances to calls.

Rules:
- Each call should be handled once.
- Avoid assigning same ambulance repeatedly to same call.
- Prefer closest available ambulance.
- Avoid idle actions if calls exist.
- Reassign only if beneficial.

Strictly return ONE JSON:
{"action_type":"dispatch|return_to_base|hold|reassign","ambulance_id":"amb_X","call_id":"call_Y"}

No explanation. No extra fields."""

def get_task() -> tuple[str, Any]:
    task_map = {
        "easy": EasyDispatchTask,
        "medium": MediumDispatchTask,
        "hard": HardDispatchTask,
    }
    task_cls = task_map.get(TASK_NAME, HardDispatchTask)
    task = task_cls()
    return task.name, task


def build_client() -> OpenAI | None:
    if not HF_TOKEN:
        return None
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        max_retries=1, 
        timeout=60.0,
    )


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def extract_first_json_object(content: str) -> dict[str, Any] | None:
    start_index: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(content):
        if start_index is None:
            if char == "{":
                start_index = index
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                candidate = content[start_index : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start_index = None
                    continue
                return parsed if isinstance(parsed, dict) else None

    return None


def compact_action(action: dict[str, Any]) -> dict[str, Any]:
    validated = Action.model_validate(action)
    return {
        "action_type": validated.action_type.value,
        "ambulance_id": validated.ambulance_id,
        "call_id": validated.call_id,
    }


def build_deadline_guard_action(
    state: dict[str, Any],
    last_action: dict[str, Any] | None,
) -> dict[str, Any] | None:
    critical_calls = [call for call in state.get("active_calls", []) if call["urgency"] == "Critical"]
    if not critical_calls:
        return None

    step_count = state.get("step_count", 0)
    ambulances = [ambulance for ambulance in state.get("ambulances", []) if ambulance["fuel_level"] > 0]
    if not ambulances:
        return None

    best_action: dict[str, Any] | None = None
    best_score: tuple[int, int, int, int, int] | None = None

    for call in critical_calls:
        elapsed = step_count - call["arrival_time"]
        slack = 15 - elapsed
        if slack > 4:
            continue

        assigned_ambulance = next(
            (
                ambulance
                for ambulance in ambulances
                if ambulance.get("assigned_call_id") == call["id"]
            ),
            None,
        )
        if assigned_ambulance is not None and slack <= 2:
            return {
                "action_type": "dispatch" if assigned_ambulance["status"] in {"idle", "holding"} else "reassign",
                "ambulance_id": assigned_ambulance["id"],
                "call_id": call["id"],
            }

        for ambulance in ambulances:
            eta = abs(ambulance["x"] - call["x"]) + abs(ambulance["y"] - call["y"])
            busy_penalty = 0 if ambulance["status"] in {"idle", "holding"} else 2
            assignment_priority = 0 if ambulance.get("assigned_call_id") == call["id"] else 1
            action = {
                "action_type": "dispatch" if ambulance["status"] in {"idle", "holding"} else "reassign",
                "ambulance_id": ambulance["id"],
                "call_id": call["id"],
            }
            repeat_penalty = 1 if action == last_action else 0
            score = (
                assignment_priority,
                elapsed * -1,
                0 if eta <= slack else 1,
                max(eta - slack, 0),
                busy_penalty,
                repeat_penalty,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_action = action

    return best_action


def build_safe_fallback(
    state: dict[str, Any],
    fallback: dict[str, Any],
    bad_pairs: set[tuple[str | None, str | None]],
    bad_calls: set[str],
    last_action: dict[str, Any] | None,
) -> dict[str, Any]:
    deadline_guard_action = build_deadline_guard_action(state=state, last_action=last_action)
    if deadline_guard_action is not None:
        return deadline_guard_action

    ambulances = [ambulance for ambulance in state.get("ambulances", []) if ambulance["fuel_level"] > 0]
    idle_ambulances = [ambulance for ambulance in ambulances if ambulance["status"] in {"idle", "holding"}]
    assigned_call_ids = {ambulance["assigned_call_id"] for ambulance in ambulances if ambulance.get("assigned_call_id")}
    fallback_action = compact_action(fallback)
    fallback_pair = (fallback_action.get("ambulance_id"), fallback_action.get("call_id"))
    has_active_calls = bool(state.get("active_calls"))

    if fallback_pair not in bad_pairs and fallback_action.get("call_id") not in bad_calls and fallback_action != last_action:
        if fallback_action.get("call_id") not in assigned_call_ids:
            if not (fallback_action["action_type"] == "hold" and has_active_calls):
                return fallback_action

    urgency_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    best_action: dict[str, Any] | None = None
    best_score: tuple[int, int, int, int] | None = None

    for call in state.get("active_calls", []):
        if call["id"] in assigned_call_ids or call["id"] in bad_calls:
            continue
        for ambulance in idle_ambulances:
            candidate = {
                "action_type": "dispatch",
                "ambulance_id": ambulance["id"],
                "call_id": call["id"],
            }
            pair = (candidate["ambulance_id"], candidate["call_id"])
            if pair in bad_pairs or candidate == last_action:
                continue

            eta = abs(ambulance["x"] - call["x"]) + abs(ambulance["y"] - call["y"])
            score = (
                urgency_rank.get(call["urgency"], 99),
                call["arrival_time"],
                eta,
                -int(ambulance["fuel_level"]),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_action = candidate

    if best_action is not None:
        return best_action

    for ambulance in idle_ambulances:
        candidate = {
            "action_type": "hold",
            "ambulance_id": ambulance["id"],
            "call_id": None,
        }
        if candidate != last_action:
            return candidate

    for call in state.get("active_calls", []):
        for ambulance in ambulances:
            candidate = {
                "action_type": "dispatch" if ambulance["status"] in {"idle", "holding"} else "reassign",
                "ambulance_id": ambulance["id"],
                "call_id": call["id"],
            }
            pair = (candidate["ambulance_id"], candidate["call_id"])
            if pair in bad_pairs or candidate["call_id"] in bad_calls or candidate == last_action:
                continue

            eta = abs(ambulance["x"] - call["x"]) + abs(ambulance["y"] - call["y"])
            busy_penalty = 0 if ambulance["status"] in {"idle", "holding"} else 2
            score = (
                urgency_rank.get(call["urgency"], 99),
                call["arrival_time"],
                eta,
                busy_penalty,
                -int(ambulance["fuel_level"]),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_action = candidate

    if best_action is not None:
        return best_action

    for ambulance in ambulances:
        if ambulance["status"] == "returning":
            candidate = {
                "action_type": "return_to_base",
                "ambulance_id": ambulance["id"],
                "call_id": None,
            }
            if candidate != last_action:
                return candidate

    for ambulance in idle_ambulances:
        candidate = {
            "action_type": "hold",
            "ambulance_id": ambulance["id"],
            "call_id": None,
        }
        if candidate != last_action:
            return candidate

    return {
        "action_type": "hold",
        "ambulance_id": state.get("ambulances", [{}])[0].get("id", "amb_0"),
        "call_id": None,
    }


def choose_action(
    client: OpenAI | None,
    model_name: str,
    state: dict[str, Any],
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    fallback_action = compact_action(fallback)
    if client is None:
        return fallback_action, "missing_hf_token"

    prompt = (
        "Current state JSON:\n"
        f"{json.dumps(state, separators=(',', ':'))}\n"
        "Choose the best next action and respond with JSON only."
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_COMPLETION_TOKENS,
        )
        content = response.choices[0].message.content or ""
        candidate = extract_first_json_object(content)
        if candidate is None:
            return fallback_action, "invalid_model_action"
        return compact_action(candidate), None
    except APIConnectionError:
        return fallback_action, "api_connection_error"
    except APITimeoutError:
        return fallback_action, "api_timeout"
    except RateLimitError:
        return fallback_action, "rate_limit"
    except (KeyError, IndexError, json.JSONDecodeError, ValidationError, AttributeError, TypeError):
        return fallback_action, "invalid_model_action"
    except Exception:
        return fallback_action, "api_error"


def main() -> None:
    task_label, task = get_task()
    env = task.create_env(seed=7)
    grader = task.create_grader()
    client = build_client()

    rewards: list[float] = []
    score = 0.0
    steps = 0
    success = False
    state = env.reset()
    bad_pairs: set[tuple[str | None, str | None]] = set()
    bad_calls: set[str] = set()
    last_action: dict[str, Any] | None = None
    last_reward = 0.0

    log_start(task=task_label, env=BENCHMARK, model=MODEL_NAME)

    try:
        done = False
        while not done:
            fallback_action = build_safe_fallback(
                state=state,
                fallback=env.heuristic_action(),
                bad_pairs=bad_pairs,
                bad_calls=bad_calls,
                last_action=last_action,
            )
            action, action_error = choose_action(
                client=client,
                model_name=MODEL_NAME,
                state=state,
                fallback=fallback_action,
            )

            if last_reward < -5:
                action = fallback_action
                action_error = "negative_reward_guard"

            if steps > 15 and last_reward < -5:
                action = fallback_action
                action_error = "late_negative_guard"

            if steps > 12:
                action = fallback_action
                action_error = "late_game_fallback"

            pair = (action.get("ambulance_id"), action.get("call_id"))
            if action.get("call_id") in bad_calls:
                action = fallback_action
                action_error = "bad_call_blocked"

            if pair in bad_pairs:
                action = fallback_action
                action_error = "bad_pair_blocked"

            if last_action and action == last_action:
                action = fallback_action
                action_error = "repeat_blocked"

            if action["action_type"] == "hold" and state.get("active_calls"):
                action = fallback_action
                action_error = "hold_blocked"

            state, reward, done, _ = env.step(action)
            steps += 1
            rewards.append(float(reward))

            if reward < -3:
                bad_pairs.add((action["ambulance_id"], action["call_id"]))
                if action["call_id"]:
                    bad_calls.add(action["call_id"])

            last_reward = reward
            last_action = action
            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=steps, action=action_str, reward=float(reward), done=done, error=action_error)

        score = grader.grade(state).final_score
        success = True
    except Exception as exc:
        success = False
        if steps == 0:
            log_step(step=1, action="null", reward=0.0, done=True, error=str(exc))
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
