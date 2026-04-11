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
TASK_SELECTION = os.getenv("EMERGENCY_DISPATCH_TASK", "all").lower()
BENCHMARK = "EmergencyDispatch"
MAX_COMPLETION_TOKENS = 60
TEMPERATURE = 0.1
TASK_ORDER = ("easy", "medium", "hard")
TASK_MAP = {
    "easy": EasyDispatchTask,
    "medium": MediumDispatchTask,
    "hard": HardDispatchTask,
}

SYSTEM_PROMPT = """You are an expert emergency dispatch coordinator managing a fleet of ambulances.

## OBJECTIVE
Maximize total reward by efficiently dispatching ambulances to emergency calls while managing fuel resources.

## STATE STRUCTURE
- `ambulances[]`: Fleet with fields: id, x, y, fuel_level (0-100), status, base_x, base_y, target_x, target_y, assigned_call_id
- `active_calls[]`: Pending emergencies with: id, x, y, urgency, arrival_time, assigned_ambulance_id
- `completed_calls[]`: Resolved calls
- `metrics`: Episode statistics (total_calls, resolved_calls, critical_calls, etc.)
- `step_count`: Current simulation step (0 to max_steps)
- `distance_matrix`: Precomputed Manhattan distances `{amb_id: {call_id: distance}}` - use this to find nearest ambulance

## URGENCY LEVELS (Critical > High > Medium > Low)
- **Critical**: Must resolve within 15 steps or receive -100 penalty
- **High**: Must resolve within 25 steps or receive -40 penalty
- **Medium**: Standard priority
- **Low**: Lowest priority

## AMBULANCE STATUS VALUES
- **idle**: Available for dispatch
- **holding**: Stationary, available for dispatch
- **dispatched**: En route to call location
- **returning**: Returning to base after call
- **out_of_fuel**: Cannot move until refueled at base

## ACTIONS
- **dispatch**: Send idle/holding ambulance to call (requires ambulance_id + call_id)
- **reassign**: Redirect already-dispatched ambulance to higher-priority call
- **return_to_base**: Send ambulance back to refuel (auto-triggers when fuel=0)
- **hold**: Keep ambulance stationary (only if no urgent calls exist)

## STRATEGIC GUIDELINES
1. **Use distance_matrix**: Look up `distance_matrix[ambulance_id][call_id]` to find the nearest available ambulance for each call
2. **Prioritize by urgency**: Always handle Critical calls first, then High, then Medium, then Low
3. **Manage fuel**: Don't dispatch ambulances with <10 fuel to distant calls (>5 distance)
4. **Avoid unnecessary holds**: Only hold if no active calls exist
5. **Don't repeat assignments**: Avoid assigning same ambulance to same call repeatedly
6. **Return empty ambulances**: Send returning ambulances to base promptly
7. **Balance coverage**: Keep ambulances distributed across the grid

## REWARD SIGNALS
- Resolving Critical call quickly (<=5 steps): +50 bonus
- Resolving High call quickly (<=10 steps): +25 bonus
- Per-step cost: -0.1 (efficiency pressure)
- Invalid dispatch: -5.0 penalty
- Fuel exhaustion mid-dispatch: -30.0 penalty
- Critical timeout (>15 steps): -100.0 penalty
- High timeout (>25 steps): -40.0 penalty

## OUTPUT FORMAT
Respond with EXACTLY ONE JSON object and NOTHING ELSE:
{"action_type":"dispatch","ambulance_id":"amb_0","call_id":"call_0"}

Valid action_type values: dispatch, return_to_base, hold, reassign
For hold/return_to_base: call_id should be null or omitted
For dispatch/reassign: must include both ambulance_id and call_id

DO NOT include explanations, markdown formatting, or extra fields."""


def iter_task_ids() -> list[str]:
    requested = [part.strip().lower() for part in TASK_SELECTION.split(",") if part.strip()]
    if not requested or requested == ["all"]:
        return list(TASK_ORDER)
    selected = [task_id for task_id in TASK_ORDER if task_id in requested]
    return selected or list(TASK_ORDER)


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


def choose_action(
    client: OpenAI | None,
    model_name: str,
    state: dict[str, Any],
    valid_actions: list[dict[str, Any]],
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    fallback_action = compact_action(fallback)
    if client is None:
        return fallback_action, "missing_hf_token"

    valid_actions_str = json.dumps(valid_actions[:20], separators=(",", ":"))
    prompt = (
        "Current state JSON:\n"
        f"{json.dumps(state, separators=(',', ':'))}\n\n"
        f"Valid actions (choose one of these if possible):\n{valid_actions_str}\n\n"
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
        validated = compact_action(candidate)
        if valid_actions and validated not in valid_actions:
            for valid_action in valid_actions:
                if (
                    valid_action["action_type"] == validated["action_type"]
                    and valid_action.get("ambulance_id") == validated.get("ambulance_id")
                ):
                    return valid_action, None
            return fallback_action, "action_not_in_mask"
        return validated, None
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


def run_episode(task_id: str, client: OpenAI | None) -> None:
    rewards: list[float] = []
    score = 0.0
    steps = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        task_cls = TASK_MAP[task_id]
        task = task_cls()
        env = task.create_env(seed=7)
        grader = task.create_grader()
        state = env.reset()
        done = False
        while not done:
            fallback_action = env.heuristic_action()
            valid_actions = env.valid_actions_for(state)
            action, action_error = choose_action(
                client=client,
                model_name=MODEL_NAME,
                state=state,
                valid_actions=valid_actions,
                fallback=fallback_action,
            )
            state, reward, done, _ = env.step(action)
            steps += 1
            rewards.append(float(reward))

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=steps, action=action_str, reward=float(reward), done=done, error=action_error)

        score = grader.grade(state, task_name=task_id).final_score
        success = True
    except Exception as exc:
        if steps == 0:
            log_step(step=1, action="null", reward=0.0, done=True, error=str(exc))
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> None:
    client = build_client()
    for task_id in iter_task_ids():
        run_episode(task_id=task_id, client=client)


if __name__ == "__main__":
    main()
