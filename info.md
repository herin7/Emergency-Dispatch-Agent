# Emergency Dispatch Agent Technical Documentation

## Overview

Emergency Dispatch Agent is a grid-based simulation environment for studying ambulance fleet coordination under uncertainty. The project models a city as an `N x N` grid, tracks multiple ambulances with individual state, generates emergency calls dynamically, and scores policies using a deterministic grader. It is structured to fit an OpenEnv-style submission with a clear simulation entrypoint, task presets, reproducible inference flow, and containerized execution.

The repository currently contains:

- a typed simulation package in `emergency_dispatch/`
- a root `inference.py` runner
- an `openenv.yaml` environment definition
- a `Dockerfile` and `requirements.txt`
- a small test suite and mock OpenAI-compatible server

## Repository Structure

Key files:

- `emergency_dispatch/models.py`
- `emergency_dispatch/env.py`
- `emergency_dispatch/tasks.py`
- `emergency_dispatch/grader.py`
- `emergency_dispatch/__init__.py`
- `inference.py`
- `openenv.yaml`
- `Dockerfile`
- `requirements.txt`
- `tests/mock_openai_server.py`
- `tests/test_env_and_grader.py`
- `tests/test_inference_format.py`

## Core Simulation Model

### 1. City Grid

The city is represented by the `CityGrid` Pydantic model.

It contains:

- `size`: the dimension of the grid
- `cells`: an `NxN` integer matrix

The current implementation initializes all cells as `0`, which gives a simple uniform road model. The validator ensures the matrix is square and matches the declared size. This makes the state representation explicit and safe for downstream serialization.

### 2. Ambulance Model

Each ambulance is represented by the `Ambulance` Pydantic model.

Fields include:

- `id`
- `x`, `y`: current position
- `fuel_level`: floating-point value between `0` and `100`
- `status`
- `base_x`, `base_y`: fixed home/base coordinates
- `target_x`, `target_y`: active target if the ambulance is moving
- `assigned_call_id`
- `dispatch_start_step`

The status uses the `AmbulanceStatus` enum:

- `idle`
- `dispatched`
- `returning`
- `holding`
- `out_of_fuel`

This allows the environment to reason about whether a vehicle is available, in transit, or no longer usable.

### 3. Emergency Call Model

Each incident is represented by the `EmergencyCall` model.

Fields include:

- `id`
- `x`, `y`: incident location
- `urgency`
- `arrival_time`
- `assigned_ambulance_id`
- `resolved`
- `timeout_penalty_applied`
- `resolved_time`

The urgency level uses the `UrgencyLevel` enum:

- `Critical`
- `High`
- `Medium`
- `Low`

This call model supports both immediate decision-making and post-episode scoring.

### 4. Action Model

Each agent decision is represented by the `Action` model.

Supported action types:

- `dispatch`
- `return_to_base`
- `hold`
- `reassign`

The schema is validated so the correct fields are present for each action type. For example:

- `dispatch` requires an `ambulance_id` and a `call_id` or explicit target coordinates
- `hold` requires an `ambulance_id`
- `return_to_base` requires an `ambulance_id`

This keeps the inference layer and simulation layer strongly typed.

## Environment Class

The main simulation logic lives in `EmergencyDispatchEnv` in `emergency_dispatch/env.py`.

The environment exposes:

- `reset()`
- `step(action)`
- `state()`
- `render()`
- `available_actions()`
- `heuristic_action()`

### reset()

`reset()` initializes a fresh episode.

For the default environment, it:

- creates a `10x10` city grid
- places `3` ambulances at fixed base coordinates
- resets fuel levels to `100`
- clears active and completed call queues
- resets metrics and counters

Task presets override these defaults for medium and hard modes.

### state()

`state()` packages the full environment into the `EnvironmentState` Pydantic model and then serializes it to a JSON-friendly dictionary.

It includes:

- current step count
- full grid matrix
- full ambulance list
- active calls
- completed calls
- cumulative reward
- configured max steps
- metrics for grading
- task name
- run mode (`running` or `done`)

This is the object used both by inference and by the grader.

### step(action)

`step()` is the central transition function. It performs one unit of simulation time.

Its high-level flow is:

1. Validate and coerce the incoming action.
2. Apply the action to ambulance state.
3. Move ambulances one grid cell toward their targets.
4. Decrease fuel for moving ambulances.
5. Resolve completed arrivals.
6. Apply timeout penalties.
7. Generate new emergency calls using a Poisson process.
8. Increment the step counter.
9. Return updated state, reward, done flag, and metadata.

The return signature is:

- `state`
- `reward`
- `done`
- `info`

This mirrors the familiar RL environment pattern.

## Movement Logic

Ambulance movement is intentionally simple and deterministic.

At each step:

- an ambulance can move at most one grid cell
- it moves toward its target along Manhattan distance
- the environment prioritizes reducing `x` distance first, then `y`
- moving costs `1` fuel

This simple routing rule keeps the simulation predictable and cheap to evaluate while still preserving the planning challenge.

## Dynamic Call Generation

New emergency calls are generated every step using a Poisson process.

The environment samples:

- number of new calls from `Poisson(lambda)`
- urgency from a configured urgency distribution
- location from uniform random coordinates within the grid

This makes demand stochastic but reproducible under a fixed RNG seed.

Default baseline:

- `lambda = 0.3` for the base environment

Task presets increase the rate for more difficult scenarios.

## Reward Function

The reward system is shaped to encourage strong dispatch behavior rather than only end-of-episode success.

### Positive rewards

- `+50` if a `Critical` call is reached within `5` steps
- `+25` if a `High` call is reached within `10` steps
- `+10` for a resolved `Medium` call
- `+5` for a resolved `Low` call
- routing bonus from `0` to `15`, inversely related to response time

### Negative rewards

- `-100` if a `Critical` call exceeds `15` steps without being reached
- `-40` if a `High` call exceeds `25` steps without being reached
- `-30` if an ambulance reaches `0` fuel while mid-dispatch
- `-5` for invalid dispatch-style actions
- `-0.5` per environment step as an efficiency cost

The reward design does two things:

- strongly prioritizes critical-response behavior
- discourages wasteful or unrealistic motion

## Constraint Logic

The environment includes several rule checks inspired by the project specification.

### Fuel-aware dispatch validation

Dispatch or reassignment is penalized if:

- the ambulance fuel is below `10`
- and the target distance is greater than `5`

This prevents obviously bad assignments.

### Busy ambulance protection

Plain `dispatch` is penalized when sent to an ambulance that is already active and not in an idle or holding state.

### Return-to-base sequencing

`return_to_base` is penalized if the ambulance is still tied to an unresolved call.

### Early episode termination

An episode ends if either of these occurs:

- maximum step count is reached
- at least one critical timeout occurs and the config enables immediate failure
- all ambulances are fully fuel-depleted and the config enables that termination rule

## Metrics and Grading

The environment accumulates step metrics in the `StepMetrics` model.

Tracked metrics include:

- total calls
- resolved calls
- critical calls
- high calls
- resolved critical calls
- critical timeouts
- high timeouts
- total response time
- movement steps
- useful steps
- fuel-out events
- total fuel consumed

These metrics are consumed by `DispatchEpisodeGrader` in `emergency_dispatch/grader.py`.

### Final score components

The grader outputs a final score from `0.0` to `1.0`.

It uses:

- `40%` Critical Response Rate
- `25%` Mean Response Time Score
- `15%` Coverage Efficiency
- `20%` Zero Timeout Score

### Score definitions

Critical Response Rate:

- resolved critical calls divided by total critical calls

Mean Response Time Score:

- normalized inverse of average response time

Coverage Efficiency:

- resolved calls divided by total fuel consumed, clipped to `1.0`

Zero Timeout Score:

- `1.0` if no critical timeout occurred
- otherwise `0.0`

The grader returns both the breakdown and the final bounded score.

## Task Presets

Three difficulty presets are defined in `emergency_dispatch/tasks.py`.

### EasyDispatchTask

- `10x10` grid
- `3` ambulances
- mostly `Low` and `Medium` calls
- lower arrival rate

### MediumDispatchTask

- `15x15` grid
- `5` ambulances
- mixed urgency calls
- fuel constraints actively matter more

### HardDispatchTask

- `20x20` grid
- `5` ambulances
- high frequency of `Critical` calls
- hardest evaluation scenario

Each task creates its own environment configuration and shares the same grader logic.

## Inference Pipeline

The executable evaluation script is `inference.py`.

Its role is to run one full episode, using an LLM to suggest actions, while printing logs in a strict format.

### Environment variables

The script reads:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

These are not hardcoded. This is important because external evaluators can inject their own credentials or API proxy.

### Client behavior

The script:

- creates an `OpenAI` client with the provided base URL and token
- uses short timeouts and low retry counts
- falls back to a built-in heuristic policy if the API fails, times out, or is rate-limited

This keeps the evaluation robust even when external inference is unstable.

### Episode flow

`inference.py`:

1. creates `HardDispatchTask`
2. resets the environment
3. prints `[START]`
4. loops until the episode is done
5. sends current state JSON to the LLM
6. parses the action JSON
7. applies the action through `env.step()`
8. prints a structured `[STEP]` log line
9. grades the final state
10. prints `[END] Score: ...`

### Expected output format

The script emits:

```text
[START]
[STEP] State: <current state JSON> Action: <action JSON> Reward: <float>
[STEP] State: <current state JSON> Action: <action JSON> Reward: <float>
[END] Score: <float between 0.0 and 1.0>
```

This format has been locally tested using a mock OpenAI-compatible HTTP server.

## Packaging

### openenv.yaml

`openenv.yaml` declares:

- environment name
- description
- simulation entrypoint
- grader entrypoint
- the three task entrypoints

This is the top-level project description for external tooling.

### Dockerfile

The Dockerfile:

- uses `python:3.11-slim`
- installs dependencies from `requirements.txt`
- copies the full repository into the container
- starts with `python inference.py`

This matches the common expectations for lightweight reproducible submission containers.

### requirements.txt

The current dependency list includes:

- `gymnasium`
- `pydantic`
- `numpy`
- `openai`
- `huggingface_hub`

These cover simulation, validation, packaging, and inference.

## Testing and Validation

The repository includes a small test suite.

### tests/test_env_and_grader.py

Checks:

- reset defaults
- step output structure
- bounded grader score

### tests/test_inference_format.py

Checks:

- `inference.py` runs end to end
- log tags match the required format
- action/state JSON is parseable
- final score remains in bounds

### tests/mock_openai_server.py

Provides:

- a local OpenAI-compatible `/chat/completions` endpoint
- deterministic mock responses for integration testing

This allows offline verification of the inference pipeline without spending API quota.

## Current Limitations

The current version is intentionally compact and hackathon-friendly. It does not yet model:

- real road graphs
- blocked cells or traffic congestion
- nearest-hospital optimization
- explicit refueling stations
- direct multi-agent policy interfaces
- visual UI layers such as Gradio or Pygame

Those would be natural next extensions, but the current implementation already provides a solid evaluation-ready core.

## Suggested Future Extensions

- Gymnasium registration with `gym.make(...)`
- Gradio or Pygame demo UI for Hugging Face Spaces
- richer map semantics with roads and zones
- support for traffic and time-of-day demand
- benchmark scripts comparing random, heuristic, PPO, and LLM controllers
- README with setup, screenshots, and example outputs

## Summary

Emergency Dispatch Agent is designed as a structured, extensible RL environment centered on a socially meaningful problem. It combines:

- typed state/action modeling
- deterministic step logic
- stochastic but seeded call generation
- multi-objective reward shaping
- bounded programmatic grading
- reproducible single-episode inference
- deployment-ready packaging

In short, it is both a usable technical benchmark and a compelling demo environment for emergency fleet coordination research.
