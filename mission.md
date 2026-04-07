# Emergency Dispatch Agent Mission

## Problem Statement

Emergency response systems are one of the clearest examples of decision-making under pressure. When a medical emergency occurs, the quality of the dispatch decision directly affects response time, resource availability, and in the most serious cases, patient survival. In many cities, especially in dense and fast-growing regions, ambulance allocation is still partially manual, heuristic-driven, or supported by brittle rule systems. These systems often struggle when demand spikes, multiple emergencies happen at once, or available vehicles are constrained by fuel, distance, or return-to-base requirements.

The core challenge is not simply routing one ambulance to one incident. Real dispatch is a sequential resource allocation problem:

- New emergencies appear over time and are not known in advance.
- Each call has a different urgency level and therefore a different consequence for delay.
- Ambulances are limited resources with physical positions, finite fuel, and changing status.
- Sending one ambulance to a low-priority call may leave a critical zone uncovered moments later.
- Reassigning an ambulance can save a high-value case, but may increase response time elsewhere.

This makes emergency coordination a strong candidate for reinforcement learning and structured agent evaluation. A good policy must reason over time, adapt to uncertainty, and continuously trade off local efficiency against system-wide readiness.

## Mission

The mission of the Emergency Dispatch Agent project is to build a programmatically evaluable environment where an agent learns how to dispatch and reposition ambulances across a city grid while balancing:

- response speed for critical emergencies
- coverage across the city
- fuel-aware planning
- return-to-base logistics
- dynamic reprioritization as new calls arrive

The goal is not just to resolve calls. The goal is to resolve the right calls fast enough, with the fewest wasted movements, while avoiding catastrophic failures such as critical timeouts or stranded ambulances.

## Why This Project Matters

This project is intentionally grounded in a real public-interest use case. Emergency dispatch is understandable to judges, meaningful in practice, and technically rich enough to demonstrate strong environment design. It showcases:

- sequential planning under uncertainty
- reward shaping with multiple competing objectives
- multi-entity state modeling
- deterministic grading without human evaluation
- extensibility toward traffic, geography, weather, congestion, and hospital capacity

For OpenEnv, this is a strong fit because the environment is:

- visual and intuitive
- fully structured and programmatically gradeable
- compatible with reproducible evaluation
- realistic enough to feel important, while still compact enough to run quickly

## Core Objective

At every step, the agent must answer a simple but high-stakes question:

Which ambulance should do what right now?

That decision can take several forms:

- dispatch an idle ambulance to a call
- reassign an en-route ambulance to a more urgent call
- return an ambulance to base after service
- hold position to preserve coverage or avoid waste

The agent succeeds when it develops a policy that consistently:

- reaches critical calls within urgency thresholds
- keeps average response time low
- uses fuel efficiently
- avoids critical timeouts
- maintains operational readiness across the full episode

## Intended Research and Demo Value

This project is useful at two levels.

For judges and demo audiences:

- it is easy to understand visually
- the action consequences are immediate
- success and failure are interpretable

For researchers and builders:

- it provides a compact benchmark for dispatch policies
- it supports heuristic, RL, and LLM-driven control approaches
- it exposes clear structured metrics instead of vague subjective scoring

## What Makes the Environment Interesting

The environment becomes interesting because it is not solved by a single greedy rule. A nearest-ambulance-first strategy may perform well on easy traffic-free cases, but it can fail badly when:

- a nearby ambulance is nearly out of fuel
- a more severe call arrives one step later
- returning an ambulance to base would improve future coverage
- reassigning one ambulance causes another area to become under-served

This means strong policies must be context-sensitive, not just reactive.

## OpenEnv Submission Intent

The submission is designed to demonstrate a complete OpenEnv-ready project:

- typed environment state and action models
- reproducible simulation
- task presets across difficulty levels
- a programmatic grader with bounded scores from 0.0 to 1.0
- a single-episode inference pipeline
- container and config files for reproducible packaging

## Final Vision

The long-term vision is to grow this environment into a richer emergency operations simulator with:

- road topology rather than uniform grids
- traffic congestion and blocked routes
- hospital load balancing
- zone-based demand hotspots
- refueling or charging stations
- explicit multi-agent training modes

The current version focuses on a strong core loop: dispatch, movement, delay, urgency, and measurable outcomes. That gives the project a clean foundation while keeping it practical for hackathon evaluation and future extension.
