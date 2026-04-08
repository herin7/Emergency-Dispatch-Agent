"""
Emergency Dispatch Agent — HuggingFace Space Entry Point
=========================================================
Runs two services on port 7860:
  • FastAPI  — OpenEnv HTTP endpoints (/reset, /step, /state)
  • Gradio   — Interactive UI mounted at /ui

The validator pings POST /reset expecting HTTP 200, so the FastAPI
routes must be live before Gradio finishes loading.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from emergency_dispatch.tasks import EasyDispatchTask, HardDispatchTask, MediumDispatchTask

# ---------------------------------------------------------------------------
# Shared environment state (one instance per Space pod)
# ---------------------------------------------------------------------------

_TASK_MAP = {
    "easy": EasyDispatchTask,
    "medium": MediumDispatchTask,
    "hard": HardDispatchTask,
}

_current_task_name = "hard"
_env = HardDispatchTask().create_env(seed=7)
_grader = HardDispatchTask().create_grader()
_env_lock = threading.Lock()


def _get_env_and_grader(task_name: str):
    global _env, _grader, _current_task_name
    if task_name != _current_task_name:
        cls = _TASK_MAP[task_name]
        _env = cls().create_env(seed=7)
        _grader = cls().create_grader()
        _current_task_name = task_name
    return _env, _grader


# ---------------------------------------------------------------------------
# FastAPI — OpenEnv HTTP endpoints
# ---------------------------------------------------------------------------

api = FastAPI(title="Emergency Dispatch OpenEnv", version="1.0.0")


class ResetRequest(BaseModel):
    task: str = "hard"
    seed: int = 7


class StepRequest(BaseModel):
    action: dict[str, Any]


class GradeRequest(BaseModel):
    state: dict[str, Any]
    task: str = "hard"


@api.post("/reset")
def api_reset(body: ResetRequest = ResetRequest()):
    task_name = body.task.lower() if body.task.lower() in _TASK_MAP else "hard"
    with _env_lock:
        env, _ = _get_env_and_grader(task_name)
        state = env.reset()
    return JSONResponse(content=state)


@api.post("/step")
def api_step(body: StepRequest):
    with _env_lock:
        obs, reward, done, info = _env.step(body.action)
    return JSONResponse(content={"observation": obs, "reward": reward, "done": done, "info": info})


@api.get("/state")
def api_state():
    with _env_lock:
        state = _env.state()
    return JSONResponse(content=state)


@api.get("/tasks")
def api_tasks():
    return JSONResponse(
        content={
            "tasks": [
                {"id": "easy", "class": "emergency_dispatch.tasks:EasyDispatchTask"},
                {"id": "medium", "class": "emergency_dispatch.tasks:MediumDispatchTask"},
                {"id": "hard", "class": "emergency_dispatch.tasks:HardDispatchTask"},
            ]
        }
    )


@api.post("/grader")
def api_grader(body: GradeRequest):
    task_name = body.task.lower() if body.task.lower() in _TASK_MAP else "hard"
    with _env_lock:
        _, grader = _get_env_and_grader(task_name)
        task_label = _TASK_MAP[task_name]().name
        grade = grader.grade(body.state, task_name=task_label)
    return JSONResponse(content=grade.model_dump(mode="json"))


@api.get("/health")
def api_health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Gradio UI helpers
# ---------------------------------------------------------------------------

URGENCY_COLORS = {
    "Critical": "#EF4444",
    "High":     "#F97316",
    "Medium":   "#EAB308",
    "Low":      "#22C55E",
}

STATUS_EMOJIS = {
    "idle":        "💤",
    "dispatched":  "🚑",
    "returning":   "↩️",
    "holding":     "⏸️",
    "out_of_fuel": "⛽",
}


def _render_grid_svg(state: dict) -> str:
    """Build an SVG grid showing ambulances and active calls."""
    grid_size = state.get("grid_size", 10)
    cell = 480 // grid_size
    svg_size = cell * grid_size

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_size}" height="{svg_size}" '
        f'style="background:#0F172A;border-radius:12px;">'
    ]

    # Grid lines
    for i in range(grid_size + 1):
        pos = i * cell
        lines.append(
            f'<line x1="{pos}" y1="0" x2="{pos}" y2="{svg_size}" '
            f'stroke="#1E293B" stroke-width="1"/>'
        )
        lines.append(
            f'<line x1="0" y1="{pos}" x2="{svg_size}" y2="{pos}" '
            f'stroke="#1E293B" stroke-width="1"/>'
        )

    # Emergency calls
    for call in state.get("active_calls", []):
        cx = call["x"] * cell + cell // 2
        cy = call["y"] * cell + cell // 2
        color = URGENCY_COLORS.get(call["urgency"], "#94A3B8")
        r = max(cell // 3, 6)
        lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" '
            f'opacity="0.85" stroke="#fff" stroke-width="1.5"/>'
        )
        if cell >= 24:
            label = call["urgency"][0]
            lines.append(
                f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
                f'font-size="{max(8, cell // 4)}" fill="#fff" font-weight="bold">{label}</text>'
            )

    # Ambulances
    for amb in state.get("ambulances", []):
        cx = amb["x"] * cell + cell // 2
        cy = amb["y"] * cell + cell // 2
        r = max(cell // 4, 5)
        fuel = amb.get("fuel_level", 100)
        opacity = 0.5 + 0.5 * (fuel / 100)
        lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r + 3}" fill="#3B82F6" '
            f'opacity="{opacity:.2f}" stroke="#93C5FD" stroke-width="2"/>'
        )
        idx = amb["id"].split("_")[-1]
        if cell >= 20:
            lines.append(
                f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
                f'font-size="{max(8, cell // 4)}" fill="#fff" font-weight="bold">{idx}</text>'
            )

    # Base markers (small diamonds)
    bases_seen: set = set()
    for amb in state.get("ambulances", []):
        key = (amb["base_x"], amb["base_y"])
        if key not in bases_seen:
            bases_seen.add(key)
            bx = amb["base_x"] * cell + cell // 2
            by = amb["base_y"] * cell + cell // 2
            half = max(cell // 5, 4)
            lines.append(
                f'<polygon points="{bx},{by - half} {bx + half},{by} '
                f'{bx},{by + half} {bx - half},{by}" '
                f'fill="#A78BFA" opacity="0.6"/>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


def _metrics_md(state: dict, score: float | None = None) -> str:
    m = state.get("metrics", {})
    step = state.get("step_count", 0)
    max_steps = state.get("max_steps", 200)
    cum_r = state.get("cumulative_reward", 0.0)

    score_line = f"**Score**: `{score:.4f}`\n\n" if score is not None else ""
    return f"""{score_line}
| Metric | Value |
|--------|-------|
| Step | `{step}` / `{max_steps}` |
| Cumulative Reward | `{cum_r:.2f}` |
| Active Calls | `{len(state.get('active_calls', []))}` |
| Total Calls | `{m.get('total_calls', 0)}` |
| Resolved | `{m.get('resolved_calls', 0)}` |
| Critical Resolved | `{m.get('resolved_critical_calls', 0)}` / `{m.get('critical_calls', 0)}` |
| Critical Timeouts | `{m.get('critical_timeouts', 0)}` |
| High Timeouts | `{m.get('high_timeouts', 0)}` |
| Fuel-Out Events | `{m.get('fuel_out_events', 0)}` |
"""


def _ambulance_table(state: dict) -> str:
    rows = []
    for amb in state.get("ambulances", []):
        emoji = STATUS_EMOJIS.get(amb["status"], "❓")
        fuel_bar = "█" * int(amb.get("fuel_level", 0) // 10) + "░" * (10 - int(amb.get("fuel_level", 0) // 10))
        rows.append(
            f"| {amb['id']} | {emoji} {amb['status']} | "
            f"`{amb.get('fuel_level', 0):.0f}%` {fuel_bar} | "
            f"`({amb['x']}, {amb['y']})` | `{amb.get('assigned_call_id') or '—'}` |"
        )
    header = "| ID | Status | Fuel | Position | Assigned Call |\n|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows) if rows else "*No ambulances*"


def _calls_table(state: dict) -> str:
    calls = state.get("active_calls", [])
    if not calls:
        return "*No active calls*"
    rows = []
    for c in calls:
        color_dot = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(c["urgency"], "⚪")
        rows.append(
            f"| {c['id']} | {color_dot} {c['urgency']} | "
            f"`({c['x']}, {c['y']})` | `{c.get('assigned_ambulance_id') or '—'}` |"
        )
    header = "| Call ID | Urgency | Location | Assigned |\n|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Gradio UI logic
# ---------------------------------------------------------------------------

_ui_state: dict = {"state": None, "score": None, "task": "hard"}


def do_reset(task_choice: str):
    task_name = task_choice.lower()
    with _env_lock:
        env, grader = _get_env_and_grader(task_name)
        state = env.reset()
    _ui_state["state"] = state
    _ui_state["score"] = None
    _ui_state["task"] = task_name
    svg = _render_grid_svg(state)
    metrics = _metrics_md(state)
    ambs = _ambulance_table(state)
    calls = _calls_table(state)
    return svg, metrics, ambs, calls, "✅ Environment reset. Ready to step."


def do_step_heuristic():
    if _ui_state["state"] is None:
        return None, "*Reset first.*", "*—*", "*—*", "⚠️ Reset the environment first."
    with _env_lock:
        action = _env.heuristic_action()
        state, reward, done, _ = _env.step(action)
    _ui_state["state"] = state
    score = None
    status_msg = f"Step {state['step_count']} — reward: `{reward:+.2f}`"
    if done:
        with _env_lock:
            grade = _grader.grade(state, task_name=_env.config.task_name)
        score = grade.final_score
        _ui_state["score"] = score
        status_msg = f"🏁 Episode done! **Score: {score:.4f}**"
    svg = _render_grid_svg(state)
    metrics = _metrics_md(state, score)
    ambs = _ambulance_table(state)
    calls = _calls_table(state)
    return svg, metrics, ambs, calls, status_msg


def do_run_full():
    if _ui_state["state"] is None:
        return None, "*Reset first.*", "*—*", "*—*", "⚠️ Reset the environment first."
    with _env_lock:
        state = _env.state()
        done = state.get("mode") == "done"
        while not done:
            action = _env.heuristic_action()
            state, _, done, _ = _env.step(action)
        grade = _grader.grade(state, task_name=_env.config.task_name)
    score = grade.final_score
    _ui_state["state"] = state
    _ui_state["score"] = score
    svg = _render_grid_svg(state)
    metrics = _metrics_md(state, score)
    ambs = _ambulance_table(state)
    calls = _calls_table(state)
    return svg, metrics, ambs, calls, f"🏁 Full episode complete! **Score: {score:.4f}**"


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------
HEADER_MD = """
<div style="text-align:center; padding: 24px 0 8px;">
  <h1 style="font-size:2.4rem; font-weight:800; background:linear-gradient(135deg,#3B82F6,#A78BFA,#EC4899);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">
  🚑 Emergency Dispatch Agent</h1>
  <p style="color:#94A3B8;font-size:1rem;margin-top:8px;">
  An OpenEnv-compatible reinforcement learning environment for ambulance fleet coordination</p>
</div>
"""

LEGEND_MD = """
<div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap; padding:12px; font-size:0.95rem; color:#E2E8F0; background:#1E293B; border-radius:12px; border: 1px solid #334155; margin-bottom: 16px;">
  <span style="display:flex; align-items:center; gap:6px;">🔵 <b>Ambulance</b></span>
  <span style="display:flex; align-items:center; gap:6px;">🟣 <b>Base</b></span>
  <span style="color:#475569; margin: 0 4px;">|</span>
  <span style="display:flex; align-items:center; gap:6px;">🔴 <b>Critical</b></span>
  <span style="display:flex; align-items:center; gap:6px;">🟠 <b>High</b></span>
  <span style="display:flex; align-items:center; gap:6px;">🟡 <b>Medium</b></span>
  <span style="display:flex; align-items:center; gap:6px;">🟢 <b>Low</b></span>
</div>
"""

CSS = """
/* Root */
body {
    font-family: 'Inter', sans-serif;
    background: #0F172A;
    color: #E2E8F0;
}

/* Main container */
.gradio-container {
    background: #0F172A !important;
    padding: 16px !important;
}

/* Panels / Cards */
.gradio-container .gr-panel,
.gradio-container .block {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    padding: 12px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}

/* Headings */
h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700;
}

/* Markdown text */
.gr-markdown {
    color: #CBD5F5 !important;
    font-size: 0.9rem;
}

/* Buttons */
button {
    border-radius: 10px !important;
    transition: all 0.2s ease-in-out !important;
}

/* Primary button */
button.primary {
    background: linear-gradient(135deg, #3B82F6, #6366F1) !important;
    border: none !important;
    color: white !important;
}

/* Secondary buttons */
button.secondary {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    color: #E2E8F0 !important;
}

/* Hover effects */
button:hover {
    transform: translateY(-1px);
    opacity: 0.95;
}

/* Dropdown */
select, .gr-dropdown {
    background: #1E293B !important;
    color: #E2E8F0 !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}

/* Tables (Markdown rendered) */
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.85rem;
}

th, td {
    border: 1px solid #334155;
    padding: 6px 8px;
    text-align: left;
}

th {
    background: #020617;
    color: #F1F5F9;
}

/* Status box */
.gr-markdown p {
    margin: 4px 0;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 4px;
}

/* Fix HTML grid container */
.gr-html {
    display: flex;
    justify-content: center;
    align-items: center;
}

/* Responsive */
@media (max-width: 768px) {
    .gradio-container {
        padding: 8px !important;
    }
}
"""

with gr.Blocks(title="Emergency Dispatch Agent") as demo:
    gr.HTML(f"<style>{CSS}</style>" + HEADER_MD)

    with gr.Row():
        with gr.Column(scale=1):
            task_dd = gr.Dropdown(
                choices=["Easy", "Medium", "Hard"],
                value="Hard",
                label="Task Difficulty",
                interactive=True,
            )
            with gr.Row():
                btn_reset = gr.Button("🔄 Reset", variant="primary")
                btn_step  = gr.Button("⏭ Step (Heuristic)", variant="secondary")
                btn_run   = gr.Button("▶ Run Full Episode", variant="secondary")
            status_box = gr.Markdown("*Press Reset to start.*")

        with gr.Column(scale=2):
            gr.HTML(LEGEND_MD)
            grid_html = gr.HTML(label="City Grid")

    with gr.Row():
        with gr.Column(scale=1):
            metrics_md = gr.Markdown("*Metrics will appear after reset.*", label="Episode Metrics")
        with gr.Column(scale=1):
            ambs_md = gr.Markdown("*—*", label="Ambulance Fleet")

    calls_md = gr.Markdown("*—*", label="Active Emergency Calls")

    btn_reset.click(
        fn=do_reset,
        inputs=[task_dd],
        outputs=[grid_html, metrics_md, ambs_md, calls_md, status_box],
    )
    btn_step.click(
        fn=do_step_heuristic,
        inputs=[],
        outputs=[grid_html, metrics_md, ambs_md, calls_md, status_box],
    )
    btn_run.click(
        fn=do_run_full,
        inputs=[],
        outputs=[grid_html, metrics_md, ambs_md, calls_md, status_box],
    )

# ---------------------------------------------------------------------------
# Mount Gradio onto FastAPI and start uvicorn
# ---------------------------------------------------------------------------

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
