"""
FastAPI application for the DataClean Environment.

Exposes HTTP endpoints for reset/step/state/health.
Compatible with OpenEnv validation and standard clients.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from server.environment import DataCleanEnvironment
from server.graders import grade_task

app = FastAPI(
    title="DataCleanEnv",
    description="Data Cleaning Environment for OpenEnv - AI agents learn to clean dirty tabular data",
    version="0.1.0",
)

# Single environment instance (one session at a time)
env = DataCleanEnvironment()


# ============================================================================
# Request/Response Models (Pydantic for FastAPI validation)
# ============================================================================

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42

class ActionRequest(BaseModel):
    action_type: str = ""
    row_index: int = -1
    column_name: str = ""
    new_value: str = ""
    reason: str = ""


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Landing page for the DataClean Environment."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DataCleanEnv - OpenEnv Data Cleaning Environment</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .container { max-width: 700px; padding: 40px; text-align: center; }
            h1 { font-size: 2.5em; margin-bottom: 10px; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .subtitle { color: #94a3b8; font-size: 1.1em; margin-bottom: 30px; }
            .badge { display: inline-block; background: #22c55e; color: #fff; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; margin-bottom: 25px; }
            .endpoints { text-align: left; background: #1e293b; border-radius: 12px; padding: 24px; margin: 20px 0; }
            .endpoints h3 { color: #38bdf8; margin-bottom: 12px; }
            .endpoint { padding: 8px 0; border-bottom: 1px solid #334155; font-family: monospace; }
            .endpoint:last-child { border-bottom: none; }
            .method { color: #22c55e; font-weight: bold; margin-right: 8px; }
            .method.post { color: #f59e0b; }
            .tasks { display: flex; gap: 12px; justify-content: center; margin: 20px 0; }
            .task { background: #1e293b; border-radius: 8px; padding: 16px; flex: 1; }
            .task h4 { color: #38bdf8; }
            .task .score { font-size: 1.5em; font-weight: bold; color: #22c55e; }
            a { color: #38bdf8; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 DataCleanEnv</h1>
            <p class="subtitle">OpenEnv Data Cleaning Environment</p>
            <div class="badge">● Running</div>
            <p style="margin-bottom: 20px;">AI agents learn to clean dirty tabular data through step/reset/state API.</p>
            <div class="tasks">
                <div class="task"><h4>Easy</h4><div class="score">5</div><div>errors</div></div>
                <div class="task"><h4>Medium</h4><div class="score">12</div><div>errors</div></div>
                <div class="task"><h4>Hard</h4><div class="score">20</div><div>errors</div></div>
            </div>
            <div class="endpoints">
                <h3>API Endpoints</h3>
                <div class="endpoint"><span class="method">GET</span> <a href="/health">/health</a> — Health check</div>
                <div class="endpoint"><span class="method post">POST</span> /reset — Reset environment</div>
                <div class="endpoint"><span class="method post">POST</span> /step — Execute action</div>
                <div class="endpoint"><span class="method">GET</span> <a href="/state">/state</a> — Get state</div>
                <div class="endpoint"><span class="method">GET</span> <a href="/grade">/grade</a> — Grade episode</div>
                <div class="endpoint"><span class="method">GET</span> <a href="/tasks">/tasks</a> — List tasks</div>
                <div class="endpoint"><span class="method">GET</span> <a href="/docs">/docs</a> — API documentation</div>
            </div>
            <p style="margin-top: 20px; color: #64748b; font-size: 0.9em;">Built for the Meta × HuggingFace OpenEnv Hackathon</p>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "data_clean_env"}


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    """
    Reset the environment with a new dirty dataset.
    Specify task_id: 'easy', 'medium', or 'hard'.
    """
    obs = env.reset(task_id=req.task_id, seed=req.seed)
    return obs


@app.post("/step")
async def step(req: ActionRequest):
    """
    Execute a cleaning action on the dataset.

    action_type: fix_value | delete_row | fill_missing | standardize | done
    row_index: Target row index
    column_name: Target column name
    new_value: Replacement value
    reason: Agent's reasoning (logged, not graded)
    """
    obs = env.step(req.model_dump())
    return obs


@app.get("/state")
async def state():
    """Get current environment state."""
    return env.get_state()


@app.get("/grade")
async def grade():
    """Grade the current episode."""
    env_state = env.get_state()
    task_id = env_state.get("task_id", "easy")
    return grade_task(task_id, env_state)


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    from server.data_generator import TASK_CONFIG
    return {
        "tasks": [
            {
                "task_id": tid,
                "description": cfg["description"],
                "num_rows": cfg["num_rows"],
                "max_steps": cfg["max_steps"],
            }
            for tid, cfg in TASK_CONFIG.items()
        ]
    }


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
