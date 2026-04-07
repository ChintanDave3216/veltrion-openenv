"""
FastAPI application for the DataClean Environment.

Exposes HTTP endpoints for reset/step/state/health.
Compatible with OpenEnv validation and standard clients.
"""

from fastapi import FastAPI
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
