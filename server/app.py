"""
FastAPI application for the DataClean Environment.

Uses openenv's create_app() to automatically provide:
  - POST /reset: Reset the environment
  - POST /step: Execute an action
  - GET /state: Get current environment state
  - GET /schema: Get action/observation schemas
  - WS /ws: WebSocket endpoint for persistent sessions
  - GET /health: Health check

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import sys
import os

# Ensure project root is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from models import DataCleanAction, DataCleanObservation
from server.environment import DataCleanEnvironment

# Create the app using OpenEnv's factory — this adds /reset, /step, /state, /ws, etc.
app = create_app(
    DataCleanEnvironment,
    DataCleanAction,
    DataCleanObservation,
    env_name="data_clean_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
