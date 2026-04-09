"""
Typed models for DataCleanEnv.

Action and Observation types using the OpenEnv base classes.
These are used by the server (create_app) and can be used by typed clients.

IMPORTANT: The OpenEnv SDK's serialize_observation() EXCLUDES `done`, `reward`,
and `metadata` from the observation dict sent to clients. All task-specific data
MUST be defined as first-class fields on the Observation subclass.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class DataCleanAction(Action):
    """Action for the data cleaning environment."""

    action_type: str = Field(
        default="done",
        description="One of: fix_value, delete_row, fill_missing, standardize, done",
    )
    row_index: int = Field(
        default=-1,
        description="Target row index (from _row_index column)",
    )
    column_name: str = Field(
        default="",
        description="Target column name",
    )
    new_value: str = Field(
        default="",
        description="Replacement value for the cell",
    )
    reason: str = Field(
        default="",
        description="Agent's reasoning (logged, not graded)",
    )


class DataCleanObservation(Observation):
    """Observation returned by the data cleaning environment.

    Inherits `done`, `reward`, and `metadata` from the base Observation class.

    All task-specific data is defined as first-class fields so that
    serialize_observation() includes them in the response sent to clients.
    (The SDK excludes `done`, `reward`, and `metadata` from the observation dict.)
    """

    # Task info
    task_id: str = Field(default="", description="Current task ID")
    task_description: str = Field(default="", description="Task description")

    # Data
    current_data: str = Field(default="", description="Current CSV data to clean")
    error_report: str = Field(default="", description="Error analysis report")
    columns: List[str] = Field(default_factory=list, description="Column names")
    total_rows: int = Field(default=0, description="Number of visible rows")

    # Progress
    errors_remaining: int = Field(default=0, description="Errors left to fix")
    errors_fixed: int = Field(default=0, description="Errors successfully fixed")
    total_errors: int = Field(default=0, description="Total errors in dataset")

    # Status
    last_action_result: str = Field(default="", description="Result of last action")
    score: float = Field(default=0.001, description="Current score (0,1) open interval")
    step_count: int = Field(default=0, description="Steps taken so far")
    max_steps: int = Field(default=0, description="Maximum allowed steps")
