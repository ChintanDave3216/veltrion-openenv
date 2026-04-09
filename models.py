"""
Typed models for DataCleanEnv.

Action and Observation types using the OpenEnv base classes.
These are used by the server (create_app) and can be used by typed clients.

IMPORTANT: The OpenEnv SDK's serialize_observation() EXCLUDES `done`, `reward`,
and `metadata` from the observation dict sent to clients. Only first-class fields
defined here will appear in the client's observation.

The evaluator checks ALL numeric values in the observation for (0, 1) compliance.
Therefore, only include SCORE-like fields (always in (0,1)) as first-class fields.
Use metadata for integer counts, text data, etc.
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

    ONLY score-like fields (always strictly in (0,1)) should be first-class fields.
    Integer counts and text should go in metadata (which is excluded from serialization).
    """

    # The only first-class numeric field: always clamped to (0.01, 0.99)
    score: float = Field(default=0.01, description="Current score in (0,1) open interval")

    # Text fields (safe — evaluator only checks numeric values)
    task_id: str = Field(default="", description="Current task ID")
    task_description: str = Field(default="", description="Task description")
    current_data: str = Field(default="", description="Current CSV data to clean")
    error_report: str = Field(default="", description="Error analysis report")
    last_action_result: str = Field(default="", description="Result of last action")
