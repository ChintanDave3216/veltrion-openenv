"""
Typed models for DataCleanEnv.

Action and Observation types using the OpenEnv base classes.
These are used by the server (create_app) and can be used by typed clients.
"""

from typing import Any, Dict, Optional
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
    All task-specific data is stored in the `metadata` dict.
    """

    pass
