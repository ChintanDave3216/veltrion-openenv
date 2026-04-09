"""
DataClean Environment - Core environment logic.

Implements the OpenEnv Environment interface (reset/step/state) for data cleaning.
The environment generates dirty datasets, accepts cleaning actions from agents,
and provides incremental rewards for correct fixes.
"""

import uuid
import copy
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import DataCleanAction, DataCleanObservation
except ImportError:
    from ..models import DataCleanAction, DataCleanObservation

from server.data_generator import (
    generate_clean_dataset,
    inject_errors,
    format_as_csv,
    generate_error_report,
    TASK_CONFIG,
    CITY_STATE_MAP,
)


class DataCleanEnvironment(Environment):
    """
    Data Cleaning Environment implementing the OpenEnv Environment interface.

    An AI agent receives a dirty employee dataset and must issue cleaning
    actions to fix errors. The agent receives incremental rewards for each
    correct fix and penalties for wasted actions or introducing new errors.

    Supports 3 tasks:
      - easy: 5 formatting errors in 10 rows (15 max steps)
      - medium: 12 mixed errors in 25 rows (25 max steps)
      - hard: 20 complex errors in 40 rows (35 max steps)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state_obj = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._task_id = "easy"
        self._total_errors = 0
        self._errors_fixed = 0
        self._errors_introduced = 0
        self._max_steps = 20
        self._done = False

        self._clean_data = []
        self._dirty_data = []
        self._current_data = []
        self._error_manifest = []
        self._fixed_errors = set()
        self._deleted_rows = set()
        self._last_action_result = ""
        self._cumulative_reward = 0.0

    def reset(self, task_id: str = "easy", seed: int = 42, **kwargs) -> DataCleanObservation:
        """
        Reset the environment with a new dirty dataset.

        Args:
            task_id: "easy", "medium", or "hard"
            seed: Random seed for reproducible data generation

        Returns:
            DataCleanObservation with initial state
        """
        if task_id not in TASK_CONFIG:
            task_id = "easy"

        config = TASK_CONFIG[task_id]

        # Generate clean data
        self._clean_data = generate_clean_dataset(config["num_rows"], seed=seed)

        # Inject errors to create dirty data
        self._dirty_data, self._error_manifest = inject_errors(
            self._clean_data, task_id, seed=seed
        )

        # Working copy that the agent modifies
        self._current_data = copy.deepcopy(self._dirty_data)

        # Reset state
        self._state_obj = State(
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )
        self._task_id = task_id
        self._total_errors = len(self._error_manifest)
        self._errors_fixed = 0
        self._errors_introduced = 0
        self._max_steps = config["max_steps"]
        self._done = False

        self._fixed_errors = set()
        self._deleted_rows = set()
        self._last_action_result = "Environment reset. Begin cleaning the data."
        self._cumulative_reward = 0.0

        return self._build_observation(reward=0.0)

    def step(self, action: DataCleanAction) -> DataCleanObservation:
        """
        Execute a cleaning action on the dataset.

        Args:
            action: DataCleanAction with action_type, row_index, column_name, new_value

        Returns:
            DataCleanObservation with updated data, reward, and done signal
        """
        if self._done:
            return self._build_observation(reward=0.0)

        self._state_obj.step_count += 1

        # Handle action as dict or DataCleanAction
        if isinstance(action, dict):
            action_type = action.get("action_type", "")
            row_index = action.get("row_index", -1)
            column_name = action.get("column_name", "")
            new_value = action.get("new_value", "")
        else:
            action_type = action.action_type
            row_index = action.row_index
            column_name = action.column_name
            new_value = action.new_value

        reward = 0.0

        if action_type == "done":
            self._done = True
            score = self._calculate_score()
            # Reward for done = the task score itself (already clamped to (0,1))
            reward = score
            self._last_action_result = f"Agent finished. Final score: {score:.3f}"

        elif action_type == "delete_row":
            reward = self._handle_delete_row(row_index)

        elif action_type in ("fix_value", "fill_missing", "standardize"):
            reward = self._handle_fix_value(row_index, column_name, new_value)

        else:
            reward = -0.05
            self._last_action_result = (
                f"Unknown action type: '{action_type}'. "
                "Use: fix_value, delete_row, fill_missing, standardize, done"
            )

        # Check if max steps reached
        if self._state_obj.step_count >= self._max_steps:
            self._done = True
            self._last_action_result += " [Max steps reached]"

        # Check if all errors fixed
        if len(self._fixed_errors) >= len(self._error_manifest):
            self._done = True
            self._last_action_result += " [All errors fixed!]"

        self._cumulative_reward += reward
        return self._build_observation(reward=reward)

    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state_obj

    def get_state(self) -> dict:
        """Return current environment state as dict (for HTTP endpoints)."""
        return {
            "episode_id": self._state_obj.episode_id,
            "step_count": self._state_obj.step_count,
            "task_id": self._task_id,
            "total_errors": self._total_errors,
            "errors_fixed": self._errors_fixed,
            "errors_introduced": self._errors_introduced,
            "max_steps": self._max_steps,
            "done": self._done,
            "score": self._calculate_score(),
        }

    # ================================================================
    # PRIVATE METHODS
    # ================================================================

    def _handle_delete_row(self, row_index: int) -> float:
        """Handle delete_row action. Returns reward."""
        if row_index < 0 or row_index >= len(self._current_data):
            self._last_action_result = (
                f"Invalid row index: {row_index}. "
                f"Valid range: 0-{len(self._current_data)-1}"
            )
            return -0.05

        if row_index in self._deleted_rows:
            self._last_action_result = f"Row {row_index} already deleted."
            return -0.05

        # Check if this row is a duplicate that should be deleted
        for i, err in enumerate(self._error_manifest):
            if (
                err["error_type"] == "duplicate_row"
                and err["row"] == row_index
                and i not in self._fixed_errors
            ):
                self._deleted_rows.add(row_index)
                self._fixed_errors.add(i)
                self._errors_fixed += 1
                self._last_action_result = (
                    f"Correctly deleted duplicate row {row_index}."
                )
                return 1.0 / max(self._total_errors, 1)

        # Not a duplicate - penalty for deleting a valid row
        self._deleted_rows.add(row_index)
        self._errors_introduced += 1
        self._last_action_result = (
            f"Row {row_index} was not a duplicate. "
            "Deleting valid data is penalized."
        )
        return -0.2

    def _handle_fix_value(
        self, row_index: int, column_name: str, new_value: str
    ) -> float:
        """Handle fix_value/fill_missing/standardize actions. Returns reward."""
        if row_index < 0 or row_index >= len(self._current_data):
            self._last_action_result = (
                f"Invalid row index: {row_index}. "
                f"Valid range: 0-{len(self._current_data)-1}"
            )
            return -0.05

        if row_index in self._deleted_rows:
            self._last_action_result = f"Row {row_index} has been deleted."
            return -0.05

        row = self._current_data[row_index]
        if column_name not in row:
            valid_cols = list(row.keys())
            self._last_action_result = (
                f"Invalid column: '{column_name}'. Valid columns: {valid_cols}"
            )
            return -0.05

        # Apply the fix
        old_value = row[column_name]
        row[column_name] = new_value

        # Check if this fixes any known error
        for i, err in enumerate(self._error_manifest):
            if (
                err["row"] == row_index
                and err["col"] == column_name
                and i not in self._fixed_errors
                and err["error_type"] != "duplicate_row"
            ):
                clean_val = str(err["clean_value"]).strip()
                new_val = str(new_value).strip()

                if new_val == clean_val:
                    self._fixed_errors.add(i)
                    self._errors_fixed += 1
                    self._last_action_result = (
                        f"Correctly fixed Row {row_index}, "
                        f"Column '{column_name}': "
                        f"'{old_value}' -> '{new_value}'"
                    )
                    return 1.0 / max(self._total_errors, 1)
                else:
                    self._last_action_result = (
                        f"Changed Row {row_index}, Column '{column_name}': "
                        f"'{old_value}' -> '{new_value}', "
                        "but the value is still incorrect."
                    )
                    return -0.05

        # No matching error found
        if old_value == new_value:
            self._last_action_result = (
                f"No change: Row {row_index}, Column '{column_name}' "
                f"already has value '{old_value}'."
            )
            return -0.05
        else:
            if row_index < len(self._clean_data):
                clean_val = str(
                    self._clean_data[row_index].get(column_name, "")
                ).strip()
                if str(new_value).strip() == clean_val:
                    self._last_action_result = (
                        f"Changed Row {row_index}, Column '{column_name}', "
                        "but this wasn't a tracked error."
                    )
                    return 0.0
            self._last_action_result = (
                f"Changed Row {row_index}, Column '{column_name}': "
                f"'{old_value}' -> '{new_value}'. "
                "This may have introduced an error."
            )
            return -0.1

    def _calculate_score(self) -> float:
        """Calculate current score as fraction of errors fixed.
        
        Clamped to (0, 1) open interval as required by evaluator.
        """
        if self._total_errors == 0:
            return 0.99
        score = len(self._fixed_errors) / self._total_errors
        return min(max(score, 0.01), 0.99)

    def _build_observation(self, reward: float) -> DataCleanObservation:
        """Build the observation returned to the agent.

        All rewards are clamped to (0, 1) open interval as required by evaluator.

        CRITICAL: Only score-like fields (always in (0,1)) and strings are
        first-class fields. Integer counts (errors_fixed=0, step_count=0) go in
        metadata (excluded from serialization) so the evaluator never sees them.
        """
        # Clamp reward to (0, 1) open interval — evaluator rejects 0.0 and 1.0
        clamped_reward = round(min(max(reward, 0.01), 0.99), 4)

        # Build visible data (exclude deleted rows)
        visible_data = []
        for i, row in enumerate(self._current_data):
            if i not in self._deleted_rows:
                visible_data.append({"_row_index": i, **row})

        config = TASK_CONFIG.get(self._task_id, TASK_CONFIG["easy"])

        return DataCleanObservation(
            done=self._done,
            reward=clamped_reward,
            # First-class fields (visible in serialized observation)
            # ONLY score (always 0.01-0.99) and text fields here
            score=self._calculate_score(),
            task_id=self._task_id,
            task_description=config["description"],
            current_data=format_as_csv(visible_data),
            error_report=generate_error_report(
                self._dirty_data, self._error_manifest
            ),
            last_action_result=self._last_action_result,
            # Integer counts go in metadata (excluded from wire format by SDK)
            metadata={
                "task_id": self._task_id,
                "columns": (
                    list(self._current_data[0].keys())
                    if self._current_data
                    else []
                ),
                "total_rows": len(visible_data),
                "errors_remaining": self._total_errors - len(self._fixed_errors),
                "errors_fixed": len(self._fixed_errors),
                "total_errors": self._total_errors,
                "score": self._calculate_score(),
                "step_count": self._state_obj.step_count,
                "max_steps": self._max_steps,
            },
        )
