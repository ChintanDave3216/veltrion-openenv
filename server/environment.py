"""
DataClean Environment - Core environment logic.

Implements the reset/step/state API for the data cleaning environment.
The environment generates dirty datasets, accepts cleaning actions from agents,
and provides incremental rewards for correct fixes.
"""

import uuid
import copy
from typing import Optional
from server.data_generator import (
    generate_clean_dataset,
    inject_errors,
    format_as_csv,
    generate_error_report,
    TASK_CONFIG,
    CITY_STATE_MAP,
)


class DataCleanEnvironment:
    """
    Data Cleaning Environment.

    An AI agent receives a dirty employee dataset and must issue cleaning
    actions to fix errors. The agent receives incremental rewards for each
    correct fix and penalties for wasted actions or introducing new errors.

    Supports 3 tasks:
      - easy: 5 formatting errors in 10 rows (15 max steps)
      - medium: 12 mixed errors in 25 rows (25 max steps)
      - hard: 20 complex errors in 40 rows (35 max steps)
    """

    def __init__(self):
        self._state = {
            "episode_id": "",
            "step_count": 0,
            "task_id": "",
            "total_errors": 0,
            "errors_fixed": 0,
            "errors_introduced": 0,
            "max_steps": 20,
            "done": False,
        }
        self._clean_data = []
        self._dirty_data = []
        self._current_data = []
        self._error_manifest = []
        self._fixed_errors = set()     # indices of fixed errors
        self._deleted_rows = set()     # indices of deleted rows
        self._last_action_result = ""
        self._cumulative_reward = 0.0

    def reset(self, task_id: str = "easy", seed: int = 42, **kwargs) -> dict:
        """
        Reset the environment with a new dirty dataset.

        Args:
            task_id: "easy", "medium", or "hard"
            seed: Random seed for reproducible data generation

        Returns:
            Initial observation dict
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
        self._state = {
            "episode_id": str(uuid.uuid4()),
            "step_count": 0,
            "task_id": task_id,
            "total_errors": len(self._error_manifest),
            "errors_fixed": 0,
            "errors_introduced": 0,
            "max_steps": config["max_steps"],
            "done": False,
        }
        self._fixed_errors = set()
        self._deleted_rows = set()
        self._last_action_result = "Environment reset. Begin cleaning the data."
        self._cumulative_reward = 0.0

        return self._build_observation(reward=0.0)

    def step(self, action: dict) -> dict:
        """
        Execute a cleaning action on the dataset.

        Args:
            action: dict with keys:
                action_type: "fix_value" | "delete_row" | "fill_missing" | "standardize" | "done"
                row_index: int (target row, -1 for N/A)
                column_name: str (target column, "" for N/A)
                new_value: str (replacement value)
                reason: str (agent's reasoning - logged but not graded)

        Returns:
            Observation dict with updated data, reward, and done signal
        """
        if self._state["done"]:
            return self._build_observation(reward=0.0)

        self._state["step_count"] += 1
        action_type = action.get("action_type", "")
        row_index = action.get("row_index", -1)
        column_name = action.get("column_name", "")
        new_value = action.get("new_value", "")

        reward = 0.0

        if action_type == "done":
            # Agent signals it's finished
            self._state["done"] = True
            # Bonus if score is high
            score = self._calculate_score()
            if score >= 0.8:
                reward = 0.1
            self._last_action_result = f"Agent finished. Final score: {score:.3f}"

        elif action_type == "delete_row":
            reward = self._handle_delete_row(row_index)

        elif action_type in ("fix_value", "fill_missing", "standardize"):
            reward = self._handle_fix_value(row_index, column_name, new_value)

        else:
            reward = -0.05
            self._last_action_result = f"Unknown action type: '{action_type}'. Use: fix_value, delete_row, fill_missing, standardize, done"

        # Check if max steps reached
        if self._state["step_count"] >= self._state["max_steps"]:
            self._state["done"] = True
            self._last_action_result += " [Max steps reached]"

        # Check if all errors fixed
        if len(self._fixed_errors) >= len(self._error_manifest):
            self._state["done"] = True
            self._last_action_result += " [All errors fixed!]"

        self._cumulative_reward += reward
        return self._build_observation(reward=reward)

    def get_state(self) -> dict:
        """Return current environment state."""
        return {**self._state, "score": self._calculate_score()}

    # ================================================================
    # PRIVATE METHODS
    # ================================================================

    def _handle_delete_row(self, row_index: int) -> float:
        """Handle delete_row action. Returns reward."""
        if row_index < 0 or row_index >= len(self._current_data):
            self._last_action_result = f"Invalid row index: {row_index}. Valid range: 0-{len(self._current_data)-1}"
            return -0.05

        if row_index in self._deleted_rows:
            self._last_action_result = f"Row {row_index} already deleted."
            return -0.05

        # Check if this row is a duplicate that should be deleted
        for i, err in enumerate(self._error_manifest):
            if err["error_type"] == "duplicate_row" and err["row"] == row_index and i not in self._fixed_errors:
                self._deleted_rows.add(row_index)
                self._fixed_errors.add(i)
                self._state["errors_fixed"] += 1
                self._last_action_result = f"Correctly deleted duplicate row {row_index}."
                return 1.0 / max(self._state["total_errors"], 1)

        # Not a duplicate - penalty for deleting a valid row
        self._deleted_rows.add(row_index)
        self._state["errors_introduced"] += 1
        self._last_action_result = f"Row {row_index} was not a duplicate. Deleting valid data is penalized."
        return -0.2

    def _handle_fix_value(self, row_index: int, column_name: str, new_value: str) -> float:
        """Handle fix_value/fill_missing/standardize actions. Returns reward."""
        if row_index < 0 or row_index >= len(self._current_data):
            self._last_action_result = f"Invalid row index: {row_index}. Valid range: 0-{len(self._current_data)-1}"
            return -0.05

        if row_index in self._deleted_rows:
            self._last_action_result = f"Row {row_index} has been deleted."
            return -0.05

        row = self._current_data[row_index]
        if column_name not in row:
            valid_cols = list(row.keys())
            self._last_action_result = f"Invalid column: '{column_name}'. Valid columns: {valid_cols}"
            return -0.05

        # Apply the fix
        old_value = row[column_name]
        row[column_name] = new_value

        # Check if this fixes any known error
        for i, err in enumerate(self._error_manifest):
            if (err["row"] == row_index and
                err["col"] == column_name and
                i not in self._fixed_errors and
                err["error_type"] != "duplicate_row"):

                # Check if the new value matches the clean value
                clean_val = str(err["clean_value"]).strip()
                new_val = str(new_value).strip()

                if new_val == clean_val:
                    self._fixed_errors.add(i)
                    self._state["errors_fixed"] += 1
                    self._last_action_result = (
                        f"Correctly fixed Row {row_index}, Column '{column_name}': "
                        f"'{old_value}' → '{new_value}'"
                    )
                    return 1.0 / max(self._state["total_errors"], 1)
                else:
                    self._last_action_result = (
                        f"Changed Row {row_index}, Column '{column_name}': "
                        f"'{old_value}' → '{new_value}', but the value is still incorrect."
                    )
                    return -0.05

        # No matching error found - this action had no effect or introduced an error
        if old_value == new_value:
            self._last_action_result = f"No change: Row {row_index}, Column '{column_name}' already has value '{old_value}'."
            return -0.05
        else:
            # Check if this matches the clean data (fixing something not in manifest)
            if row_index < len(self._clean_data):
                clean_val = str(self._clean_data[row_index].get(column_name, "")).strip()
                if str(new_value).strip() == clean_val:
                    self._last_action_result = f"Changed Row {row_index}, Column '{column_name}', but this wasn't a tracked error."
                    return 0.0  # Neutral
            self._last_action_result = (
                f"Changed Row {row_index}, Column '{column_name}': '{old_value}' → '{new_value}'. "
                f"This may have introduced an error."
            )
            return -0.1

    def _calculate_score(self) -> float:
        """Calculate current score as fraction of errors fixed."""
        if self._state["total_errors"] == 0:
            return 1.0
        score = len(self._fixed_errors) / self._state["total_errors"]
        return min(max(score, 0.0), 1.0)

    def _build_observation(self, reward: float) -> dict:
        """Build the observation dict returned to the agent."""
        # Build visible data (exclude deleted rows)
        visible_data = []
        for i, row in enumerate(self._current_data):
            if i not in self._deleted_rows:
                visible_data.append({"_row_index": i, **row})

        config = TASK_CONFIG.get(self._state["task_id"], TASK_CONFIG["easy"])

        return {
            "done": self._state["done"],
            "reward": round(reward, 4),
            "metadata": {
                "task_id": self._state["task_id"],
                "task_description": config["description"],
                "current_data": format_as_csv(visible_data),
                "error_report": generate_error_report(self._dirty_data, self._error_manifest),
                "columns": list(self._current_data[0].keys()) if self._current_data else [],
                "total_rows": len(visible_data),
                "errors_remaining": self._state["total_errors"] - len(self._fixed_errors),
                "errors_fixed": len(self._fixed_errors),
                "total_errors": self._state["total_errors"],
                "last_action_result": self._last_action_result,
                "score": self._calculate_score(),
                "step_count": self._state["step_count"],
                "max_steps": self._state["max_steps"],
            },
        }
