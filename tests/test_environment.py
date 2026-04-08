"""
Unit tests for the DataClean Environment.

Tests reset, step, state, grading, and data generation.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import DataCleanEnvironment
from server.data_generator import generate_clean_dataset, inject_errors, TASK_CONFIG
from server.graders import grade_task


def obs_to_dict(obs):
    """Convert DataCleanObservation to dict for backward compat."""
    if isinstance(obs, dict):
        return obs
    return {
        "done": obs.done,
        "reward": obs.reward,
        "metadata": obs.metadata,
    }


class TestDataGenerator:
    """Tests for the data generation engine."""

    def test_clean_dataset_generation(self):
        """Clean dataset has correct number of rows and columns."""
        data = generate_clean_dataset(10, seed=42)
        assert len(data) == 10
        assert "name" in data[0]
        assert "email" in data[0]
        assert "phone" in data[0]
        assert "department" in data[0]
        assert "salary" in data[0]
        assert "hire_date" in data[0]
        assert "city" in data[0]
        assert "state" in data[0]

    def test_deterministic_generation(self):
        """Same seed produces same data."""
        data1 = generate_clean_dataset(10, seed=42)
        data2 = generate_clean_dataset(10, seed=42)
        assert data1 == data2

    def test_different_seeds_different_data(self):
        """Different seeds produce different data."""
        data1 = generate_clean_dataset(10, seed=42)
        data2 = generate_clean_dataset(10, seed=99)
        assert data1 != data2

    def test_easy_error_injection(self):
        """Easy task injects exactly 5 errors."""
        clean = generate_clean_dataset(10, seed=42)
        dirty, manifest = inject_errors(clean, "easy", seed=42)
        assert len(manifest) == 5

    def test_medium_error_injection(self):
        """Medium task injects 12 errors (including duplicates)."""
        clean = generate_clean_dataset(25, seed=42)
        dirty, manifest = inject_errors(clean, "medium", seed=42)
        assert len(manifest) == 12
        assert len(dirty) == 27

    def test_hard_error_injection(self):
        """Hard task injects 20 errors."""
        clean = generate_clean_dataset(40, seed=42)
        dirty, manifest = inject_errors(clean, "hard", seed=42)
        assert len(manifest) == 20

    def test_error_manifest_structure(self):
        """Each error in manifest has required keys."""
        clean = generate_clean_dataset(10, seed=42)
        _, manifest = inject_errors(clean, "easy", seed=42)
        for err in manifest:
            assert "row" in err
            assert "col" in err
            assert "error_type" in err
            assert "dirty_value" in err
            assert "clean_value" in err

    def test_deterministic_errors(self):
        """Same seed produces same errors."""
        clean = generate_clean_dataset(10, seed=42)
        _, m1 = inject_errors(clean, "easy", seed=42)
        _, m2 = inject_errors(clean, "easy", seed=42)
        assert m1 == m2


class TestEnvironment:
    """Tests for the core environment."""

    def setup_method(self):
        self.env = DataCleanEnvironment()

    def test_reset_returns_observation(self):
        """Reset returns a valid observation."""
        obs = obs_to_dict(self.env.reset(task_id="easy", seed=42))
        assert "done" in obs
        assert "reward" in obs
        assert "metadata" in obs
        assert obs["done"] is False
        assert obs["reward"] == 0.0

    def test_reset_metadata_fields(self):
        """Reset observation contains all required metadata."""
        obs = obs_to_dict(self.env.reset(task_id="easy", seed=42))
        meta = obs["metadata"]
        assert "task_id" in meta
        assert "task_description" in meta
        assert "current_data" in meta
        assert "error_report" in meta
        assert "errors_remaining" in meta
        assert "errors_fixed" in meta
        assert "total_errors" in meta
        assert meta["task_id"] == "easy"
        assert meta["total_errors"] == 5

    def test_reset_cleans_state(self):
        """Reset produces clean state with no prior data leakage."""
        self.env.reset(task_id="easy", seed=42)
        self.env.step({"action_type": "done"})
        obs = obs_to_dict(self.env.reset(task_id="medium", seed=42))
        assert obs["done"] is False
        assert obs["metadata"]["task_id"] == "medium"

    def test_step_fix_value(self):
        """Fixing a known error gives positive reward."""
        self.env.reset(task_id="easy", seed=42)
        err = self.env._error_manifest[0]
        action = {
            "action_type": "fix_value",
            "row_index": err["row"],
            "column_name": err["col"],
            "new_value": err["clean_value"],
        }
        obs = obs_to_dict(self.env.step(action))
        assert obs["reward"] > 0

    def test_step_invalid_action(self):
        """Invalid action type gives penalty."""
        self.env.reset(task_id="easy", seed=42)
        obs = obs_to_dict(self.env.step({"action_type": "invalid_action"}))
        assert obs["reward"] < 0

    def test_step_done_action(self):
        """Done action ends the episode."""
        self.env.reset(task_id="easy", seed=42)
        obs = obs_to_dict(self.env.step({"action_type": "done"}))
        assert obs["done"] is True

    def test_max_steps_terminates(self):
        """Episode terminates after max steps."""
        self.env.reset(task_id="easy", seed=42)
        for i in range(20):
            obs = obs_to_dict(self.env.step({
                "action_type": "fix_value",
                "row_index": 0,
                "column_name": "name",
                "new_value": "test",
            }))
            if obs["done"]:
                break
        assert obs["done"] is True

    def test_get_state(self):
        """State returns correct structure."""
        self.env.reset(task_id="easy", seed=42)
        state = self.env.get_state()
        assert "episode_id" in state
        assert "step_count" in state
        assert "task_id" in state
        assert "total_errors" in state
        assert state["task_id"] == "easy"

    def test_delete_duplicate_row(self):
        """Deleting a duplicate row gives positive reward."""
        self.env.reset(task_id="medium", seed=42)
        for err in self.env._error_manifest:
            if err["error_type"] == "duplicate_row":
                action = {"action_type": "delete_row", "row_index": err["row"]}
                obs = obs_to_dict(self.env.step(action))
                assert obs["reward"] > 0
                break

    def test_delete_valid_row_penalized(self):
        """Deleting a non-duplicate row gives penalty."""
        self.env.reset(task_id="easy", seed=42)
        obs = obs_to_dict(self.env.step({"action_type": "delete_row", "row_index": 0}))
        assert obs["reward"] < 0

    def test_fix_all_errors_terminates(self):
        """Fixing all errors ends the episode."""
        self.env.reset(task_id="easy", seed=42)
        for err in self.env._error_manifest:
            if err["error_type"] == "duplicate_row":
                self.env.step({"action_type": "delete_row", "row_index": err["row"]})
            else:
                self.env.step({
                    "action_type": "fix_value",
                    "row_index": err["row"],
                    "column_name": err["col"],
                    "new_value": err["clean_value"],
                })
        state = self.env.get_state()
        assert state["done"] is True

    def test_reward_range(self):
        """Rewards stay in reasonable range."""
        self.env.reset(task_id="easy", seed=42)
        for i in range(15):
            obs = obs_to_dict(self.env.step({
                "action_type": "fix_value",
                "row_index": 0,
                "column_name": "name",
                "new_value": "X",
            }))
            assert -1.0 <= obs["reward"] <= 1.0
            if obs["done"]:
                break

    def test_state_property(self):
        """State property returns State object."""
        self.env.reset(task_id="easy", seed=42)
        state = self.env.state
        assert hasattr(state, "episode_id")
        assert hasattr(state, "step_count")
        assert state.step_count == 0


class TestGraders:
    """Tests for task graders."""

    def test_grader_zero_score(self):
        """No errors fixed = 0.0 score."""
        env = DataCleanEnvironment()
        env.reset(task_id="easy", seed=42)
        env.step({"action_type": "done"})
        result = grade_task("easy", env.get_state())
        assert result["score"] == 0.0

    def test_grader_perfect_score(self):
        """All errors fixed = 1.0 score."""
        env = DataCleanEnvironment()
        env.reset(task_id="easy", seed=42)
        for err in env._error_manifest:
            if err["error_type"] == "duplicate_row":
                env.step({"action_type": "delete_row", "row_index": err["row"]})
            else:
                env.step({
                    "action_type": "fix_value",
                    "row_index": err["row"],
                    "column_name": err["col"],
                    "new_value": err["clean_value"],
                })
        result = grade_task("easy", env.get_state())
        assert result["score"] == 1.0

    def test_grader_partial_score(self):
        """Fixing some errors = partial score."""
        env = DataCleanEnvironment()
        env.reset(task_id="easy", seed=42)
        err = env._error_manifest[0]
        env.step({
            "action_type": "fix_value",
            "row_index": err["row"],
            "column_name": err["col"],
            "new_value": err["clean_value"],
        })
        env.step({"action_type": "done"})
        result = grade_task("easy", env.get_state())
        assert 0.0 < result["score"] < 1.0

    def test_grader_deterministic(self):
        """Same actions produce same score."""
        scores = []
        for _ in range(3):
            env = DataCleanEnvironment()
            env.reset(task_id="easy", seed=42)
            err = env._error_manifest[0]
            env.step({
                "action_type": "fix_value",
                "row_index": err["row"],
                "column_name": err["col"],
                "new_value": err["clean_value"],
            })
            env.step({"action_type": "done"})
            result = grade_task("easy", env.get_state())
            scores.append(result["score"])
        assert scores[0] == scores[1] == scores[2]

    def test_all_tasks_have_graders(self):
        """All 3 tasks have working graders."""
        for task_id in ["easy", "medium", "hard"]:
            env = DataCleanEnvironment()
            env.reset(task_id=task_id, seed=42)
            env.step({"action_type": "done"})
            result = grade_task(task_id, env.get_state())
            assert "score" in result
            assert 0.0 <= result["score"] <= 1.0
