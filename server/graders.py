"""
Task Graders for DataCleanEnv.

Each grader evaluates an agent's performance on a specific task.
Returns a score between 0.0 and 1.0 based on how many errors were correctly fixed.
Graders are deterministic and reproducible.
"""


class TaskGrader:
    """
    Deterministic grader that scores agent performance.

    Score = errors_correctly_fixed / total_errors
    Clamped to [0.0, 1.0]
    """

    def grade(self, env_state: dict) -> float:
        """
        Grade the agent's performance based on the environment state.

        Args:
            env_state: The environment state dict from get_state()

        Returns:
            Score between 0.0 and 1.0
        """
        total_errors = env_state.get("total_errors", 0)
        errors_fixed = env_state.get("errors_fixed", 0)

        if total_errors == 0:
            return 1.0

        score = errors_fixed / total_errors
        return min(max(score, 0.0), 1.0)


class EasyGrader(TaskGrader):
    """Grader for the easy task: Format Fixer."""

    TASK_ID = "easy"
    DESCRIPTION = "Fix formatting errors (whitespace, dates, case, typos, phone formats)"
    EXPECTED_ERRORS = 5
    MAX_STEPS = 15

    def grade(self, env_state: dict) -> float:
        return super().grade(env_state)


class MediumGrader(TaskGrader):
    """Grader for the medium task: Data Quality Resolver."""

    TASK_ID = "medium"
    DESCRIPTION = "Fix data quality issues (missing values, duplicates, format errors)"
    EXPECTED_ERRORS = 12
    MAX_STEPS = 25

    def grade(self, env_state: dict) -> float:
        return super().grade(env_state)


class HardGrader(TaskGrader):
    """Grader for the hard task: Cross-Column Consistency Auditor."""

    TASK_ID = "hard"
    DESCRIPTION = "Fix complex errors (cross-column mismatches, logical impossibilities)"
    EXPECTED_ERRORS = 20
    MAX_STEPS = 35

    def grade(self, env_state: dict) -> float:
        return super().grade(env_state)


GRADERS = {
    "easy": EasyGrader(),
    "medium": MediumGrader(),
    "hard": HardGrader(),
}


def grade_task(task_id: str, env_state: dict) -> dict:
    """
    Grade a completed task.

    Args:
        task_id: "easy", "medium", or "hard"
        env_state: Environment state dict

    Returns:
        Dict with grading results
    """
    grader = GRADERS.get(task_id, TaskGrader())
    score = grader.grade(env_state)

    return {
        "task_id": task_id,
        "score": round(score, 4),
        "errors_fixed": env_state.get("errors_fixed", 0),
        "total_errors": env_state.get("total_errors", 0),
        "steps_taken": env_state.get("step_count", 0),
        "max_steps": env_state.get("max_steps", 0),
        "passed": score >= 0.6,
    }
