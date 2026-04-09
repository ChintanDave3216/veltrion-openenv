"""
Graders for DataCleanEnv tasks.

Each grader is a top-level function that accepts *args, **kwargs
and returns a float score strictly in (0, 1) open interval.

The evaluator discovers these via openenv.yaml:
  grader: "server.graders:easy_grader"
"""


def safe_extract(args, kwargs, key: str, default: float) -> float:
    """Safely extract a value from trajectory/state data passed by evaluator."""
    try:
        data = args[0] if args else kwargs.get('trajectory', kwargs.get('state'))
        last_step = data[-1] if isinstance(data, list) else data

        obs = getattr(last_step, 'observation', last_step)
        if hasattr(obs, 'model_dump'):
            obs = obs.model_dump()
        elif hasattr(obs, '__dict__'):
            obs = vars(obs)

        if isinstance(obs, dict):
            # Check metadata first (SDK excludes metadata from serialized obs)
            meta = obs.get("metadata", {})
            if key in meta:
                return float(meta[key])
            return float(obs.get(key, default))
        return float(getattr(obs, key, default))
    except Exception:
        return float(default)


def _compute_score(*args, **kwargs) -> float:
    """Compute task score from errors_fixed / total_errors.
    
    Always returns a value strictly in (0.01, 0.99).
    """
    errors_fixed = safe_extract(args, kwargs, 'errors_fixed', 0.0)
    total_errors = safe_extract(args, kwargs, 'total_errors', 1.0)

    if total_errors == 0:
        return 0.99

    raw_score = errors_fixed / total_errors
    return min(max(raw_score, 0.01), 0.99)


def easy_grader(*args, **kwargs) -> float:
    """Grader for the easy task."""
    return _compute_score(*args, **kwargs)


def medium_grader(*args, **kwargs) -> float:
    """Grader for the medium task."""
    return _compute_score(*args, **kwargs)


def hard_grader(*args, **kwargs) -> float:
    """Grader for the hard task."""
    return _compute_score(*args, **kwargs)


# Legacy support — keep grade_task for internal use
def grade_task(task_id: str, env_state: dict) -> dict:
    """Grade a completed task. Used internally by tests."""
    total_errors = env_state.get("total_errors", 0)
    errors_fixed = env_state.get("errors_fixed", 0)

    if total_errors == 0:
        score = 0.99
    else:
        score = errors_fixed / total_errors
        score = min(max(score, 0.01), 0.99)

    return {
        "score": score,
        "passed": score >= 0.5,
        "errors_fixed": errors_fixed,
        "total_errors": total_errors,
    }
