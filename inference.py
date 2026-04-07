"""
Baseline Inference Script for DataCleanEnv.

Uses the OpenAI API client to run an LLM agent against the data cleaning environment.
Reads API credentials from environment variables.
Produces reproducible baseline scores on all 3 tasks.

Required environment variables:
    API_BASE_URL  - The API endpoint for the LLM (default: https://api.openai.com/v1)
    MODEL_NAME    - The model identifier (default: gpt-4o-mini)
    HF_TOKEN      - Your API key (also checks OPENAI_API_KEY)

Usage:
    # Set environment variables first, then:
    python inference.py

Output format:
    [START] {"task": ..., "env": ..., "model": ...}
    [STEP]  {"step": ..., "action": ..., "reward": ..., "done": ..., "error": ...}
    [END]   {"success": ..., "steps": ..., "score": ..., "rewards": [...]}
"""

import os
import sys
import json
import time
import requests
from typing import List, Optional
from openai import OpenAI


# ============================================================================
# CONFIGURATION — Read from environment variables
# ============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")

# Environment configuration
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK = "data_clean_env"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_MAP = {"easy": 15, "medium": 25, "hard": 35}
SUCCESS_SCORE_THRESHOLD = 0.5
SEED = 42


# ============================================================================
# STRUCTURED LOGGING — Mandatory format for hackathon evaluation
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] {json.dumps({"task": task, "env": env, "model": model})}', flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f'[STEP] {json.dumps({"step": step, "action": action, "reward": reward, "done": done, "error": error})}', flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f'[END] {json.dumps({"success": success, "steps": steps, "score": score, "rewards": rewards})}', flush=True)


# ============================================================================
# SYSTEM PROMPT FOR THE LLM AGENT
# ============================================================================

SYSTEM_PROMPT = """You are a data cleaning agent. You receive a dirty CSV dataset and must fix errors.

You will receive:
1. The current state of the dataset (CSV format)
2. An error report listing detected issues
3. Feedback on your last action

For EACH step, respond with a SINGLE JSON object (no markdown, no explanation):

{
    "action_type": "<one of: fix_value, delete_row, fill_missing, standardize, done>",
    "row_index": <integer: the row to modify, use the _row_index column value>,
    "column_name": "<string: the column to modify>",
    "new_value": "<string: the corrected value>",
    "reason": "<string: brief explanation>"
}

Action types:
- fix_value: Replace a cell's value (for typos, wrong formats, wrong data)
- delete_row: Remove a duplicate row (only use for rows identified as duplicates)
- fill_missing: Fill an empty cell with the correct value
- standardize: Fix formatting (dates→YYYY-MM-DD, phone→(XXX) XXX-XXXX, proper case)
- done: Signal you've finished all fixes

IMPORTANT RULES:
- Dates should be in YYYY-MM-DD format
- Phone numbers should be in (XXX) XXX-XXXX format
- Department/city names should be properly capitalized (Title Case)
- Salary should be a plain number (no $, no commas, no K suffix)
- Remove leading/trailing whitespace from all values
- Check city-state consistency (e.g., Chicago should be in IL, not NY)
- Negative salaries are invalid — use the absolute value
- Future hire dates (beyond 2026) are invalid
- ONLY output the JSON object, nothing else."""


# ============================================================================
# LLM AGENT
# ============================================================================

def get_model_action(
    client: OpenAI,
    observation: dict,
    history: List[dict],
    step_num: int,
) -> dict:
    """
    Ask the LLM to decide the next cleaning action based on the current observation.

    Returns:
        Parsed action dict, or a default "done" action on failure.
    """
    meta = observation.get("metadata", {})

    # Build the user message with current state
    user_msg = f"""Step {step_num}/{meta.get('max_steps', 20)}
Errors fixed: {meta.get('errors_fixed', 0)}/{meta.get('total_errors', 0)}
Last action result: {meta.get('last_action_result', 'N/A')}

--- CURRENT DATA ---
{meta.get('current_data', 'No data')}

--- ERROR REPORT ---
{meta.get('error_report', 'No report')}

Respond with a single JSON action:"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add recent history (last 4 exchanges to stay within context)
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        action = json.loads(content)

        # Validate required fields
        if "action_type" not in action:
            action["action_type"] = "done"

        return action, content

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "done", "reason": f"Model error: {exc}"}, "done"


# ============================================================================
# ENVIRONMENT CLIENT (HTTP)
# ============================================================================

class EnvClient:
    """Simple HTTP client for the DataClean environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy", seed: int = 42) -> dict:
        """Reset the environment."""
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        """Take a step in the environment."""
        resp = requests.post(
            f"{self.base_url}/step",
            json=action,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        """Get current state."""
        resp = requests.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> dict:
        """Grade current episode."""
        resp = requests.get(f"{self.base_url}/grade", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

def run_task(client: OpenAI, env: EnvClient, task_id: str) -> float:
    """
    Run the agent on a single task and return the score.
    """
    max_steps = MAX_STEPS_MAP.get(task_id, 20)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[dict] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        obs = env.reset(task_id=task_id, seed=SEED)

        for step in range(1, max_steps + 1):
            if obs.get("done", False):
                break

            # Get LLM action
            action, raw_response = get_model_action(client, obs, history, step)

            # Step the environment
            obs = env.step(action)
            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)

            rewards.append(reward)
            steps_taken = step

            # Log step
            action_str = json.dumps(action) if isinstance(action, dict) else str(action)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            # Update history for context
            history.append({"role": "user", "content": f"Step {step} observation"})
            history.append({"role": "assistant", "content": raw_response})

            if done:
                break

        # Get final grade
        grade_result = env.grade()
        score = grade_result.get("score", 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    """Main entry point — runs all 3 tasks and reports scores."""

    if not API_KEY:
        print("[ERROR] No API key found. Set OPENAI_API_KEY or HF_TOKEN environment variable.", flush=True)
        print("[INFO] Example: export OPENAI_API_KEY='sk-...'", flush=True)
        sys.exit(1)

    print(f"[INFO] DataCleanEnv Baseline Inference", flush=True)
    print(f"[INFO] API Base: {API_BASE_URL}", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", flush=True)
    print(f"[INFO] Environment: {ENV_BASE_URL}", flush=True)
    print(f"[INFO] Tasks: {TASKS}", flush=True)
    print("", flush=True)

    # Initialize OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize environment client
    env = EnvClient(base_url=ENV_BASE_URL)

    # Wait for environment to be ready
    for attempt in range(10):
        try:
            resp = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                print(f"[INFO] Environment ready.", flush=True)
                break
        except requests.ConnectionError:
            pass
        print(f"[INFO] Waiting for environment... (attempt {attempt + 1}/10)", flush=True)
        time.sleep(3)
    else:
        print("[ERROR] Environment not reachable. Make sure the server is running.", flush=True)
        print(f"[INFO] Start with: cd server && uvicorn app:app --host 0.0.0.0 --port 8000", flush=True)
        sys.exit(1)

    # Run all tasks
    all_scores = {}
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"[INFO] Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)

        score = run_task(client, env, task_id)
        all_scores[task_id] = score

        print(f"[INFO] Task '{task_id}' score: {score:.4f}", flush=True)

    # Summary
    avg_score = sum(all_scores.values()) / len(all_scores)

    print(f"\n{'='*60}", flush=True)
    print(f"  BASELINE RESULTS — DataCleanEnv", flush=True)
    print(f"{'='*60}", flush=True)
    for task_id, score in all_scores.items():
        status = "✅ PASS" if score >= SUCCESS_SCORE_THRESHOLD else "❌ FAIL"
        print(f"  {task_id:8s}: {score:.4f}  {status}", flush=True)
    print(f"  {'Average':8s}: {avg_score:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
