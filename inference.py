"""
Baseline Inference Script for DataCleanEnv.

Uses the OpenAI Python client (`from openai import OpenAI`) to run an LLM agent
against the data cleaning environment.
Reads API credentials from environment variables.
Produces reproducible baseline scores on all 3 tasks.

Required environment variables:
    API_BASE_URL  - The API endpoint for the LLM (default: https://api.openai.com/v1)
    MODEL_NAME    - The model identifier (default: gpt-4o-mini)
    HF_TOKEN      - Your HuggingFace / API token (no default - must be set)

Optional:
    LOCAL_IMAGE_NAME - Docker image name when using from_docker_image()

Usage:
    python inference.py

Output format:
    [START] {"task": ..., "env": ..., "model": ...}
    [STEP]  {"step": ..., "action": ..., "reward": ..., "done": ..., "error": ...}
    [END]   {"success": ..., "steps": ..., "score": ..., "rewards": [...]}
"""

import os
import sys
import json
import asyncio
from typing import List, Optional
from openai import OpenAI

# ============================================================================
# CONFIGURATION - Read from environment variables
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "data_clean_env"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_MAP = {"easy": 15, "medium": 25, "hard": 35}
SUCCESS_SCORE_THRESHOLD = 0.5
SEED = 42


# ============================================================================
# STRUCTURED LOGGING - Mandatory format for hackathon evaluation
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
- standardize: Fix formatting (dates->YYYY-MM-DD, phone->(XXX) XXX-XXXX, proper case)
- done: Signal you've finished all fixes

IMPORTANT RULES:
- Dates should be in YYYY-MM-DD format
- Phone numbers should be in (XXX) XXX-XXXX format
- Department/city names should be properly capitalized (Title Case)
- Salary should be a plain number (no $, no commas, no K suffix)
- Remove leading/trailing whitespace from all values
- Check city-state consistency (e.g., Chicago should be in IL, not NY)
- Negative salaries are invalid - use the absolute value
- Future hire dates (beyond 2026) are invalid
- ONLY output the JSON object, nothing else."""


# ============================================================================
# LLM AGENT
# ============================================================================

def get_model_action(client: OpenAI, observation: dict, history: List[dict], step_num: int):
    """Ask the LLM to decide the next cleaning action."""
    meta = observation.get("metadata", observation)  # handle both dict and obs

    user_msg = f"""Step {step_num}/{meta.get('max_steps', 20)}
Errors fixed: {meta.get('errors_fixed', 0)}/{meta.get('total_errors', 0)}
Last action result: {meta.get('last_action_result', 'N/A')}

--- CURRENT DATA ---
{meta.get('current_data', 'No data')}

--- ERROR REPORT ---
{meta.get('error_report', 'No report')}

Respond with a single JSON action:"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
        if "action_type" not in action:
            action["action_type"] = "done"
        return action, content

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "done", "reason": f"Model error: {exc}"}, "done"


# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

async def run_task(llm_client: OpenAI, env, task_id: str) -> float:
    """Run the agent on a single task using the OpenEnv SDK client."""
    max_steps = MAX_STEPS_MAP.get(task_id, 20)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[dict] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment via OpenEnv SDK
        result = await env.reset(task_id=task_id, seed=SEED)
        obs = result.observation if hasattr(result, 'observation') else result
        done = result.done if hasattr(result, 'done') else obs.get("done", False)

        for step in range(1, max_steps + 1):
            if done:
                break

            # Get metadata from observation
            if isinstance(obs, dict):
                meta = obs.get("metadata", obs)
            elif hasattr(obs, 'metadata'):
                meta = obs.metadata if isinstance(obs.metadata, dict) else {}
            else:
                meta = {}

            obs_for_llm = {"metadata": meta}

            # Get LLM action
            action, raw_response = get_model_action(llm_client, obs_for_llm, history, step)

            # Step the environment via OpenEnv SDK
            result = await env.step(action)
            obs = result.observation if hasattr(result, 'observation') else result
            reward = result.reward if hasattr(result, 'reward') else (obs.get("reward", 0.0) if isinstance(obs, dict) else 0.0)
            done = result.done if hasattr(result, 'done') else (obs.get("done", False) if isinstance(obs, dict) else False)

            rewards.append(float(reward) if reward is not None else 0.0)
            steps_taken = step

            action_str = json.dumps(action) if isinstance(action, dict) else str(action)
            log_step(step=step, action=action_str, reward=float(reward) if reward else 0.0, done=done, error=None)

            history.append({"role": "user", "content": f"Step {step} observation"})
            history.append({"role": "assistant", "content": raw_response})

            if done:
                break

        # Calculate score from metadata
        if isinstance(obs, dict):
            meta = obs.get("metadata", obs)
        elif hasattr(obs, 'metadata'):
            meta = obs.metadata if isinstance(obs.metadata, dict) else {}
        else:
            meta = {}

        errors_fixed = meta.get("errors_fixed", 0)
        total_errors = meta.get("total_errors", 1)
        score = errors_fixed / max(total_errors, 1)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def async_main() -> None:
    """Async main - uses OpenEnv SDK to connect to the environment."""
    from openenv import GenericEnvClient

    if not HF_TOKEN:
        print("[ERROR] No API key found. Set HF_TOKEN environment variable.", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    print(f"[INFO] DataCleanEnv Baseline Inference", flush=True)
    print(f"[INFO] API Base: {API_BASE_URL}", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", flush=True)
    print(f"[INFO] Tasks: {TASKS}", flush=True)

    # Initialize OpenAI client (using HF_TOKEN as api_key)
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Connect to environment using OpenEnv SDK
    env = None
    try:
        if LOCAL_IMAGE_NAME:
            print(f"[INFO] Starting container from image: {LOCAL_IMAGE_NAME}", flush=True)
            env = await GenericEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            # Fallback: connect directly to a running server
            env_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
            print(f"[INFO] Connecting to environment at: {env_url}", flush=True)
            env = GenericEnvClient(base_url=env_url)
            await env.connect()

        print(f"[INFO] Environment connected.", flush=True)

        # Run all tasks
        all_scores = {}
        for task_id in TASKS:
            print(f"\n{'='*60}", flush=True)
            print(f"[INFO] Running task: {task_id}", flush=True)
            print(f"{'='*60}", flush=True)

            score = await run_task(llm_client, env, task_id)
            all_scores[task_id] = score
            print(f"[INFO] Task '{task_id}' score: {score:.4f}", flush=True)

        # Summary
        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        print(f"\n{'='*60}", flush=True)
        print(f"  BASELINE RESULTS - DataCleanEnv", flush=True)
        print(f"{'='*60}", flush=True)
        for task_id, score in all_scores.items():
            status = "PASS" if score >= SUCCESS_SCORE_THRESHOLD else "FAIL"
            print(f"  {task_id:8s}: {score:.4f}  {status}", flush=True)
        print(f"  {'Average':8s}: {avg_score:.4f}", flush=True)
        print(f"{'='*60}", flush=True)

    except Exception as exc:
        print(f"[ERROR] Fatal error: {exc}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


def main() -> None:
    """Main entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
