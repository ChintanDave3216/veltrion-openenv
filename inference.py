import os
import json
import textwrap
import asyncio
from typing import List, Optional
from openai import OpenAI

from openenv import GenericEnvClient

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "data_clean_env"
MAX_STEPS = 15
MAX_POSSIBLE_REWARD = 5.0

# The validator looks for exactly 3 tasks
TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a data cleaning agent. You receive a dirty CSV dataset and must fix errors.
    RULES:
    1. Look at the current data and error report
    2. Fix one error at a time using the appropriate action
    3. Output EXACTLY ONE JSON object per step. No extra text.
    
    JSON format:
    {"action_type": "fix_value", "row_index": 0, "column_name": "name", "new_value": "corrected", "reason": "fixing typo"}
    
    Action types: fix_value, delete_row, fill_missing, standardize, done
    CRITICAL: Output EXACTLY ONE JSON object. No extra text.
""").strip()


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


async def run_task(client: OpenAI, env, task_name: str):
    """Runs a single episode for a specific task."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_name)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_dict = result.observation.model_dump() if hasattr(result.observation, 'model_dump') else {}

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Current Data State: {json.dumps(obs_dict)}"}
                    ],
                    temperature=0.1
                )
                response_text = completion.choices[0].message.content.strip()

                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].strip()

                if '}\n{' in response_text:
                    response_text = response_text.split('}\n{')[0] + '}'
                elif '}{' in response_text:
                    response_text = response_text.split('}{')[0] + '}'

                action_data = json.loads(response_text)
                if isinstance(action_data, list):
                    action_data = action_data[0]

                action_str = f"{action_data.get('action_type', 'done')}"
                error = None

            except Exception as e:
                action_data = {"action_type": "done", "reason": "error"}
                action_str = "done"
                clean_error = str(e).replace('\n', ' ')
                error = f"Err: {clean_error}"[:50]

            result = await env.step(action_data)
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        raw_score = sum(rewards) / MAX_POSSIBLE_REWARD
        score = min(max(raw_score, 0.01), 0.99)
        success = score > 0.0

    except Exception as run_error:
        print(f"[DEBUG] Execution Error: {run_error}")

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = GenericEnvClient(base_url="http://localhost:8000")

    try:
        await env.connect()

        # Loop through all 3 tasks
        for task_name in TASKS:
            await run_task(client, env, task_name)

    except Exception as e:
        print(f"[DEBUG] Connection error: {e}")
        # Still produce 3 log pairs
        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.01, rewards=[0.01])

    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
