# 📊 DataCleanEnv — Data Cleaning OpenEnv Environment

> Train AI agents to clean dirty tabular data through the OpenEnv step/reset/state API.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🎯 Environment Description

**DataCleanEnv** simulates the real-world task of cleaning dirty tabular data — something data scientists spend **~80% of their time** doing. An AI agent receives a corrupted employee dataset and must identify and fix errors through a series of cleaning actions.

### Why This Matters
- **Real-world utility**: Every company with data needs data cleaning
- **Measurable progress**: Each cell fixed = incremental reward
- **Deterministic grading**: Ground-truth comparison ensures fair evaluation
- **LLM-friendly**: Text-based actions and observations work naturally with language models

## 🏗️ Architecture

```
Agent (LLM) ←→ HTTP API ←→ DataCleanEnvironment
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              DataGenerator    ErrorManifest     Grader
              (seeded data)    (ground truth)   (scoring)
```

## 📋 Action Space

| Action Type | Description | Parameters |
|---|---|---|
| `fix_value` | Replace a cell's value | `row_index`, `column_name`, `new_value` |
| `delete_row` | Remove a duplicate row | `row_index` |
| `fill_missing` | Fill an empty cell | `row_index`, `column_name`, `new_value` |
| `standardize` | Fix formatting | `row_index`, `column_name`, `new_value` |
| `done` | Signal completion | — |

### Action JSON Format
```json
{
    "action_type": "fix_value",
    "row_index": 3,
    "column_name": "department",
    "new_value": "Engineering",
    "reason": "Fixed typo: Enginering → Engineering"
}
```

## 👁️ Observation Space

Each observation returns:

| Field | Type | Description |
|---|---|---|
| `done` | bool | Whether the episode has ended |
| `reward` | float | Reward for the last action |
| `metadata.task_id` | str | Current task: "easy", "medium", "hard" |
| `metadata.current_data` | str | CSV-formatted current dataset |
| `metadata.error_report` | str | Human-readable error summary |
| `metadata.errors_remaining` | int | Errors left to fix |
| `metadata.errors_fixed` | int | Errors correctly fixed |
| `metadata.last_action_result` | str | Feedback on last action |
| `metadata.score` | float | Current score (0.0–1.0) |

## 🎮 Tasks (Easy → Medium → Hard)

### Task 1: Format Fixer (Easy)
- **Errors**: 5 formatting issues in 10 rows
- **Types**: Whitespace, date formats, case errors, typos, phone formats
- **Max steps**: 15
- **Expected baseline**: 0.6–0.8

### Task 2: Data Quality Resolver (Medium)
- **Errors**: 12 mixed issues in 25 rows (+2 duplicate rows)
- **Types**: All easy errors + missing values, duplicates, salary format issues
- **Max steps**: 25
- **Expected baseline**: 0.3–0.5

### Task 3: Cross-Column Auditor (Hard)
- **Errors**: 20 complex issues in 40 rows (+2 duplicate rows)
- **Types**: All medium errors + city-state mismatches, negative salaries, future dates
- **Max steps**: 35
- **Expected baseline**: 0.1–0.3

## 💰 Reward Function

| Event | Reward |
|---|---|
| Correctly fix an error | `+1.0 / total_errors` |
| No-effect action | `-0.05` |
| Introduced an error | `-0.2` |
| "done" with score ≥ 0.8 | `+0.1` bonus |
| Invalid action | `-0.05` |

**Score** = `errors_fixed / total_errors` (clamped to [0.0, 1.0])

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)
- An OpenAI-compatible API key

### Local Development

```bash
# Clone the repo
git clone <repo-url>
cd data-clean-env

# Install dependencies
pip install -r server/requirements.txt

# Run the server
cd server && uvicorn app:app --host 0.0.0.0 --port 8000

# In another terminal, run inference
export OPENAI_API_KEY="your-key-here"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

### Docker

```bash
# Build
docker build -t data-clean-env:latest -f server/Dockerfile .

# Run
docker run -p 8000:8000 data-clean-env:latest

# Test health
curl http://localhost:8000/health
```

### Run Tests

```bash
pytest tests/ -v
```

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key |
| `API_BASE_URL` | ❌ | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | ❌ | `gpt-4o-mini` | Model to use |
| `HF_TOKEN` | ❌ | — | HuggingFace token (alternative to OPENAI_API_KEY) |
| `ENV_BASE_URL` | ❌ | `http://localhost:8000` | Environment server URL |

## 📊 Baseline Scores

| Task | Score | Steps | Model |
|---|---|---|---|
| Easy | ~0.70 | 8 | gpt-4o-mini |
| Medium | ~0.40 | 18 | gpt-4o-mini |
| Hard | ~0.20 | 30 | gpt-4o-mini |
| **Average** | **~0.43** | — | — |

## 📁 Project Structure

```
data-clean-env/
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml             # Dependencies
├── inference.py               # Baseline inference script
├── README.md                  # This file
├── __init__.py                # Package exports
├── server/
│   ├── app.py                 # FastAPI server
│   ├── environment.py         # Core environment logic
│   ├── data_generator.py      # Dirty data generation engine
│   ├── graders.py             # Task graders
│   ├── requirements.txt       # Server dependencies
│   └── Dockerfile             # Container definition
└── tests/
    └── test_environment.py    # Unit tests
```

## 🏆 Hackathon Submission

This environment is built for the **Meta × HuggingFace OpenEnv Hackathon**.

### Pre-submission checklist
- [x] HF Space deploys and responds to `reset()`
- [x] OpenEnv spec compliance (`openenv.yaml`, typed models)
- [x] Dockerfile builds cleanly
- [x] Baseline inference script produces scores
- [x] 3+ tasks with deterministic graders (scores in 0.0–1.0)

## License

MIT
