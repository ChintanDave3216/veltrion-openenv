"""
Data Generator for DataCleanEnv.

Generates realistic dirty datasets with known ground truth for deterministic grading.
Supports 3 difficulty levels: easy, medium, hard.
Each generated dataset includes an error manifest tracking every injected error.
"""

import random
import copy
from datetime import datetime, timedelta

# ============================================================================
# LOOKUP TABLES - Realistic data for employee records
# ============================================================================

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Charles", "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony",
    "Margaret", "Mark", "Sandra", "Steven", "Ashley", "Andrew", "Dorothy",
    "Paul", "Kimberly", "Joshua", "Emily", "Kenneth", "Donna",
    "Kevin", "Michelle", "Brian", "Carol", "George", "Amanda",
    "Timothy", "Melissa", "Ronald", "Deborah",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera",
    "Campbell", "Mitchell", "Carter", "Roberts",
]

DEPARTMENTS = [
    "Engineering", "Marketing", "Sales", "Human Resources",
    "Finance", "Operations", "Legal", "Customer Support",
]

CITY_STATE_MAP = {
    "New York": "NY",
    "Los Angeles": "CA",
    "Chicago": "IL",
    "Houston": "TX",
    "Phoenix": "AZ",
    "Philadelphia": "PA",
    "San Antonio": "TX",
    "San Diego": "CA",
    "Dallas": "TX",
    "San Jose": "CA",
    "Austin": "TX",
    "Jacksonville": "FL",
    "Fort Worth": "TX",
    "Columbus": "OH",
    "Charlotte": "NC",
    "Indianapolis": "IN",
    "Seattle": "WA",
    "Denver": "CO",
    "Boston": "MA",
    "Nashville": "TN",
    "Portland": "OR",
    "Miami": "FL",
    "Atlanta": "GA",
    "Detroit": "MI",
    "Minneapolis": "MN",
}

DEPARTMENT_TYPOS = {
    "Engineering": ["Enginering", "Engneering", "Engineerng"],
    "Marketing": ["Marketng", "Marketting", "Marekting"],
    "Sales": ["Saels", "Slaes", "Saless"],
    "Human Resources": ["Human Resorces", "Humaan Resources", "Human Resurces"],
    "Finance": ["Finace", "Finanace", "Finnance"],
    "Operations": ["Operatons", "Opeartions", "Operattions"],
    "Legal": ["Leagl", "Legall", "Lgeal"],
    "Customer Support": ["Custmer Support", "Customer Suport", "Cusotmer Support"],
}


# ============================================================================
# CLEAN DATA GENERATION
# ============================================================================

def generate_clean_dataset(num_rows: int, seed: int = 42) -> list[dict]:
    """Generate a clean, correctly formatted employee dataset."""
    rng = random.Random(seed)
    rows = []
    used_emails = set()

    for i in range(num_rows):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)

        # Ensure unique emails
        email_base = f"{first.lower()}.{last.lower()}"
        email = f"{email_base}@company.com"
        counter = 1
        while email in used_emails:
            email = f"{email_base}{counter}@company.com"
            counter += 1
        used_emails.add(email)

        city = rng.choice(list(CITY_STATE_MAP.keys()))
        dept = rng.choice(DEPARTMENTS)
        salary = round(rng.uniform(35000, 150000), 2)
        hire_date = (
            datetime(2020, 1, 1) + timedelta(days=rng.randint(0, 1500))
        ).strftime("%Y-%m-%d")

        area = rng.randint(200, 999)
        prefix = rng.randint(200, 999)
        line = rng.randint(1000, 9999)

        rows.append({
            "name": f"{first} {last}",
            "email": email,
            "phone": f"({area}) {prefix}-{line}",
            "department": dept,
            "salary": str(salary),
            "hire_date": hire_date,
            "city": city,
            "state": CITY_STATE_MAP[city],
        })

    return rows


# ============================================================================
# ERROR INJECTION
# ============================================================================

def _inject_whitespace(row: dict, col: str) -> str:
    """Add leading/trailing whitespace."""
    return f"  {row[col]}  "


def _inject_date_format(row: dict) -> str:
    """Change date from YYYY-MM-DD to MM/DD/YYYY."""
    try:
        dt = datetime.strptime(row["hire_date"], "%Y-%m-%d")
        return dt.strftime("%m/%d/%Y")
    except (ValueError, KeyError):
        return row["hire_date"]


def _inject_case_error(row: dict, col: str) -> str:
    """Change to lowercase."""
    return row[col].lower()


def _inject_typo(row: dict, rng: random.Random) -> str:
    """Inject a typo into the department name."""
    dept = row["department"]
    if dept in DEPARTMENT_TYPOS:
        return rng.choice(DEPARTMENT_TYPOS[dept])
    return dept


def _inject_phone_format(row: dict, rng: random.Random) -> str:
    """Change phone format from (XXX) XXX-XXXX to other formats."""
    phone = row["phone"]
    digits = "".join(c for c in phone if c.isdigit())
    fmt = rng.choice(["dashes", "dots", "plain"])
    if fmt == "dashes":
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    elif fmt == "dots":
        return f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"
    else:
        return digits


def _inject_missing(row: dict, col: str) -> str:
    """Clear a cell to simulate missing data."""
    return ""


def _inject_salary_format(row: dict, rng: random.Random) -> str:
    """Change salary format to include commas or dollar sign."""
    try:
        val = float(row["salary"])
        fmt = rng.choice(["comma", "dollar", "k"])
        if fmt == "comma":
            return f"{val:,.2f}"
        elif fmt == "dollar":
            return f"${val:,.2f}"
        else:
            return f"{val/1000:.1f}K"
    except (ValueError, KeyError):
        return row["salary"]


def _inject_cross_column_error(row: dict, rng: random.Random) -> str:
    """Set state to wrong value (doesn't match city)."""
    correct_state = CITY_STATE_MAP.get(row["city"], row["state"])
    wrong_states = [s for s in set(CITY_STATE_MAP.values()) if s != correct_state]
    return rng.choice(wrong_states) if wrong_states else row["state"]


def _inject_negative_salary(row: dict) -> str:
    """Make salary negative (logical impossibility)."""
    try:
        val = float(row["salary"])
        return str(-abs(val))
    except (ValueError, KeyError):
        return row["salary"]


def _inject_future_date(row: dict) -> str:
    """Set hire date to a future date (logical impossibility)."""
    return "2028-06-15"


def inject_errors(clean_data: list[dict], task_id: str, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Inject errors into the clean dataset based on task difficulty.

    Args:
        clean_data: List of clean row dicts
        task_id: "easy", "medium", or "hard"
        seed: Random seed for reproducibility

    Returns:
        (dirty_data, error_manifest)
        error_manifest: list of dicts with keys:
            row, col, error_type, dirty_value, clean_value
    """
    rng = random.Random(seed + hash(task_id) % 10000)
    dirty_data = copy.deepcopy(clean_data)
    error_manifest = []
    num_rows = len(dirty_data)

    if task_id == "easy":
        # 5 errors: whitespace(1), date_format(1), case(1), typo(1), phone_format(1)
        targets = rng.sample(range(num_rows), min(5, num_rows))

        # Error 1: Whitespace on name
        r = targets[0]
        old_val = dirty_data[r]["name"]
        dirty_data[r]["name"] = _inject_whitespace(dirty_data[r], "name")
        error_manifest.append({
            "row": r, "col": "name", "error_type": "whitespace",
            "dirty_value": dirty_data[r]["name"], "clean_value": old_val
        })

        # Error 2: Date format
        r = targets[1]
        old_val = dirty_data[r]["hire_date"]
        dirty_data[r]["hire_date"] = _inject_date_format(dirty_data[r])
        error_manifest.append({
            "row": r, "col": "hire_date", "error_type": "date_format",
            "dirty_value": dirty_data[r]["hire_date"], "clean_value": old_val
        })

        # Error 3: Case error on department
        r = targets[2]
        old_val = dirty_data[r]["department"]
        dirty_data[r]["department"] = _inject_case_error(dirty_data[r], "department")
        error_manifest.append({
            "row": r, "col": "department", "error_type": "case_error",
            "dirty_value": dirty_data[r]["department"], "clean_value": old_val
        })

        # Error 4: Typo in department
        r = targets[3]
        old_val = dirty_data[r]["department"]
        dirty_data[r]["department"] = _inject_typo(dirty_data[r], rng)
        error_manifest.append({
            "row": r, "col": "department", "error_type": "typo",
            "dirty_value": dirty_data[r]["department"], "clean_value": old_val
        })

        # Error 5: Phone format
        r = targets[4]
        old_val = dirty_data[r]["phone"]
        dirty_data[r]["phone"] = _inject_phone_format(dirty_data[r], rng)
        error_manifest.append({
            "row": r, "col": "phone", "error_type": "phone_format",
            "dirty_value": dirty_data[r]["phone"], "clean_value": old_val
        })

    elif task_id == "medium":
        # 12 errors: easy errors(5) + missing(2) + duplicate(2) + salary_format(2) + case(1)
        targets = rng.sample(range(num_rows), min(12, num_rows))

        # Easy errors (5)
        r = targets[0]
        old_val = dirty_data[r]["name"]
        dirty_data[r]["name"] = _inject_whitespace(dirty_data[r], "name")
        error_manifest.append({"row": r, "col": "name", "error_type": "whitespace",
                               "dirty_value": dirty_data[r]["name"], "clean_value": old_val})

        r = targets[1]
        old_val = dirty_data[r]["hire_date"]
        dirty_data[r]["hire_date"] = _inject_date_format(dirty_data[r])
        error_manifest.append({"row": r, "col": "hire_date", "error_type": "date_format",
                               "dirty_value": dirty_data[r]["hire_date"], "clean_value": old_val})

        r = targets[2]
        old_val = dirty_data[r]["department"]
        dirty_data[r]["department"] = _inject_case_error(dirty_data[r], "department")
        error_manifest.append({"row": r, "col": "department", "error_type": "case_error",
                               "dirty_value": dirty_data[r]["department"], "clean_value": old_val})

        r = targets[3]
        old_val = dirty_data[r]["department"]
        dirty_data[r]["department"] = _inject_typo(dirty_data[r], rng)
        error_manifest.append({"row": r, "col": "department", "error_type": "typo",
                               "dirty_value": dirty_data[r]["department"], "clean_value": old_val})

        r = targets[4]
        old_val = dirty_data[r]["phone"]
        dirty_data[r]["phone"] = _inject_phone_format(dirty_data[r], rng)
        error_manifest.append({"row": r, "col": "phone", "error_type": "phone_format",
                               "dirty_value": dirty_data[r]["phone"], "clean_value": old_val})

        # Missing values (2)
        r = targets[5]
        old_val = dirty_data[r]["email"]
        dirty_data[r]["email"] = _inject_missing(dirty_data[r], "email")
        error_manifest.append({"row": r, "col": "email", "error_type": "missing_value",
                               "dirty_value": "", "clean_value": old_val})

        r = targets[6]
        old_val = dirty_data[r]["city"]
        dirty_data[r]["city"] = _inject_missing(dirty_data[r], "city")
        error_manifest.append({"row": r, "col": "city", "error_type": "missing_value",
                               "dirty_value": "", "clean_value": old_val})

        # Duplicate rows (2) - insert duplicate of existing rows at end
        dup_source_1 = targets[7]
        dup_row_1 = copy.deepcopy(dirty_data[dup_source_1])
        dup_idx_1 = len(dirty_data)
        dirty_data.append(dup_row_1)
        error_manifest.append({"row": dup_idx_1, "col": "__duplicate__", "error_type": "duplicate_row",
                               "dirty_value": f"duplicate_of_{dup_source_1}", "clean_value": "__delete__"})

        dup_source_2 = targets[8]
        dup_row_2 = copy.deepcopy(dirty_data[dup_source_2])
        dup_idx_2 = len(dirty_data)
        dirty_data.append(dup_row_2)
        error_manifest.append({"row": dup_idx_2, "col": "__duplicate__", "error_type": "duplicate_row",
                               "dirty_value": f"duplicate_of_{dup_source_2}", "clean_value": "__delete__"})

        # Salary format errors (2)
        r = targets[9]
        old_val = dirty_data[r]["salary"]
        dirty_data[r]["salary"] = _inject_salary_format(dirty_data[r], rng)
        error_manifest.append({"row": r, "col": "salary", "error_type": "salary_format",
                               "dirty_value": dirty_data[r]["salary"], "clean_value": old_val})

        r = targets[10]
        old_val = dirty_data[r]["salary"]
        dirty_data[r]["salary"] = _inject_salary_format(dirty_data[r], rng)
        error_manifest.append({"row": r, "col": "salary", "error_type": "salary_format",
                               "dirty_value": dirty_data[r]["salary"], "clean_value": old_val})

        # Extra case error (1)
        r = targets[11]
        old_val = dirty_data[r]["city"]
        dirty_data[r]["city"] = _inject_case_error(dirty_data[r], "city")
        error_manifest.append({"row": r, "col": "city", "error_type": "case_error",
                               "dirty_value": dirty_data[r]["city"], "clean_value": old_val})

    elif task_id == "hard":
        # 20 errors: medium errors(12) + cross_column(3) + negative_salary(2) + future_date(2) + whitespace(1)
        targets = rng.sample(range(num_rows), min(20, num_rows))

        # ---- Easy errors (5) ----
        r = targets[0]
        old_val = dirty_data[r]["name"]
        dirty_data[r]["name"] = _inject_whitespace(dirty_data[r], "name")
        error_manifest.append({"row": r, "col": "name", "error_type": "whitespace",
                               "dirty_value": dirty_data[r]["name"], "clean_value": old_val})

        r = targets[1]
        old_val = dirty_data[r]["hire_date"]
        dirty_data[r]["hire_date"] = _inject_date_format(dirty_data[r])
        error_manifest.append({"row": r, "col": "hire_date", "error_type": "date_format",
                               "dirty_value": dirty_data[r]["hire_date"], "clean_value": old_val})

        r = targets[2]
        old_val = dirty_data[r]["department"]
        dirty_data[r]["department"] = _inject_case_error(dirty_data[r], "department")
        error_manifest.append({"row": r, "col": "department", "error_type": "case_error",
                               "dirty_value": dirty_data[r]["department"], "clean_value": old_val})

        r = targets[3]
        old_val = dirty_data[r]["department"]
        dirty_data[r]["department"] = _inject_typo(dirty_data[r], rng)
        error_manifest.append({"row": r, "col": "department", "error_type": "typo",
                               "dirty_value": dirty_data[r]["department"], "clean_value": old_val})

        r = targets[4]
        old_val = dirty_data[r]["phone"]
        dirty_data[r]["phone"] = _inject_phone_format(dirty_data[r], rng)
        error_manifest.append({"row": r, "col": "phone", "error_type": "phone_format",
                               "dirty_value": dirty_data[r]["phone"], "clean_value": old_val})

        # ---- Medium errors (5) ----
        r = targets[5]
        old_val = dirty_data[r]["email"]
        dirty_data[r]["email"] = _inject_missing(dirty_data[r], "email")
        error_manifest.append({"row": r, "col": "email", "error_type": "missing_value",
                               "dirty_value": "", "clean_value": old_val})

        r = targets[6]
        old_val = dirty_data[r]["salary"]
        dirty_data[r]["salary"] = _inject_salary_format(dirty_data[r], rng)
        error_manifest.append({"row": r, "col": "salary", "error_type": "salary_format",
                               "dirty_value": dirty_data[r]["salary"], "clean_value": old_val})

        r = targets[7]
        old_val = dirty_data[r]["salary"]
        dirty_data[r]["salary"] = _inject_salary_format(dirty_data[r], rng)
        error_manifest.append({"row": r, "col": "salary", "error_type": "salary_format",
                               "dirty_value": dirty_data[r]["salary"], "clean_value": old_val})

        # Duplicate (2)
        dup_source_1 = targets[8]
        dup_row_1 = copy.deepcopy(dirty_data[dup_source_1])
        dup_idx_1 = len(dirty_data)
        dirty_data.append(dup_row_1)
        error_manifest.append({"row": dup_idx_1, "col": "__duplicate__", "error_type": "duplicate_row",
                               "dirty_value": f"duplicate_of_{dup_source_1}", "clean_value": "__delete__"})

        dup_source_2 = targets[9]
        dup_row_2 = copy.deepcopy(dirty_data[dup_source_2])
        dup_idx_2 = len(dirty_data)
        dirty_data.append(dup_row_2)
        error_manifest.append({"row": dup_idx_2, "col": "__duplicate__", "error_type": "duplicate_row",
                               "dirty_value": f"duplicate_of_{dup_source_2}", "clean_value": "__delete__"})

        # ---- Hard-only errors (8) ----
        # Cross-column: city-state mismatch (3)
        for i in range(3):
            r = targets[10 + i]
            old_val = dirty_data[r]["state"]
            dirty_data[r]["state"] = _inject_cross_column_error(dirty_data[r], rng)
            error_manifest.append({"row": r, "col": "state", "error_type": "cross_column",
                                   "dirty_value": dirty_data[r]["state"], "clean_value": old_val})

        # Negative salary (2)
        for i in range(2):
            r = targets[13 + i]
            old_val = dirty_data[r]["salary"]
            dirty_data[r]["salary"] = _inject_negative_salary(dirty_data[r])
            error_manifest.append({"row": r, "col": "salary", "error_type": "negative_value",
                                   "dirty_value": dirty_data[r]["salary"], "clean_value": old_val})

        # Future hire date (2)
        for i in range(2):
            r = targets[15 + i]
            old_val = dirty_data[r]["hire_date"]
            dirty_data[r]["hire_date"] = _inject_future_date(dirty_data[r])
            error_manifest.append({"row": r, "col": "hire_date", "error_type": "future_date",
                                   "dirty_value": dirty_data[r]["hire_date"], "clean_value": old_val})

        # Extra whitespace errors (1)
        r = targets[17]
        old_val = dirty_data[r]["email"]
        dirty_data[r]["email"] = _inject_whitespace(dirty_data[r], "email")
        error_manifest.append({"row": r, "col": "email", "error_type": "whitespace",
                               "dirty_value": dirty_data[r]["email"], "clean_value": old_val})

        # Extra case error in city (1)
        r = targets[18]
        old_val = dirty_data[r]["city"]
        dirty_data[r]["city"] = _inject_case_error(dirty_data[r], "city")
        error_manifest.append({"row": r, "col": "city", "error_type": "case_error",
                               "dirty_value": dirty_data[r]["city"], "clean_value": old_val})

        # Extra missing value (1)
        r = targets[19]
        old_val = dirty_data[r]["phone"]
        dirty_data[r]["phone"] = _inject_missing(dirty_data[r], "phone")
        error_manifest.append({"row": r, "col": "phone", "error_type": "missing_value",
                               "dirty_value": "", "clean_value": old_val})

    return dirty_data, error_manifest


# ============================================================================
# DATASET FORMATTING
# ============================================================================

def format_as_csv(data: list[dict]) -> str:
    """Format list of dicts as CSV string for display."""
    if not data:
        return ""
    columns = list(data[0].keys())
    lines = [",".join(columns)]
    for row in data:
        values = []
        for col in columns:
            val = str(row.get(col, ""))
            if "," in val or '"' in val:
                val = f'"{val}"'
            values.append(val)
        lines.append(",".join(values))
    return "\n".join(lines)


def generate_error_report(dirty_data: list[dict], error_manifest: list[dict]) -> str:
    """Generate a human-readable error report for the agent."""
    lines = [f"=== DATA QUALITY REPORT ==="]
    lines.append(f"Total rows: {len(dirty_data)}")
    lines.append(f"Total issues detected: {len(error_manifest)}")
    lines.append("")

    # Group errors by type
    error_types = {}
    for err in error_manifest:
        et = err["error_type"]
        if et not in error_types:
            error_types[et] = []
        error_types[et].append(err)

    for et, errors in error_types.items():
        lines.append(f"  [{et.upper()}] - {len(errors)} issue(s)")
        for err in errors:
            if err["error_type"] == "duplicate_row":
                lines.append(f"    Row {err['row']}: Duplicate row ({err['dirty_value']})")
            elif err["error_type"] == "missing_value":
                lines.append(f"    Row {err['row']}, Column '{err['col']}': Empty/missing value")
            elif err["error_type"] == "cross_column":
                lines.append(f"    Row {err['row']}, Column '{err['col']}': Value doesn't match related column")
            elif err["error_type"] in ("negative_value", "future_date"):
                lines.append(f"    Row {err['row']}, Column '{err['col']}': Logically invalid value '{err['dirty_value']}'")
            else:
                lines.append(f"    Row {err['row']}, Column '{err['col']}': Current='{err['dirty_value']}'")

    return "\n".join(lines)


# ============================================================================
# TASK CONFIGURATIONS
# ============================================================================

TASK_CONFIG = {
    "easy": {
        "description": (
            "Fix formatting errors in a small employee dataset. "
            "Errors include: whitespace issues, date format inconsistencies, "
            "case errors, typos in department names, and phone format variations."
        ),
        "num_rows": 10,
        "max_steps": 15,
    },
    "medium": {
        "description": (
            "Clean a medium employee dataset with data quality issues. "
            "Includes all easy-level errors plus: missing values, duplicate rows, "
            "salary format inconsistencies, and additional case errors."
        ),
        "num_rows": 25,
        "max_steps": 25,
    },
    "hard": {
        "description": (
            "Perform deep cleaning on a large employee dataset with complex errors. "
            "Includes all medium-level errors plus: cross-column inconsistencies "
            "(city-state mismatches), logically impossible values (negative salaries, "
            "future hire dates), and additional formatting issues."
        ),
        "num_rows": 40,
        "max_steps": 35,
    },
}
