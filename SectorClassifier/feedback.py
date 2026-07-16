from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

FEEDBACK_DIR = Path("feedback")
PENDING_FILE = FEEDBACK_DIR / "pending_feedback.csv"

COLUMNS = [
    "Organization",
    "Definition",
    "Predicted_Sector",
    "Predicted_Subsector",
    "Correct_Sector",
    "Correct_Subsector",
    "Status",
    "Reason",
    "Timestamp",
]

def _ensure_feedback_file():
    FEEDBACK_DIR.mkdir(exist_ok=True)

    if not PENDING_FILE.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(
            PENDING_FILE,
            index=False,
        )

def _append_feedback(row: dict):
    _ensure_feedback_file()

    df = pd.read_csv(PENDING_FILE)

    df.loc[len(df)] = row

    df.to_csv(
        PENDING_FILE,
        index=False,
    )

def save_reward(
    organization: str,
    definition: str,
    predicted_sector: str,
    predicted_subsector: str,
):
    row = {
        "Organization": organization,
        "Definition": definition,
        "Predicted_Sector": predicted_sector,
        "Predicted_Subsector": predicted_subsector,
        "Correct_Sector": predicted_sector,
        "Correct_Subsector": predicted_subsector,
        "Status": "Reward",
        "Reason": "",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _append_feedback(row)

def save_punishment(
    organization: str,
    definition: str,
    predicted_sector: str,
    predicted_subsector: str,
    correct_sector: str,
    correct_subsector: str,
    reason: str = "",
):

    row = {
        "Organization": organization,
        "Definition": definition,
        "Predicted_Sector": predicted_sector,
        "Predicted_Subsector": predicted_subsector,
        "Correct_Sector": correct_sector,
        "Correct_Subsector": correct_subsector,
        "Status": "Punish",
        "Reason": reason,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _append_feedback(row)

def load_pending_feedback():
    _ensure_feedback_file()
    return pd.read_csv(PENDING_FILE)

def pending_feedback_count() -> int:
    _ensure_feedback_file()
    return len(pd.read_csv(PENDING_FILE))