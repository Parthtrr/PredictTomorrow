"""Read the project's shared configuration."""

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def load_config():
    """Load config.json and resolve 'today' whenever the program starts."""
    config_path = Path(__file__).parent / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    # Use the configured timezone instead of the computer's timezone.
    data = config["data"]
    if data["end_date"] == "today":
        data["end_date"] = datetime.now(ZoneInfo(data["timezone"])).date().isoformat()

    if date.fromisoformat(data["start_date"]) > date.fromisoformat(data["end_date"]):
        raise ValueError("start_date must be on or before end_date.")

    return config
