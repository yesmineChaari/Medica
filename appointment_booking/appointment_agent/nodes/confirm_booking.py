from datetime import datetime, timedelta
from caldav import DAVClient
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

def get_env(*keys):
    for key in keys:
        value = os.getenv(key)
        if value and value.lower() not in ("none", "null"):
            return value
    return None

RADICALE_URL = get_env("RADICALE_URL", "CALDAV_URL", "CALENDAR_URL")
USERNAME = get_env("RADICALE_USERNAME", "CALDAV_USERNAME", "USERNAME")
PASSWORD = get_env("RADICALE_PASSWORD", "CALDAV_PASSWORD", "PASSWORD")

def build_client():
    if not RADICALE_URL:
        return None
    if USERNAME and PASSWORD:
        return DAVClient(RADICALE_URL, username=USERNAME, password=PASSWORD)
    return DAVClient(RADICALE_URL)

def confirm_booking(state: dict) -> dict:
    state.setdefault("bot_messages", [])
    state.setdefault("date", None)
    state.setdefault("time", None)
    
    date = state.get("date")
    time = state.get("time")

    if not date or not time:
        state["bot_messages"].append(
            "I'm missing the date or time. Please provide the complete details."
        )
        state["awaiting_user_response"] = True
        return state

    try:
        client = build_client()
        if client is None:
            state["bot_messages"].append(
                "Radicale is not configured. Please set RADICALE_URL in .env."
            )
            state["awaiting_user_response"] = True
            return state
        principal = client.principal()

        calendars = principal.calendars()
        if not calendars:
            state["bot_messages"].append(
                "No calendars found on the server. Cannot check availability."
            )
            state["awaiting_user_response"] = True
            return state
        calendar = calendars[0]

        appointment_start = datetime.fromisoformat(f"{date}T{time}")
        appointment_end = appointment_start + timedelta(minutes=30)

        day_start = appointment_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        events = calendar.date_search(day_start, day_end)

        slot_taken = False
        for event in events:
            raw = event.data

            dtstart_line = next((line for line in raw.splitlines() if line.startswith("DTSTART")), None)
            dtend_line = next((line for line in raw.splitlines() if line.startswith("DTEND")), None)
            if not dtstart_line or not dtend_line:
                continue

            dtstart_val = dtstart_line.split(":")[-1]
            dtend_val = dtend_line.split(":")[-1]

            ev_start = datetime.strptime(dtstart_val, "%Y%m%dT%H%M%S")
            ev_end = datetime.strptime(dtend_val, "%Y%m%dT%H%M%S")

            latest_start = max(ev_start, appointment_start)
            earliest_end = min(ev_end, appointment_end)
            overlap = (earliest_end - latest_start).total_seconds()

            if overlap > 0:
                slot_taken = True
                break

        if slot_taken:
            state["bot_messages"].append(
                "Unfortunately, that time slot is not available. Could you please choose a different time?"
            )
            state["time"] = None
            state["awaiting_user_response"] = True
            return state

        state["confirmed"] = True
        state["awaiting_user_response"] = False
        state["bot_messages"].append(
            "Great news! The time slot is available. Proceeding to book your appointment."
        )
        return state

    except Exception as e:
        logger.exception("Error in confirm_booking")
        error_text = str(e)
        if "Unauthorized" in error_text:
            state["bot_messages"].append(
                "Authorization failed. Please check RADICALE_USERNAME/RADICALE_PASSWORD "
                "(or start Radicale with --auth-type none)."
            )
        else:
            state["bot_messages"].append(
                f"An error occurred while checking availability: {error_text}"
            )
        state["awaiting_user_response"] = True
        return state
