from datetime import datetime, timedelta
from caldav import DAVClient
import os
from dotenv import load_dotenv
from pathlib import Path

#  load variables from .env
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)
print("Loading environment variables from:", env_path)
print(".env contents : ", open(env_path).read())
load_dotenv(dotenv_path=env_path)

# Environment variables
RADICALE_URL = os.getenv("RADICALE_URL")
USERNAME = os.getenv("RADICALE_USERNAME")
PASSWORD = os.getenv("RADICALE_PASSWORD")

TIMEZONE = "Europe/Paris"  # Adjust your timezone

def confirm_booking(state: dict) -> dict:
    print("DEBUG: confirm_booking function called")
    
    # Ensure all required keys exist
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

    print(f"DEBUG: Checking availability for {date} at {time}")

    try:
        # Connect to Radicale server
        client = DAVClient(RADICALE_URL, username=USERNAME, password=PASSWORD)
        principal = client.principal()

        calendars = principal.calendars()
        if not calendars:
            state["bot_messages"].append(
                "No calendars found on the server. Cannot check availability."
            )
            state["awaiting_user_response"] = True
            return state
        calendar = calendars[0]

        # Parse appointment start and end times
        appointment_start = datetime.fromisoformat(f"{date}T{time}")
        appointment_end = appointment_start + timedelta(minutes=30)

        # Format as strings to match event DTSTART and DTEND format (without separators)
        start_str = appointment_start.strftime("%Y%m%dT%H%M%S")
        end_str = appointment_end.strftime("%Y%m%dT%H%M%S")

        # Fetch all events for the day
        day_start = appointment_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        events = calendar.date_search(day_start, day_end)

        # Check if any event overlaps
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

        # Slot is free
        state["confirmed"] = True
        state["awaiting_user_response"] = False
        state["bot_messages"].append(
            "Great news! The time slot is available. Proceeding to book your appointment."
        )
        return state

    except Exception as e:
        print(f"Error in confirm_booking: {e}")
        state["bot_messages"].append(
            f"An error occurred while checking availability: {str(e)}"
        )
        state["awaiting_user_response"] = True
        return state
