from datetime import datetime, timedelta
from caldav import DAVClient
import uuid
from dotenv import load_dotenv
import os
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from pathlib import Path


env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)
print("Loading environment variables from:", env_path)
print(".env contents : ", open(env_path).read())
load_dotenv(dotenv_path=env_path)

def get_env(*keys):
    for key in keys:
        value = os.getenv(key)
        if value and value.lower() not in ("none", "null"):
            return value
    return None

# Environment variables
RADICALE_URL = get_env("RADICALE_URL", "CALDAV_URL", "CALENDAR_URL")
USERNAME = get_env("RADICALE_USERNAME", "CALDAV_USERNAME", "USERNAME")
PASSWORD = get_env("RADICALE_PASSWORD", "CALDAV_PASSWORD", "PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER")
smtp_port_str = os.getenv("SMTP_PORT")

TIMEZONE = "Europe/Paris"


if smtp_port_str is None:
    raise ValueError("SMTP_PORT environment variable is not set.")
SMTP_PORT = int(smtp_port_str)
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
FROM_EMAIL = os.getenv("FROM_EMAIL")

print("Loaded SMTP_PORT:", os.getenv("SMTP_PORT"))

def create_appointment(state: dict) -> dict:
    print("DEBUG: create_appointment function called")
    
    # Ensure all required keys exist
    state.setdefault("bot_messages", [])
    state.setdefault("date", None)
    state.setdefault("time", None)
    state.setdefault("email", None)
    
    date = state.get("date")
    time = state.get("time")
    email = state.get("email")

    if not (date and time and email):
        state["bot_messages"].append(
            "Missing some details, so I can't create the appointment. Please provide the date, time, and email."
        )
        state["awaiting_user_response"] = True
        return state

    if not RADICALE_URL:
        state["bot_messages"].append(
            "Radicale is not configured. Please set RADICALE_URL in .env."
        )
        state["awaiting_user_response"] = True
        return state

    print(f"DEBUG: Creating appointment for {date} at {time} for {email}")

    try:
        # Connect to Radicale
        if USERNAME and PASSWORD:
            client = DAVClient(RADICALE_URL, username=USERNAME, password=PASSWORD)
        else:
            client = DAVClient(RADICALE_URL)
        principal = client.principal()

        calendars = principal.calendars()
        if calendars:
            calendar = calendars[0]
        else:
            calendar = principal.make_calendar(name="MyCalendar")

        # Parse start and end times
        start_dt = datetime.fromisoformat(f"{date}T{time}")
        end_dt = start_dt + timedelta(minutes=30)

        # Unique UID
        event_uid = str(uuid.uuid4())

        # Create iCalendar event
        event_template = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//YourApp//AppointmentBot//EN
BEGIN:VEVENT
UID:{event_uid}
SUMMARY:Appointment booked via Chatbot
DESCRIPTION:Appointment booked by user {email}
DTSTART;TZID={TIMEZONE}:{start_dt.strftime('%Y%m%dT%H%M%S')}
DTEND;TZID={TIMEZONE}:{end_dt.strftime('%Y%m%dT%H%M%S')}
END:VEVENT
END:VCALENDAR
"""

        # Add event
        calendar.add_event(event_template)
        #send email
        # Send confirmation email
        subject = "Appointment Confirmation"
        body = (
            f"Hello,\n\n"
            f"Your appointment has been successfully booked.\n\n"
            f"Date: {date}\n"
            f"Time: {time}\n\n"
            f"Thank you!"
        )

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = email

        print("DEBUG: Connecting to SMTP server...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        print("DEBUG: Email sent successfully.")


        state["bot_messages"].append(
            f"Your appointment is confirmed for {date} at {time}. A confirmation email has been sent to {email}."
        )
        state["awaiting_user_response"] = False
        return state

    except Exception as e:
        print(f"Error in create_appointment: {e}")
        state["bot_messages"].append(
            f"Failed to create the appointment: {str(e)}"
        )
        state["awaiting_user_response"] = True
        return state


