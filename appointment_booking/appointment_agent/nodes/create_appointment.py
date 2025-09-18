from datetime import datetime, timedelta
from caldav import DAVClient
from dotenv import load_dotenv
from email.mime.text import MIMEText
import logging
import os
from pathlib import Path
import smtplib
import uuid

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
SMTP_SERVER = os.getenv("SMTP_SERVER")
smtp_port_str = os.getenv("SMTP_PORT")

TIMEZONE = "Europe/Paris"

try:
    SMTP_PORT = int(smtp_port_str) if smtp_port_str else None
except ValueError:
    logger.warning("Invalid SMTP_PORT value; confirmation emails are disabled")
    SMTP_PORT = None
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
FROM_EMAIL = os.getenv("FROM_EMAIL")


def smtp_configured() -> bool:
    return all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, FROM_EMAIL])


def create_appointment(state: dict) -> dict:
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

    try:
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

        start_dt = datetime.fromisoformat(f"{date}T{time}")
        end_dt = start_dt + timedelta(minutes=30)
        event_uid = str(uuid.uuid4())

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

        calendar.add_event(event_template)
        subject = "Appointment Confirmation"
        body = (
            f"Hello,\n\n"
            f"Your appointment has been successfully booked.\n\n"
            f"Date: {date}\n"
            f"Time: {time}\n\n"
            f"Thank you!"
        )

        email_sent = False
        if smtp_configured():
            try:
                msg = MIMEText(body)
                msg["Subject"] = subject
                msg["From"] = FROM_EMAIL
                msg["To"] = email

                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                    server.starttls()
                    server.login(SMTP_USERNAME, SMTP_PASSWORD)
                    server.send_message(msg)
                email_sent = True
            except Exception:
                logger.exception("Appointment created, but confirmation email failed")

        if email_sent:
            state["bot_messages"].append(
                f"Your appointment is confirmed for {date} at {time}. A confirmation email has been sent to {email}."
            )
        else:
            state["bot_messages"].append(
                f"Your appointment is confirmed for {date} at {time}. I could not send a confirmation email."
            )
        state["awaiting_user_response"] = False
        return state

    except Exception as e:
        logger.exception("Error creating appointment")
        state["bot_messages"].append(
            f"Failed to create the appointment: {str(e)}"
        )
        state["awaiting_user_response"] = True
        return state


