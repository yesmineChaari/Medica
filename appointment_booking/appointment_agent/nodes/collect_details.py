from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from validate_email import validate_email as validate_email_lib   # [validate_email]
import json
import re
from datetime import datetime, timedelta

llm = OllamaLLM(model="llama3")
TODAY = datetime.now().strftime("%Y-%m-%d")

# Prompt to extract appointment details in JSON
extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You extract appointment booking details from the user's message and return them as a JSON object. "
            "Today is {today}. "
            "Return a JSON object with keys: date (YYYY-MM-DD), time (24h HH:MM), email. "
            "If any value is missing, use null. "
            "For dates, convert to YYYY-MM-DD format. "
            "For times, convert to 24-hour HH:MM format. "
            "For emails, extract any email address found in the message."
            "If the date format is ambiguous (e.g., it's unclear which number is the day and which is the month), "
            "then return null for the date instead of guessing. "
            "Only extract the date if you are certain about the interpretation."
        )
    ),
    (
        "human",
        "Message: {input}\n\n"
        "Respond only with the JSON object.\n\n"
        "Example: {{\"date\": \"2025-07-05\", \"time\": \"15:00\", \"email\": \"example@example.com\"}}"
    )
])

# ---------- Validators ----------

def validate_date(date_str):
    """Returns 'valid', 'past', 'too_far', or 'invalid'."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        today = datetime.today()

        if date_obj.date() < today.date():
            return "past"
        if date_obj > today + timedelta(days=365):
            return "too_far"
        return "valid"
    except Exception:
        return "invalid"

def is_valid_time(time_str):
    try:
        hour, minute = map(int, time_str.split(":"))
        return 0 <= hour <= 23 and 0 <= minute <= 59
    except Exception:
        return False

def is_valid_email(email_str):
    if not email_str:
        return False
    # Quick regex check first
    regex = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    if not re.match(regex, email_str):
        return False
    # [validate_email] Deep validation (DNS+SMTP)
    try:
        result = validate_email_lib(
            email_address=email_str,
            check_format=True,
            check_dns=True,
            check_smtp=True,
            smtp_timeout=10,
            dns_timeout=10,
            smtp_helo_host="localhost",
            smtp_from_address="noreply@yourdomain.com"
        )
        return result
    except Exception as e:
        print(f"[validate_email] Error validating email '{email_str}': {e}")
        return False

# ---------- Main Function ----------

def collect_details(state: dict) -> dict:
    print("DEBUG: collect_details function called")
    
    past = False
    # Ensure all required keys exist
    state.setdefault("bot_messages", [])
    state.setdefault("user_messages", [])
    state.setdefault("date", None)
    state.setdefault("time", None)
    state.setdefault("email", None)
    
    user_message = state.get("user_messages", [])[-1] if state.get("user_messages") else ""
    
    # If no user message, wait for input
    if not user_message:
        print("DEBUG: No user message in collect_details")
        # Ask for what we need
        missing = []
        if not state["date"]:
            missing.append("date")
        if not state["time"]:
            missing.append("time")
        if not state["email"]:
            missing.append("email")
        state["bot_messages"].append(
            f"Please provide the appointment {', '.join(missing)}."
        )
        state["awaiting_user_response"] = True
        return state

    print(f"DEBUG: Processing user message in collect_details: '{user_message}'")
    print(f"DEBUG: Current state - date: {state['date']}, time: {state['time']}, email: {state['email']}")

    # If we already have all details, move to confirm_booking
    if state.get("date") and state.get("time") and state.get("email"):
        print("DEBUG: All details already present, moving to confirm_booking")
        state["awaiting_user_response"] = False
        return state

    # Check if this message is just an email (common case)
    if "@" in user_message and "." in user_message and " " not in user_message.strip():
        # This looks like a standalone email
        print("DEBUG: Detected standalone email")
        email_candidate = user_message.strip()
        # Check if we now have all details
        if is_valid_email(email_candidate):
            state["email"] = email_candidate
            if state.get("date") and state.get("time") and state.get("email"):
                print("DEBUG: All details complete with email, moving to confirm_booking")
            state["awaiting_user_response"] = False
        else:
            # Still need more details
            missing = []
            if not state["date"]:
                missing.append("date")
            if not state["time"]:
                missing.append("time")
            state["bot_messages"].append(
                f"Great! I have your email. I still need the appointment {', '.join(missing)}. Could you please provide that?"
            )
            state["awaiting_user_response"] = True
        return state

    prompt = extraction_prompt.format_messages(
        input=user_message,
        today=TODAY
    )

    try:
        raw_response = llm.invoke(prompt)
        
        # Handle potential None response
        if raw_response is None:
            state["bot_messages"].append("Sorry, I couldn't process your message. Please try again.")
            state["awaiting_user_response"] = True
            return state

        print(f"DEBUG: LLM response: {raw_response}")
        details = json.loads(raw_response)
        date = details.get("date")
        time = details.get("time")
        email = details.get("email")
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        print(f"Error parsing response: {e}")
        state["bot_messages"].append(
            "Sorry, I couldn't understand that. Could you please rephrase the date, time, and email?"
        )
        state["awaiting_user_response"] = True
        return state
    except Exception as e:
        print(f"Error in collect_details: {e}")
        state["bot_messages"].append("Sorry, I encountered an error. Please try again.")
        state["awaiting_user_response"] = True
        return state

    # Update state only if not already filled
    if not state["date"] and date:
        state["date"] = date
    if not state["time"] and time:
        state["time"] = time
    if not state["email"] and email:
        state["email"] = email

    print(f"DEBUG: After extraction - date: {state['date']}, time: {state['time']}, email: {state['email']}")

    # Validate fields
    invalid = []
    too_far = False

    if state["date"]:
        date_status = validate_date(state["date"])
        if date_status == "invalid":
            invalid.append("date")
            state["date"] = None
        elif date_status == "too_far":
            too_far = True
            state["date"] = None
        elif date_status == "past":
            past = True
            state["date"] = None

    if state["time"] and not is_valid_time(state["time"]):
        invalid.append("time")
        state["time"] = None

    if state["email"] and not is_valid_email(state["email"]):
        invalid.append("email")
        state["email"] = None

    if too_far:
        state["bot_messages"].append(
            "That date is too far in the future. Could you please choose a date within the next 12 months?"
        )
        state["awaiting_user_response"] = True
        return state
    if past:
        state["bot_messages"].append(
            "That date is in the past. Could you please choose a future date?"
        )
        state["awaiting_user_response"] = True
        return state

    missing = []
    if not state["date"]:
        missing.append("date")
    if not state["time"]:
        missing.append("time")
    if not state["email"]:
        missing.append("email")

    if missing:
        if invalid:
            state["bot_messages"].append(
                f"I noticed some details seem invalid: {', '.join(invalid)}. "
                f"Could you please re-enter the {', '.join(missing)}?"
            )
        else:
            if len(missing) == 1:
                state["bot_messages"].append(
                    f"I still need the appointment {missing[0]}. Could you please provide that?"
                )
            else:
                state["bot_messages"].append(
                    f"I still need the appointment {', '.join(missing)}. Could you please provide that?"
                )
        state["awaiting_user_response"] = True
        return state

    # All details valid
    print("DEBUG: All details valid, moving to confirm_booking")
    state["awaiting_user_response"] = False
    return state
