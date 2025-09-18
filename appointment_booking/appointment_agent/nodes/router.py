def route(state: dict) -> str:
    if state.get("date") and state.get("time") and state.get("email"):
        return "confirm_booking"
    if state.get("last_user_intent") == "APPOINTMENT_DETAILS":
        return "collect_details"
    return "greet_user"
