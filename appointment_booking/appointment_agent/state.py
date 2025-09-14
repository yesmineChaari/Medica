from typing import Optional, List, Dict, Any

class AppointmentState(dict):
    """
    Appointment state with mandatory initialization of all required keys.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        defaults = {
            "user_messages": [],
            "bot_messages": [],
            "greeting_sent": False,
            "last_user_intent": "",
            "date": None,
            "time": None,
            "email": None,
            "confirmed": None,
            "awaiting_user_response": False,
        }
        for key, value in defaults.items():
            if key not in self or self[key] is None:
                self[key] = value

    def __setitem__(self, key, value):
        # Ensure lists stay as lists when updated
        if key in ["user_messages", "bot_messages"] and not isinstance(value, list):
            value = [value] if value else []
        super().__setitem__(key, value)

    def add_user_message(self, message: str):
        if "user_messages" not in self:
            self["user_messages"] = []
        self["user_messages"].append(message)

    def add_bot_message(self, message: str):
        if "bot_messages" not in self:
            self["bot_messages"] = []
        self["bot_messages"].append(message)

    def get_last_user_message(self) -> str:
        return self["user_messages"][-1] if self.get("user_messages") else ""

    def get_last_bot_message(self) -> str:
        return self["bot_messages"][-1] if self.get("bot_messages") else ""