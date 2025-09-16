# state.py
from typing import TypedDict, Annotated, Optional
from operator import add

class AppointmentState(TypedDict):
    user_messages: Annotated[list[str], add]
    bot_messages:  Annotated[list[str], add]
    last_user_intent: str
    date:      Optional[str]
    time:      Optional[str]
    email:     Optional[str]
    confirmed: Optional[bool]
    greeting_sent: bool