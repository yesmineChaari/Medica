from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)

llm = OllamaLLM(model="llama3")

intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an intent classification assistant for an appointment booking chatbot.\n\n"
            "Classify the user's message as exactly one of the following intents:\n"
            "1. GREETING - The user is greeting you.\n"
            "2. APPOINTMENT_REQUEST - The user wants to book an appointment but gives NO specific date/time/email.\n"
            "3. APPOINTMENT_DETAILS - The user provides specific date, time, or email info.\n"
            "4. OTHER - Everything else.\n\n"
            "IMPORTANT RULE:\n"
            "If the message contains ANY date, time, or email - even if it also includes a request to book - classify it as APPOINTMENT_DETAILS.\n"
            "Only classify as APPOINTMENT_REQUEST if NO date,time or email is present.\n"
            "Respond ONLY with the intent label.\n"
        )
    ),
    ("human", "Message: hi\nAnswer: GREETING"),
    ("human", "Message: I'd like to book something\nAnswer: APPOINTMENT_REQUEST"),
    ("human", "Message: I want to book an appointment\nAnswer: APPOINTMENT_REQUEST"),
    ("human", "Message: I want to book an appointment on August 15 at 9 am\nAnswer: APPOINTMENT_DETAILS"),
    ("human", "Message: 15 August at 9 am\nAnswer: APPOINTMENT_DETAILS"),
    ("human", "Message: Please book me for 2025-07-05 10:00\nAnswer: APPOINTMENT_DETAILS"),
    ("human", "Message: Email: user@example.com\nAnswer: APPOINTMENT_DETAILS"),
    ("human", "Message: What's the weather today?\nAnswer: OTHER"),
    ("human", "Message: {input}\nAnswer:")
])

response_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful appointment booking assistant.\n"
            "You help users book appointments in a doctor's office.\n"
            "Based on the user's intent and message, generate a short, friendly reply.\n"
            "Keep it under 2 sentences. Be helpful and on-topic."
        )
    ),
    ("human", "Intent: GREETING\nMessage: hi\nReply: Hi there! How can I assist you with booking an appointment today?"),
    ("human", "Intent: APPOINTMENT_REQUEST\nMessage: I'd like to book something\nReply: Sure! Could you please tell me the preferred date and time?"),
    ("human", "Intent: APPOINTMENT_REQUEST\nMessage:  I want to book an appointment. Is that possible? \nReply: Absolutely! Please let me know the date and time you prefer, also please enter your email."),
    ("human", "Intent: OTHER\nMessage: What's your favorite movie?\nReply: I'm here to help with booking appointments. Could you tell me when you'd like one?"),
    ("human", "Intent: {intent}\nMessage: {message}\nReply:")
])


INTENT_SYNONYMS = {
    "APPOINTMENT_DETAILS": "APPOINTMENT_DETAILS",
    "APPOINT_DETAILS": "APPOINTMENT_DETAILS",
    "APPOINTMENT_REQUEST": "APPOINTMENT_REQUEST",
    "REQUEST_APPOINTMENT": "APPOINTMENT_REQUEST",
    "GREETING": "GREETING",
    "OTHER": "OTHER",
}


def greet_user(state: dict) -> dict:
    state.setdefault("bot_messages", [])
    state.setdefault("greeting_sent", False)
    state.setdefault("awaiting_user_response", False)
    state.setdefault("last_user_intent", "")
    state.setdefault("user_messages", [])

    if state.get("awaiting_user_response"):
        return state

    if not state.get("user_messages"):
        state["awaiting_user_response"] = True
        return state

    user_messages = state.get("user_messages", [])
    user_message = user_messages[-1]

    try:
        intent_chain = intent_prompt | llm
        response = intent_chain.invoke({"input": user_message})

        if response is None:
            state["bot_messages"].append("Sorry, I couldn't process your message. Please try again.")
            state["awaiting_user_response"] = True
            return state

        label = response.strip().upper()
        classification = INTENT_SYNONYMS.get(label, "OTHER")

        state["last_user_intent"] = classification

        if classification == "GREETING":
            response_chain = response_prompt | llm
            response_msg = response_chain.invoke({"intent": classification, "message": user_message})
            state["bot_messages"].append(response_msg.strip())
            state["awaiting_user_response"] = True
        elif classification == "APPOINTMENT_REQUEST":
            response_chain = response_prompt | llm
            response_msg = response_chain.invoke({"intent": classification, "message": user_message})
            state["bot_messages"].append(response_msg.strip())
            state["awaiting_user_response"] = True
        elif classification == "APPOINTMENT_DETAILS":
            state["awaiting_user_response"] = False
        elif classification == "OTHER":
            state["bot_messages"].append(
                "I can only help with booking appointments. Please tell me when you'd like to book and your email address."
            )
            state["awaiting_user_response"] = True
        else:
            state["bot_messages"].append(
                "I'm here to help with booking appointments. Please tell me when you'd like to book."
            )
            state["awaiting_user_response"] = True

    except Exception:
        logger.exception("Error in greet_user")
        state["bot_messages"].append("Sorry, I encountered an error. Please try again.")
        state["awaiting_user_response"] = True

    return state

