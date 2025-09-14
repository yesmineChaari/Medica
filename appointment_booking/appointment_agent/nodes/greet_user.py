from prompt_toolkit import prompt
from appointment_booking.appointment_agent import state
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

llm = OllamaLLM(model="llama3")

# Prompt to classify user input intent
intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an intent classification assistant for an appointment booking chatbot.\n\n"
            "Classify the user's message as exactly one of the following intents:\n"
            "1. GREETING – The user is greeting you.\n"
            "2. APPOINTMENT_REQUEST – The user wants to book an appointment but gives NO specific date/time/email.\n"
            "3. APPOINTMENT_DETAILS – The user provides specific date, time, or email info.\n"
            "4. OTHER – Everything else.\n\n"
            "IMPORTANT RULE:\n"
            "If the message contains ANY date, time, or email — even if it also includes a request to book — classify it as APPOINTMENT_DETAILS.\n"
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


def greet_user(state: dict) -> dict:
    print("DEBUG: greet_user function called")
    print(f"DEBUG: Input state type: {type(state)}")
    print(f"DEBUG: Input state keys: {list(state.keys())}")
    print(f"DEBUG: Input state user_messages: {state.get('user_messages', 'NOT_FOUND')}")
    
    # Ensure all required keys exist
    state.setdefault("bot_messages", [])
    state.setdefault("greeting_sent", False)
    state.setdefault("awaiting_user_response", False)
    state.setdefault("last_user_intent", "")
    state.setdefault("user_messages", [])

    print(f"DEBUG: After setdefault, user_messages: {state.get('user_messages', 'NOT_FOUND')}")

    # If we're already awaiting user response, don't process further
    if state.get("awaiting_user_response"):
        print("DEBUG: Already awaiting user response, returning")
        return state

    # Check if we have user messages to process
    if not state.get("user_messages"):
        # No user messages yet, this is the initial state
        print("DEBUG: No user messages, setting awaiting_user_response=True")
        state["awaiting_user_response"] = True
        return state

    # Get the last user message
    user_messages = state.get("user_messages", [])
    if not user_messages:
        print("DEBUG: No last user message, setting awaiting_user_response=True")
        state["awaiting_user_response"] = True
        return state
    
    user_message = user_messages[-1]
    print(f"DEBUG: Processing user message: '{user_message}'")

    try:
        # Classify intent
        print("DEBUG: Creating prompt messages")
        prompt_messages = intent_prompt.format_messages(input=user_message)
        print("DEBUG: Invoking LLM")
        response = llm.invoke(prompt_messages)
        print(f"DEBUG: LLM response: {response}")
        
        # Handle potential None response
        if response is None:
            print("DEBUG: LLM returned None")
            state["bot_messages"].append("Sorry, I couldn't process your message. Please try again.")
            state["awaiting_user_response"] = True
            return state
            
        print(f"DEBUG: Processing response: {response}")
        INTENT_SYNONYMS = {
            "APPOINTMENT_DETAILS": "APPOINTMENT_DETAILS",
            "APPOINT_DETAILS": "APPOINTMENT_DETAILS",
            "APPOINTMENT_REQUEST": "APPOINTMENT_REQUEST",
            "REQUEST_APPOINTMENT": "APPOINTMENT_REQUEST",
            "GREETING": "GREETING",
            "OTHER": "OTHER"
        }

        label = response.strip().upper()
        classification = INTENT_SYNONYMS.get(label, "OTHER")
        print(f"DEBUG: Classification: {classification}")

        state["last_user_intent"] = classification

        if classification == "GREETING":
            print("DEBUG: Adding greeting response")
            prompt = response_prompt.format_messages(intent=classification, message=user_message)
            response_msg = llm.invoke(prompt)
            state["bot_messages"].append(response_msg.strip())
            state["awaiting_user_response"] = True
        elif classification == "APPOINTMENT_REQUEST":
            print("DEBUG: Adding appointment request response")
            prompt = response_prompt.format_messages(intent=classification, message=user_message)
            response_msg = llm.invoke(prompt)
            state["bot_messages"].append(response_msg.strip())
            state["awaiting_user_response"] = True
        elif classification == "APPOINTMENT_DETAILS":
            print("DEBUG: User provided details, moving to collect_details")
            # If user provided details, move to collect_details node
            state["awaiting_user_response"] = False
            # Don't add a bot message here, let collect_details handle it
        elif classification == "OTHER":
            print("DEBUG: Adding other response")
            state["bot_messages"].append(
                "I can only help with booking appointments. Please tell me when you'd like to book and your email address."
            )
            state["awaiting_user_response"] = True
        else:
            print("DEBUG: Adding default response")
            state["bot_messages"].append(
                "I'm here to help with booking appointments. Please tell me when you'd like to book."
            )
            state["awaiting_user_response"] = True
            
        print("DEBUG: Function completed successfully")
            
    except Exception as e:
        print(f"Error in greet_user: {e}")
        import traceback
        traceback.print_exc()
        state["bot_messages"].append("Sorry, I encountered an error. Please try again.")
        state["awaiting_user_response"] = True
    
    return state




