from appointment_booking.appointment_agent.graph import app_graph
from dotenv import load_dotenv

load_dotenv()


def main():
    state = {
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
    
    print("=== Appointment Chatbot ===")
    print("Type 'quit' anytime to exit.\n")

    state["bot_messages"].append(
        "Hello! I'm here to help you book an appointment. How can I assist you today?"
    )
    for msg in state["bot_messages"]:
        print("Chatbot:", msg)
    state["bot_messages"] = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            print("Exiting. Goodbye!")
            break

        state["user_messages"].append(user_input)
        state["awaiting_user_response"] = False

        try:
            config = {"configurable": {"thread_id": "cli-session"}}
            result = app_graph.invoke(state, config=config)

            if result is None:
                print("Chatbot: Sorry, I encountered an error. Please try again.")
                continue

            state = result

        except Exception as e:
            print(f"Chatbot: Sorry, I encountered an error: {e}")
            continue

        for msg in state.get("bot_messages", []):
            if msg:
                print("Chatbot:", msg)
        state["bot_messages"] = []

        if state.get("confirmed") and state.get("date") and state.get("time"):
            print("\nYour appointment has been booked successfully!")
            print(f"Date: {state['date']}")
            print(f"Time: {state['time']}")
            if state.get("email"):
                print(f"Confirmation will be sent to: {state['email']}")
            break


if __name__ == "__main__":
    main()
