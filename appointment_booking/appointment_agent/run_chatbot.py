# run_chatbot.py

from appointment_agent.graph import app_graph


def main():
    # Initialize state as a simple dict
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

    # Send initial greeting
    state["bot_messages"].append(
        "Hello! I'm here to help you book an appointment. How can I assist you today?"
    )
    for msg in state["bot_messages"]:
        print("Chatbot:", msg)
    state["bot_messages"] = []

    # Main loop
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            print("Exiting. Goodbye!")
            break

        print(f"DEBUG: User input: '{user_input}'")
        
        # Add user message to state
        state["user_messages"].append(user_input)
        state["awaiting_user_response"] = False
        
        print(f"DEBUG: State before graph invoke: {state}")

        try:
            # Determine which node to start from based on current state
            if not (state.get("date") and state.get("time") and state.get("email")):
                # If any detail is missing, always go to collect_details
                print("DEBUG: Missing details, going to collect_details")
                state["last_user_intent"] = "APPOINTMENT_DETAILS"
            else:
                print("DEBUG: All details present, going to confirm_booking")
                # We'll handle this in the graph flow
            
            # Invoke graph
            print("DEBUG: About to invoke app_graph")
            result = app_graph.invoke(state)
            print(f"DEBUG: Graph result: {result}")
            
            # Check if result is None (graph error)
            if result is None:
                print("DEBUG: Graph returned None")
                print("Chatbot: Sorry, I encountered an error. Please try again.")
                continue
                
            print("DEBUG: Using result as new state")
            state = result
            print(f"DEBUG: New state: {state}")
            
        except Exception as e:
            print(f"DEBUG: Exception caught: {e}")
            import traceback
            traceback.print_exc()
            print(f"Chatbot: Sorry, I encountered an error: {e}")
            continue

        # Show bot messages
        print("DEBUG: Showing bot messages")
        for msg in state.get("bot_messages", []):
            if msg:
                print("Chatbot:", msg)
        state["bot_messages"] = []

        # If appointment is confirmed, show summary and exit
        if state.get("confirmed") and state.get("date") and state.get("time"):
            print("\nYour appointment has been booked successfully!")
            print(f"Date: {state['date']}")
            print(f"Time: {state['time']}")
            if state.get("email"):
                print(f"Confirmation will be sent to: {state['email']}")
            break


if __name__ == "__main__":
    main()
