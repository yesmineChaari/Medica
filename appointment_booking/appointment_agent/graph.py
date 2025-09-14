from langgraph.graph import END, StateGraph

from appointment_booking.appointment_agent.nodes.greet_user import greet_user
from appointment_booking.appointment_agent.nodes.collect_details import collect_details
from appointment_booking.appointment_agent.nodes.confirm_booking import confirm_booking
from appointment_booking.appointment_agent.nodes.create_appointment import create_appointment

# Use a simple dict for state instead of custom class
graph = StateGraph(dict)

graph.add_node("greet_user", greet_user)
graph.add_node("collect_details", collect_details)
graph.add_node("confirm_booking", confirm_booking)
graph.add_node("create_appointment", create_appointment)

graph.set_entry_point("greet_user")

# Greet user node transitions
graph.add_conditional_edges(
    "greet_user",
    lambda s: (
        "collect_details"
        if s.get("last_user_intent") == "APPOINTMENT_DETAILS"
        else END
    )
)

# Collect details node transitions
graph.add_conditional_edges(
    "collect_details",
    lambda s: (
        "confirm_booking"
        if s.get("date") and s.get("time") and s.get("email")
        else END
    )
)

# Confirm booking node transitions
graph.add_conditional_edges(
    "confirm_booking",
    lambda s: (
        "create_appointment"
        if s.get("confirmed") is True
        else END
    )
)

# Create appointment node goes to END
graph.add_edge("create_appointment", END)

# Compile the graph
app_graph = graph.compile()
