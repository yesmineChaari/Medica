# graph.py
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from appointment_booking.appointment_agent.nodes.greet_user import greet_user
from appointment_booking.appointment_agent.nodes.collect_details import collect_details
from appointment_booking.appointment_agent.nodes.confirm_booking import confirm_booking
from appointment_booking.appointment_agent.nodes.create_appointment import create_appointment
from appointment_booking.appointment_agent.nodes.router import route

graph = StateGraph(dict)

graph.add_node("greet_user", greet_user)
graph.add_node("collect_details", collect_details)
graph.add_node("confirm_booking", confirm_booking)
graph.add_node("create_appointment", create_appointment)

# Router is the single entry point for every turn
graph.set_entry_point("router")
graph.add_node("router", lambda s: s)        # passthrough — routing is in the condition
graph.add_conditional_edges("router", route) # route() returns the node name as a string
graph.add_conditional_edges(
    "greet_user",
    lambda s: "collect_details" if s.get("last_user_intent") == "APPOINTMENT_DETAILS" else END
)
graph.add_conditional_edges(
    "collect_details",
    lambda s: "confirm_booking" if s.get("date") and s.get("time") and s.get("email") else END
)
graph.add_conditional_edges(
    "confirm_booking",
    lambda s: "create_appointment" if s.get("confirmed") is True else END
)
graph.add_edge("create_appointment", END)

app_graph = graph.compile(checkpointer=MemorySaver())