from langgraph.graph import StateGraph, END
from state import AgentState
from agent_qa import make_qa_agent
from agent_verifier import make_verifier_agent

MAX_RETRIES = 2

def build_graph(vectorstore_path: str = "vectorstore"):
    qa_node = make_qa_agent(vectorstore_path)
    verifier_node = make_verifier_agent(vectorstore_path)

    graph = StateGraph(AgentState)

    graph.add_node("qa_agent", qa_node)
    graph.add_node("verifier_agent", verifier_node)

    graph.set_entry_point("qa_agent")
    graph.add_edge("qa_agent", "verifier_agent")

    # ✅ Replace the static edge below with conditional routing:
    # graph.add_edge("verifier_agent", END)  ← remove this

    def route_after_verify(state: AgentState):
        if state["verdict"] == "INCORRECT" and state.get("retry_count", 0) < MAX_RETRIES:
            return "qa_agent"   # retry with justification as hint
        return END

    graph.add_conditional_edges("verifier_agent", route_after_verify)

    return graph.compile()