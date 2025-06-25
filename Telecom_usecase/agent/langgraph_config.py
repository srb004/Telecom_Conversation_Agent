
from langgraph.graph import StateGraph, END, START
from agent.supervisor_node import supervisor_node
from agent.Sql_agent import sql_agent
from agent.rag_agent import rag_agent
from agent.plan_summary import Plan_Explainer
from agent.summarizer import summarizer_agent
# from agent.router import router_node, AgentState  # Assuming both are defined
from agent.router import Router, AgentState


# Build graph
app_workflow = StateGraph(AgentState)

app_workflow.add_node("supervisor_agent", supervisor_node)
app_workflow.add_node("SQL_agent", sql_agent)
app_workflow.add_node("RAG_agent", rag_agent)
app_workflow.add_node("Plan_Explainer", Plan_Explainer)
app_workflow.add_node("summarizer_agent", summarizer_agent)

app_workflow.add_edge(START, "supervisor_agent")
app_workflow.add_edge("supervisor_agent", "SQL_agent")

app_workflow.add_conditional_edges(
    "SQL_agent",
    Router,
    {
        "Plan_Explainer": "Plan_Explainer",
        "RAG_agent": "RAG_agent",
        "summarizer_agent": "summarizer_agent"
    }
)

app_workflow.add_edge("Plan_Explainer", "summarizer_agent")
app_workflow.add_edge("RAG_agent", "summarizer_agent")
app_workflow.add_edge("summarizer_agent", END)

# ✅ IMPORTANT
app = app_workflow.compile()
