from typing import TypedDict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

class AgentState(TypedDict):
    messages: List
    customer_id: Optional[str]
    intent: Optional[str]
    user_query: Optional[str]
    customer_data: Optional[str]
    retrieved_context: Optional[str]

def router_node(state: AgentState) -> str:
    print("--------------------------Router Node---------------------------")
    intent = state.get("intent", "").lower()

    if intent == "plan":
        return "Plan_Explainer"
    elif intent == "complaint":
        return "RAG_agent"
    else:
        return "summarizer_agent"
