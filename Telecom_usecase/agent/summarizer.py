from typing import TypedDict, List, Optional
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
import re

# --- AgentState Definition ---
class AgentState(TypedDict):
    messages: List
    customer_id: Optional[str]
    intent: Optional[str]
    user_query: Optional[str]
    customer_data: Optional[str]
    retrieved_context: Optional[str]
    complaint_resolution: Optional[str]
    query_response: Optional[str]
    plan_details: Optional[str]
    cross_sell_recommendation: Optional[str]

# --- LLM Initialization ---
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.3, streaming=True)

# --- Summarizer Node ---
def summarizer_agent(state: AgentState) -> AgentState:
    print("--------------------------Summarizer Node---------------------------")

    intent = state.get("intent", "").lower()
    print("Intent:", intent)

    customer_id = state.get("customer_id", "Unknown")
    query = state.get("user_query", "N/A")
    query_response = state.get("query_response", "N/A")
    plan_details = state.get("plan_details", "N/A")
    cross_sell_recommendation = state.get("cross_sell_recommendation", "N/A")

    base_inputs = {
        "customer_id": customer_id,
        "query": query,
    }

    if intent == "plan":
        template = """
You are a warm, friendly telecom assistant. Write a helpful and human-like message in response to the user's plan concerns.

- Begin with a friendly greeting and apology (include emoji like 👋 and 😊).
- Mention the current plan (use what's given in "Plan Details").
- Acknowledge the user's concern and recommend an upgraded plan if relevant.
- Explain benefits in a simple and friendly tone.
- End with a question like: "Would you like me to help you upgrade or explore other options?"

Compose the message in a conversational tone. DO NOT include <think> or reasoning steps.

---
Customer ID: {customer_id}
User Query: {query}
Plan Details: {plan_details}
Query Response: {query_response}
Recommended Upgrade: {cross_sell_recommendation}
        """
        inputs = {
            **base_inputs,
            "plan_details": plan_details,
            "query_response": query_response,
            "cross_sell_recommendation": cross_sell_recommendation,
        }

    elif intent == "complaint":
        grievance_context = state.get("retrieved_context", "")
        complaint_response = state.get("complaint_resolution", "")

        template = """
You are a kind and professional telecom assistant helping a customer with a complaint.

Your goal is to:
- Understand the user’s concern from their query.
- Read the retrieved context (which may include a general solution from other similar cases).
- Use the resolution (if any) to provide a reassuring and helpful response.
- Empathize with the customer. Keep your tone friendly, polite, and professional.
- DO NOT refer to the context or resolution as being from another user — just use it to help you answer.
- DO NOT mention bullet points, tags, or internal data. Just give a natural, conversational response in 3–4 lines.

---
Customer ID: {customer_id}
User Query: {query}
Context: {grievance_context}
Resolution: {complaint_response}
        """
        inputs = {
            **base_inputs,
            "grievance_context": grievance_context,
            "complaint_response": complaint_response
        }

    else:
        template = """
You are a friendly assistant. Acknowledge the user's query and let them know you'll look into it.

---
Customer ID: {customer_id}
User Query: {query}
        """
        inputs = base_inputs

    # Prompt + Chain + Invoke
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke(inputs)

    raw_summary = response.content if hasattr(response, "content") else str(response)

    # Remove <think> ... </think> section if present
    summary_text = re.sub(r"<think>.*?</think>", "", raw_summary, flags=re.DOTALL).strip()
    print("Summary Text:", summary_text)

    return {
        **state,
        "messages": state.get("messages", []) + [AIMessage(content=summary_text)],
        "summary": summary_text
    }
