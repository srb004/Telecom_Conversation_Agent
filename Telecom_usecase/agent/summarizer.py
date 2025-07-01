from typing import TypedDict, List, Optional
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
import re
import ast


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
    customer_id = state.get("customer_id")
    user_query = state["messages"][0].content

    # Get the last assistant message
    last_message = state["messages"][-1]

    try:
        # Parse the string as a Python dict
        plan_explainer_msg = ast.literal_eval(last_message.content)
    except Exception as e:
        print("Failed to parse message content:", e)
        plan_explainer_msg = {}

    plan_details = plan_explainer_msg.get("plan_details", "")
    query_response = plan_explainer_msg.get("query_response", "")
    cross_sell_recommendation = plan_explainer_msg.get("cross_sell_recommendation", "")
    reasoning = plan_explainer_msg.get("reasoning", "")

    print("Intent:", intent)
    print("Customer ID:", customer_id)
    print("User Query:", user_query)
    print("Plan Details:", plan_details)
    print("Query Response:", query_response)
    print("Cross-sell Recommendation:", cross_sell_recommendation)

    base_inputs = {
        "customer_id": customer_id,
        "query": user_query,
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
User Query: {user_query}
Plan Details: {plan_details}
Query Response: {query_response}
Recommended Upgrade: {cross_sell_recommendation}
        """

        inputs = {
            **base_inputs,
            "user_query": user_query,  #  corrected to user_query
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
User Query: {user_query}
        """
        inputs = base_inputs

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke(inputs)

    raw_summary = response.content if hasattr(response, "content") else str(response)
    summary_text = re.sub(r"<think>.*?</think>", "", raw_summary, flags=re.DOTALL).strip()
    print("Summary Text:", summary_text)
    print("Generated Summary:", summary_text)

    return {
        **state,
        "messages": state.get("messages", []) + [AIMessage(content=summary_text)],
        "summary": summary_text
    }



