import json
import re
from typing import TypedDict, List, Optional
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# --- 1. Define Agent State ---
class AgentState(TypedDict):
    messages: List
    customer_id: Optional[str]
    intent: Optional[str]
    user_query: Optional[str]
    customer_data: Optional[str]
    retrieved_context: Optional[str]

# --- 2. Initialize LLM & Parser ---
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0, streaming=True)
plan_explainer_parser = StrOutputParser()

# --- 3. Plan Explainer Node Function ---
def Plan_Explainer(state: AgentState) -> AgentState:
    print("--------------------------Plan Explainer---------------------------")

    query = state.get("user_query")
    intent = "plan"
    customer_id = state.get("customer_id")
    customer_data = state.get("customer_data")

    plan_info = """
    1. Basic Plan: $20–$30/month, includes 5 GB high-speed data, unlimited calls, 1000 SMS/month, 128 Kbps after limit, no roaming.
    2. Unlimited Plan: $30–$50/month, truly unlimited data (FUP after 50 GB), unlimited calls/SMS, 1 Mbps post-FUP, 5G, cloud storage, hotspot.
    3. Family Plan: $60–$100/month for 3–5 lines, shared 50–100 GB data, unlimited calls/SMS, 256 Kbps post-limit, includes parental controls.
    4. Premium Plan: $40–$60/month, unlimited high-speed (throttled after 100 GB), includes international calls, priority support, streaming.
    5. Data-Only Plan: $15–$25/month, 10–20 GB high-speed data, no calls/SMS, 512 Kbps post-limit, supports tethering and hotspots.
    """

    prompt = PromptTemplate(
        input_variables=["customer_data", "query", "intent", "customer_id", "plan_info"],
        template="""
        You are a telecom marketing assistant. Your job is to address the user by name and explain the user's current plan clearly and offer an upgrade suggestion **only if appropriate**.

        You must perform the following tasks:
        1. Summarize the customer's **current plan** in simple, friendly language.
        2. Explain how the plan features address the user's query.
        3. If the plan is **outdated**, **limited**, or **not optimal** based on their usage, suggest an **upgrade** (cross-sell) with a compelling reason.
        4. Be helpful, not pushy. Do not upsell if the user is already on a top-tier plan.
        5. Keep the intent in mind while responding.
        6. Do not change the intent response — keep it exactly like "Plan" or "Complaint", no additional words.

        Use the information below:

        ---
        **Available Data**:
        {plan_info}

        **Customer Data**:
        {customer_data}

        **User Query**:
        {query}

        **Intent**: {intent}  
        
        **Customer ID**: {customer_id}

        Given a customer's current data and query, respond in the following JSON format:

        {{
        "plan_details": "...",
        "query_response": "...",
        "cross_sell_recommendation": "...",
        "reasoning": "..."
        }}
        """
    )

    chain = prompt | llm | plan_explainer_parser

    response_text = chain.invoke({
        "query": query,
        "customer_data": customer_data,
        "intent": intent,
        "customer_id": customer_id,
        "plan_info": plan_info
    })

    print("Plan Explainer Response:", response_text)

    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if json_match:
        json_part = json_match.group(0)
        try:
            response = json.loads(json_part)
        except json.JSONDecodeError as e:
            print("JSON parsing error:", e)
            return state
    else:
        print("No JSON object found in response.")
        return state

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=str(response))],
        "plan_details": response["plan_details"],
        "query_response": response["query_response"],
        "cross_sell_recommendation": response["cross_sell_recommendation"],
        "reasoning": response["reasoning"],
        "intent": "plan",
        "customer_id": customer_id
    }
