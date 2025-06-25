# agent/sql_agent.py

import re
import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent


# --- 1. LLM and DB Setup ---
sql_llm = ChatGroq(model="gemma2-9b-it", temperature=0.2, streaming=True)

db = SQLDatabase.from_uri("sqlite:///C:/Users/v-niranr/OneDrive - Microsoft/Desktop/Telecom Usecase/Data/telecom_usecase.db")
toolkit = SQLDatabaseToolkit(db=db, llm=sql_llm)

agent_executor = initialize_agent(
    tools=toolkit.get_tools(),
    llm=sql_llm,
    agent="zero-shot-react-description",
    verbose=True
)


# --- 2. Helper Function to Parse JSON ---
def extract_json_from_string(text: str) -> Dict[str, Any]:
    text = text.strip("` \n")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response.")
    json_str = match.group(0)
    json_str = json_str.replace("\\x93", "-").replace("'", '"')
    return json.loads(json_str)


# --- 3. SQL Agent Node Function ---
def sql_agent(state: dict) -> dict:
    print("------------ SQL Agent Node ------------")

    customer_id = state.get("customer_id")
    if not customer_id:
        return {
            **state,
            "messages": state["messages"] + [("assistant", "Customer ID is missing.")],
            "customer_data": None
        }

    query = f"Get customer details for ID '{customer_id}' from the 'customers' table. Respond only with the result in JSON format."

    response = agent_executor.invoke({"input": query})
    full_output = response.get("output", str(response))

    try:
        parsed_data = extract_json_from_string(full_output)
    except Exception as e:
        parsed_data = None
        full_output += f"\n\n Parsing error: {str(e)}"

    return {
        **state,
        "messages": state["messages"] + [("assistant", full_output)],
        "customer_data": parsed_data
    }

