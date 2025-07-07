# agent/sql_agent.py (Updated to use direct SQLite access)

import sqlite3
from typing import Dict, Any
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")


# --- 1. Direct SQL Query Function ---
def get_customer_details(customer_id: str, db_path: str) -> dict:
    db_path = r"C:\Users\niranjan.r.lv.LV-SL2316\Desktop\Telecom Usecase\Telecom_Conversation_Agent\Final Files\telecom_customer.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT 
        "Customer ID", "Customer Name", Age, Gender, Location,
        "Plan Subscribed", "Device Used", "Plan Details",
        "Network Type", "Join Date", "Recent Issue Reported", "Response Provided"
    FROM customers
    WHERE "Customer ID" = ?
    """

    cursor.execute(query, (customer_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    keys = [
        "Customer ID", "Customer Name", "Age", "Gender", "Location",
        "Plan Subscribed", "Device Used", "Plan Details",
        "Network Type", "Join Date", "Recent Issue Reported", "Response Provided"
    ]
    return dict(zip(keys, row))


# --- 2. SQL Agent Node Function ---
def sql_agent(state: dict) -> dict:
    print("------------ SQL Agent Node (Direct SQL) ------------")
    
    customer_id = state.get("customer_id")
    if not customer_id:
        return {
            **state,
            "messages": state["messages"] + [("assistant", "Customer ID is missing.")],
            "customer_data": None
        }
    
    customer_data = get_customer_details(customer_id, db_path="telecom_usecase.db")

    if customer_data is None:
        return {
            **state,
            "messages": state["messages"] + [("assistant", f"No customer found with ID: {customer_id}")],
            "customer_data": None
        }

    print(f"Customer_data: {customer_data}")

    return {
        **state,
        "messages": state["messages"] + [("assistant", f"Customer data fetched for ID {customer_id}.")],
        "customer_data": customer_data
    }