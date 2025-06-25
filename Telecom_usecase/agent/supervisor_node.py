from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq  # Adjust import if you use a different provider


# --- 1. Agent State Definition ---
class AgentState(TypedDict):
    messages: List
    customer_id: Optional[str]
    intent: Optional[str]
    user_query: Optional[str]
    customer_data: Optional[str]
    retrieved_context: Optional[str]


# --- 2. Intent Classifier Model ---
class Intent_Classifier(BaseModel):
    """Intent Classifier"""
    customer_id: str = Field(description="The customer ID in the format CUSTXXXX")
    intent: str = Field(description="Intent of the user query")
    query: str = Field(description="User query")
    Reasoning: str = Field(description='Reasoning behind topic selection')


# --- 3. Parser and LLM Setup ---
parser = PydanticOutputParser(pydantic_object=Intent_Classifier)

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b", streaming=True)

# --- 4. Supervisor Node ---
def supervisor_node(state: AgentState) -> AgentState:
    print("--------------------------Supervisor---------------------------")
    user_question = state["messages"][-1].content

    prompt = PromptTemplate(
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
        You are a telecom assistant that classifies the user's intent into one of the following:
        - Plan: If the user is asking about mobile, broadband, or 5G plans.
        - Complaint: If the user is reporting issues like network outage, slow internet, or billing problems.
        - Other: For greetings, general questions, or unrelated topics.

        Your job is to classify the intent and explain your reasoning.

        User Query: {question}

        {format_instructions}
        """
    )

    chain = prompt | llm | parser
    response = chain.invoke({"question": user_question})

    print("Parsed Response: ", response)

    return {
        "messages": state["messages"] + [
            AIMessage(content=f"Intent: {response.intent}"),
            AIMessage(content=f"Reasoning: {response.Reasoning}"),
            AIMessage(content=f"Customer ID: {response.customer_id}")
        ],
        "customer_id": response.customer_id,
        "intent": response.intent,
        "user_query": response.query,
        "original_intent": response.intent
    }
