from typing import TypedDict, List, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage

# --- 1. Define Agent State ---
class AgentState(TypedDict):
    messages: List
    customer_id: Optional[str]
    intent: Optional[str]
    user_query: Optional[str]
    customer_data: Optional[str]
    retrieved_context: Optional[str]

# --- 2. Embedding Model ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# --- 3. Load FAISS Vector Store ---
local_vector_db = FAISS.load_local(
    r'C:\Users\v-niranr\OneDrive - Microsoft\Desktop\Telecom Usecase\.telecomusecase_FAISS_DB',
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

from langchain_core.tools import tool

def RAG_agent(state: AgentState) -> AgentState:
    print("--------------------------RAG Agent Node---------------------------")
    query = state.get("user_query")

    """
    Combines the user query and SQL customer context to retrieve the most relevant information 
    from the vector DB (e.g. troubleshooting guides, plan info, etc.).

    Parameters:
    - query (str): The user's original question (e.g., "Why is my network slow?")
    - customer_context (str): The context retrieved from SQL agent (e.g., customer location, plan, device)

    Returns:
    - str: Top-k retrieved document contents concatenated.
    """

    # Assuming local_vector_db is already loaded globally with embeddings
    retriever = local_vector_db.as_retriever(search_kwargs={"k": 5})
    documents = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in documents])

    print(f"Retrieved context for query '{query}':\n{context}\n")

    return {
        "messages": state["messages"] + [AIMessage(content=context)],
        "retrieved_context": context
    }

