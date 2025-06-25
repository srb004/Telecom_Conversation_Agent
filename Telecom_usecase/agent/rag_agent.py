from typing import TypedDict, List, Optional
# from langchain.embeddings import HuggingFaceEmbeddings
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

# --- 4. RAG Agent Node Function ---
def rag_agent(state: AgentState) -> AgentState:
    print("--------------------------RAG Agent Node---------------------------")
    query = state.get("user_query")

    if not query:
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="No query found to retrieve context.")],
            "retrieved_context": None
        }

    # Retrieve from vector store
    retriever = local_vector_db.as_retriever(search_kwargs={"k": 1})
    documents = retriever.invoke(query)

    # Concatenate retrieved docs
    context = "\n\n".join([doc.page_content for doc in documents])
    print(f"Retrieved context for query '{query}':\n{context}\n")

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=context)],
        "retrieved_context": context
    }
