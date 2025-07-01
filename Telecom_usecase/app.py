from flask import Flask, request, jsonify, render_template
from agent.langgraph_config import app as langgraph_app  # ✅ your LangGraph workflow
from langchain_core.messages import HumanMessage

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # loads your HTML page

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("user_input", "")

    # Format input into LangGraph's expected state format
    state = {
        "messages": [HumanMessage(content=user_input)],
        "customer_id": None,
        "intent": None,
        "user_query": user_input,
        "customer_data": None,
        "retrieved_context": None
    }

    try:
        result = langgraph_app.invoke(state)
        # Get the final message content
        final_message = result["messages"][-1].content
    except Exception as e:
        final_message = f"Something went wrong: {str(e)}"

    return jsonify({"response": final_message})

if __name__ == "__main__":
    app.run(debug=True)