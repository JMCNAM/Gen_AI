# app.py
from flask import Flask, request, jsonify
from langchain_llms import get_weather  # Import from your existing code
from langchain_testing import ask_question  # Import from your existing code

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = ask_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5001)