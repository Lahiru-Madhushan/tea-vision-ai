from flask import Flask, request, jsonify
from flask_cors import CORS

from chatBot.tea_chatbot import ask_question

app = Flask(__name__)
CORS(app)


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    question = data.get("question")

    if not question:
        return jsonify({
            "success": False,
            "message": "Question is required"
        }), 400

    answer = ask_question(question)

    return jsonify({
        "success": True,
        "question": question,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )