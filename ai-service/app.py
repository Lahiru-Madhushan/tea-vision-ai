import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from chatBot.tea_chatbot import ask_question
from services.predictor import predict_disease

app = Flask(__name__)
CORS(app)

os.makedirs("temp", exist_ok=True)


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

@app.route("/predict", methods=["POST"])
def predict():

    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "message": "Image file is required"
        }), 400

    image_file = request.files['image']
    image_path = f"temp/{image_file.filename}"
    image_file.save(image_path)

    disease, confidence = predict_disease(image_path)

    return jsonify({
        "success": True,
        "predicted_disease": disease,
        "confidence": round(confidence * 100, 2),
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )