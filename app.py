# app.py
from flask import Flask, render_template, request, jsonify
from chatbot import predict_disease, get_health_response
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # existing chat endpoint (rule-based)
    user_input = request.json.get("message", "")
    # try ML first
    ml_result = predict_disease(user_input)
    if ml_result:
        # If ML confidence is decent (>0.4), return ML prediction
        if ml_result["confidence"] >= 0.4:
            reply = (f"Predicted: {ml_result['disease']} "
                     f"(confidence: {ml_result['confidence']:.2f})\nAdvice: {ml_result['advice']}")
            return jsonify({"response": reply})
    # fallback to rule-based
    bot_response = get_health_response(user_input)
    return jsonify({"response": bot_response})
@app.route("/predict", methods=["POST"])
def predict():
    # a raw prediction endpoint that returns structured info
    user_input = request.json.get("message", "")
    ml_result = predict_disease(user_input)
    if ml_result:
        return jsonify({"ok": True, "prediction": ml_result})
    else:
        return jsonify({"ok": False, "error": "Model not available. Please run training (train_model.py)."})
if __name__ == "__main__":
    app.run(debug=True)
