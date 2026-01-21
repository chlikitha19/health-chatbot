import os
import joblib
import numpy as np
MODEL_PATH = "models/model.pkl"
VEC_PATH = "models/vectorizer.pkl"
if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
else:
    model = None
    vectorizer = None
ADVICE = {
    "Flu": "Rest, fluids, paracetamol if needed. See a doctor if breathing becomes difficult.",
    "COVID-19": "Isolate, get tested, monitor oxygen levels, seek immediate help if breathing is difficult.",
    "Common Cold": "Rest, warm fluids, steam inhalation.",
    "Migraine": "Rest in dark room, avoid triggers, consult a neurologist if frequent.",
    "Food Poisoning": "Hydrate (ORS), rest, seek medical care if vomiting/diarrhea are severe.",
    "Allergy": "Avoid allergen, antihistamines may help, consult an allergist for severe symptoms.",
    "Diabetes": "Get blood sugar tested, reduce sugar intake, consult an endocrinologist.",
    "Chest Problem": "Chest pain can be serious. Seek emergency medical help immediately.",
    "Hypertension": "Check blood pressure, reduce salt, consult a physician for management.",
    "Gastritis": "Avoid spicy and oily food, eat light, seek medical care if severe.",
    "Asthma": "Use inhaler if prescribed, avoid triggers, seek immediate help for severe breathlessness."
}
def predict_disease(text):
    """
    Returns: dict {disease (str), confidence (float 0-1), advice (str)}
    If model not available returns None.
    """
    if model is None or vectorizer is None:
        return None
    text = text.lower()
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]  # array of probabilities
    classes = model.classes_
    best_idx = np.argmax(probs)
    disease = classes[best_idx]
    confidence = float(probs[best_idx])  # convert numpy float
    advice = ADVICE.get(disease, "Please consult a healthcare professional for accurate advice.")
    return {"disease": disease, "confidence": confidence, "advice": advice}
def get_health_response(message):
    message = message.lower()

    if "fever" in message or "temperature" in message:
        return "It seems like you have a fever. Drink fluids and rest. If it persists more than 2 days see a doctor."

    if "cough" in message or "sore throat" in message:
        return "Cough or sore throat — try warm fluids and rest. If breathing difficulty occurs, seek help."
    return "Please describe symptoms in a single line, e.g., 'fever cough tiredness'. I will try to predict possible disease."
