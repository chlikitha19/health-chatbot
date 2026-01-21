# train_model.py
import os
import joblib
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure models dir exists
os.makedirs("models", exist_ok=True)

# -------------- Create a small reproducible dataset --------------
# Each example is a free-text list of symptoms. Label is disease.
data = [
    ("fever cough sore throat tiredness", "Flu"),
    ("fever cough loss of smell taste difficulty breathing", "COVID-19"),
    ("runny nose sneezing congestion sore throat", "Common Cold"),
    ("severe headache nausea sensitivity to light", "Migraine"),
    ("stomach pain vomiting diarrhea fever", "Food Poisoning"),
    ("itching rash redness swelling", "Allergy"),
    ("frequent urination excessive thirst blurred vision", "Diabetes"),
    ("chest pain shortness of breath sweating", "Chest Problem"),
    ("high blood pressure headache dizziness", "Hypertension"),
    ("abdominal pain gas bloating indigestion", "Gastritis"),
    ("fever body ache headache cough", "Flu"),
    ("cough sore throat runny nose", "Common Cold"),
    ("nausea vomiting stomach cramps", "Food Poisoning"),
    ("sneezing itchy eyes runny nose", "Allergy"),
    ("severe chest pain pressure in chest", "Chest Problem"),
    ("dizziness fainting fast heartbeat", "Hypertension"),
    ("tiredness weight loss increased appetite", "Diabetes"),
    ("blurred vision headaches high blood sugar", "Diabetes"),
    ("severe unilateral throbbing headache nausea", "Migraine"),
    ("shortness of breath wheeze cough", "Asthma")
]

# Augment dataset slightly by paraphrasing / permutations
augmented = []
for text, label in data:
    words = text.split()
    for _ in range(3):  # add three shuffled variations
        random.shuffle(words)
        augmented.append((" ".join(words), label))
# Combine
texts = [t for t, l in augmented]
labels = [l for t, l in augmented]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

# Vectorize
vectorizer = CountVectorizer(binary=True)  # presence of symptom keywords
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(clf, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("Saved model to models/model.pkl and vectorizer to models/vectorizer.pkl")
