# =====================================================
# BEAUTIFUL EMAIL CLASSIFIER UI (CustomTkinter)
# =====================================================

import customtkinter as ctk
import joblib
import numpy as np

class SoftVotingEnsemble:
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for name, m in self.models.items():
            print(f"Training {name}...")
            m.fit(X, y)
    
    def predict(self, X):
        preds = []
        for name, m in self.models.items():
            p = m.predict(X)
            preds.append(p)

        preds = np.array(preds)
        final = []

        for i in range(preds.shape[1]):
            values, counts = np.unique(preds[:, i], return_counts=True)
            final.append(values[np.argmax(counts)])

        return final


# Load Model


MODEL_PATH = "email_classifier_ENSEMBLE_MODEL.joblib"

model_data = joblib.load(MODEL_PATH)
pipeline = model_data["pipeline"]
label_encoder = model_data["label_encoder"]

# --------------------------------------
# CLEANING FUNCTION (same as training)
# --------------------------------------
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def advanced_clean(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    cleaned = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(cleaned)


# ---------------------------------------------------
# Prediction Function (includes CONFIDENCE score)
# ---------------------------------------------------
def predict_email(subject, body):
    text = advanced_clean(subject) + " " + advanced_clean(body)
    
    vectorized = pipeline.named_steps["features"].transform([text])

    svm_pred = pipeline.named_steps["ensemble"].models["SVM"].predict(vectorized)[0]
    lr = pipeline.named_steps["ensemble"].models["LR"]
    nb = pipeline.named_steps["ensemble"].models["NB"]

    lr_proba = lr.predict_proba(vectorized)[0]
    nb_proba = nb.predict_proba(vectorized)[0]

    avg_proba = (lr_proba + nb_proba) / 2
    confidence = float(np.max(avg_proba))

    preds = np.array([
        svm_pred,
        lr.predict(vectorized)[0],
        nb.predict(vectorized)[0]
    ])

    values, counts = np.unique(preds, return_counts=True)
    final_label = values[np.argmax(counts)]

    return final_label, round(confidence, 4)



# =====================================================
#                  UI STARTS HERE
# =====================================================

ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")  

app = ctk.CTk()
app.title("AI-Powered Email Classifier")
app.geometry("900x650")


# -------------------------
# Heading
# -------------------------
title = ctk.CTkLabel(app, text="📬 Email Classification System",
                     font=("Arial Rounded MT Bold", 32))
title.pack(pady=20)


# -------------------------
# SUBJECT INPUT
# -------------------------
subject_label = ctk.CTkLabel(app, text="Email Subject:", font=("Arial", 18))
subject_label.pack(anchor="w", padx=40)

subject_entry = ctk.CTkEntry(app, width=800, height=40, font=("Arial", 16))
subject_entry.pack(padx=40, pady=10)


# -------------------------
# BODY INPUT
# -------------------------
body_label = ctk.CTkLabel(app, text="Email Body:", font=("Arial", 18))
body_label.pack(anchor="w", padx=40)

body_text = ctk.CTkTextbox(app, width=800, height=200, font=("Arial", 15))
body_text.pack(padx=40, pady=10)


# -------------------------
# RESULT BOX
# -------------------------
result_frame = ctk.CTkFrame(app, width=800, height=140, corner_radius=20)
result_frame.pack(pady=20)

result_label = ctk.CTkLabel(result_frame, text="Prediction will appear here",
                            font=("Arial Rounded MT Bold", 22))
result_label.pack(pady=20)


# -------------------------
# PREDICT BUTTON
# -------------------------
def on_predict():
    subject = subject_entry.get()
    body = body_text.get("0.0", "end")

    label, conf = predict_email(subject, body)

    result_label.configure(
        text=f"📌 Prediction: {label.upper()}\n🔥 Confidence: {conf*100:.2f}%"
    )


predict_btn = ctk.CTkButton(app, text="Predict Category", height=50,
                            font=("Arial Rounded MT Bold", 20), command=on_predict)
predict_btn.pack(pady=10)



# RUN APP
app.mainloop()