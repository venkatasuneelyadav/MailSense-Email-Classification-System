import sys
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------
#  FIX: Define and register SoftVotingEnsemble
# ---------------------------------------------
class SoftVotingEnsemble:
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for name, m in self.models.items():
            m.fit(X, y)
    
    def predict(self, X):
        preds = []
        for name, m in self.models.items():
            preds.append(m.predict(X))

        preds = np.array(preds)

        final = []
        for i in range(preds.shape[1]):
            values, counts = np.unique(preds[:, i], return_counts=True)
            final.append(values[np.argmax(counts)])

        return final

# Register the class under __main__ (required for joblib)
sys.modules["__main__"].SoftVotingEnsemble = SoftVotingEnsemble


# ----------------------------
# Load model after registering
# ----------------------------
model_data = joblib.load("email_classifier_ENSEMBLE_MODEL.joblib")
pipeline = model_data["pipeline"]
label_encoder = model_data["label_encoder"]


# ---------------------------------------------
# NLTK Setup
# ---------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------------------------------------
# Cleaning Function
# ---------------------------------------------
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


# ---------------------------------------------
# Prediction Function
# ---------------------------------------------
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

    return final_label, confidence
