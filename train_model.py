# ============================================
# 1. IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np
import re
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 2. LOAD DATASET
# ============================================

DATA_PATH = "email_dataset_100000_fixed_final.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
df.head()

# ============================================
# 3. ADVANCED TEXT CLEANING
# ============================================

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def advanced_clean(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove numbers & special chars
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove long repeating characters
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Remove stopwords & lemmatize
    cleaned = []
    for w in text.split():
        if w not in stop_words:
            cleaned.append(lemmatizer.lemmatize(w))
    
    return " ".join(cleaned)

# Apply cleaning
df["clean_subject"] = df["subject"].apply(advanced_clean)
df["clean_body"] = df["body"].apply(advanced_clean)

df["text"] = df["clean_subject"] + " " + df["clean_body"]

# ============================================
# 4. ENCODE LABELS
# ============================================

le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

print("\nLabel Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# ============================================
# 5. TRAIN-VAL-TEST SPLIT
# ============================================

X_train, X_temp, y_train, y_temp = train_test_split(
    df["text"], df["label"], test_size=0.2,
    random_state=42, stratify=df["label"]
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5,
    random_state=42, stratify=y_temp
)

print("\nTrain size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# ============================================
# 6. FEATURE ENGINEERING (VERY POWERFUL)
# ============================================

# Word-level TF-IDF
word_vectorizer = TfidfVectorizer(
    max_features=60000,
    ngram_range=(1,2),
    min_df=3
)

# Character-level TF-IDF (best for spam tricks)
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    min_df=5,
    max_features=40000
)

# Combine both
combined_features = FeatureUnion([
    ("word_tfidf", word_vectorizer),
    ("char_tfidf", char_vectorizer)
])

# ============================================
# 7. ENSEMBLE MODEL (SVM + LR + NB)
# ============================================

svm_model = LinearSVC(C=1.0, class_weight="balanced")
lr_model = LogisticRegression(max_iter=1200, class_weight="balanced")
nb_model = MultinomialNB()

# Custom simple soft voting ensemble
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


ensemble_clf = SoftVotingEnsemble({
    "SVM": svm_model,
    "LR": lr_model,
    "NB": nb_model
})

# Pipeline
pipeline = Pipeline([
    ("features", combined_features),
    ("ensemble", ensemble_clf)
])

# ============================================
# 8. TRAIN MODEL
# ============================================

print("\nTraining Ensemble Model...")
pipeline.fit(X_train, y_train)
print("Training Completed!")

# ============================================
# 9. VALIDATION EVALUATION
# ============================================

val_pred = pipeline.predict(X_val)
print("\nValidation Accuracy:", accuracy_score(y_val, val_pred))
print("Validation Macro F1:", f1_score(y_val, val_pred, average="macro"))
print("\nValidation Report:")
print(classification_report(y_val, val_pred))

# ============================================
# 10. FINAL TEST EVALUATION
# ============================================

test_pred = pipeline.predict(X_test)
print("\nTEST Accuracy:", accuracy_score(y_test, test_pred))
print("TEST Macro F1:", f1_score(y_test, test_pred, average="macro"))
print("\nTEST Report:")
print(classification_report(y_test, test_pred))

# ============================================
# 11. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, test_pred, labels=le.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Ensemble + Char Ngrams)")
plt.show()

# ============================================
# 12. SAVE MODEL
# ============================================

MODEL_PATH = "email_classifier_ENSEMBLE_MODEL.joblib"
joblib.dump({"pipeline": pipeline, "label_encoder": le}, MODEL_PATH)

print("\nModel Saved Successfully At:", MODEL_PATH)

# ============================================
# 13. USER INPUT PREDICTOR
# ============================================

def classify_email(subject, body):
    text = advanced_clean(subject) + " " + advanced_clean(body)
    label = pipeline.predict([text])[0]
    return label

print("\n==== ACCURATE ENSEMBLE CLASSIFIER READY ====")

while True:
    print("\n--- Test Custom Email ---")
    subject = input("Enter Email Subject: ")
    body = input("Enter Email Body: ")

    label = classify_email(subject, body)
    print("\nPredicted Category:", label)

    if input("Test another? (y/n): ").lower() != "y":
        break