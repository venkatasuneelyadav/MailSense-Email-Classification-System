# 📧 MailSense – AI Email Classification System

MailSense is a **GenAI-powered intelligent email classification system** designed to automatically categorize emails using **Machine Learning and Natural Language Processing (NLP)** techniques.  
The system improves productivity by identifying important emails and filtering unwanted messages such as spam and promotions.

---

## 🎯 Project Objectives
- Automatically classify emails into predefined categories
- Reduce manual effort in email sorting
- Provide real-time predictions with confidence scores
- Maintain user-specific prediction history
- Offer a professional web-based interface

---

## 🧠 Key Concepts Used
- Natural Language Processing (NLP)
- TF-IDF Feature Engineering (Word + Character N-grams)
- Supervised Machine Learning
- Soft Voting Ensemble Model
- Web Application Development using Flask
- SQLite Database Management
- Authentication & Authorization
- Data Visualization & Reporting

---

## ⚙️ Technology Stack

### 🔹 Frontend
- HTML5
- CSS3
- Bootstrap 5
- JavaScript
- Dark Mode UI

### 🔹 Backend
- Python
- Flask Framework

### 🔹 Machine Learning
- Scikit-learn
- TF-IDF Vectorization
- Linear SVM
- Logistic Regression
- Multinomial Naive Bayes

### 🔹 Database
- SQLite

---

## 🏗️ System Architecture
1. User logs in or signs up
2. Email subject & body are entered
3. Text is cleaned and preprocessed
4. TF-IDF features are generated
5. Ensemble model predicts category
6. Prediction + confidence is stored
7. User can view history or download CSV

---

## 🧪 Machine Learning Model Details

### Feature Engineering
- Word-level TF-IDF (unigrams & bigrams)
- Character-level TF-IDF (3–5 grams)

### Model Architecture
- Linear Support Vector Machine (SVM)
- Logistic Regression
- Multinomial Naive Bayes

### Ensemble Strategy
- Custom **Soft Voting Ensemble**
- Majority voting across models
- Confidence score computed using probability averaging

---

## 🗂️ Project Structure

```

MailSense-AI-Email-Classification-System/
│
├── app.py
├── classifier.py
├── database.py
├── auth.py
├── train_model.py
├── requirements.txt
├── email_classifier_ENSEMBLE_MODEL.joblib
│
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── signup.html
│   ├── about.html
│   ├── dashboard.html
│   ├── history.html
│   ├── contact.html
│   ├── team.html
│
├── static/
│   ├── styles.css
│   ├── scripts.js
│
├── README.md
└── .gitignore

````

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/MailSense-AI-Email-Classification-System.git
cd MailSense-AI-Email-Classification-System
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python app.py
```

### 4️⃣ Open Browser

```
http://127.0.0.1:5000
```

---

## 📥 Features

* Secure Login & Signup
* Email Classification with Confidence Score
* Prediction History Tracking
* CSV Download of History
* Dark Mode Toggle
* About, Contact & Team Pages

---

## 📊 Evaluation Metrics

* Accuracy
* Macro F1-Score
* Confusion Matrix
* Classification Report

---

## 🎓 Project Information

* **Project Type:** Final Year Major Project (GenAI)
* **Department:** Computer Science & Engineering
* **University:** SRM University AP

---

## 👥 Team Members

* **Ashish Muppalla** – Machine Learning Engineer
* **Pranav Krishna** – Machine Learning Engineer
* **Taraka Prabhu** – Machine Learning Engineer
* **Nikhil** – Machine Learning Engineer
* **Suneel** – Machine Learning Engineer

---

## 📜 License

This project is developed for academic purposes under SRM University AP.

---

## ⭐ Acknowledgements

* Scikit-learn Documentation
* NLTK Library
* Flask Documentation
* SRM University AP Faculty