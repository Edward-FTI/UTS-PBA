import io
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.title("Model Evaluation - UTS PBA")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

st.header("1. Load Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='latin1')
    st.subheader("Sample Data")
    st.write(data.head())

    st.subheader("Data Information")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Data Description")
    st.write(data.describe())

    st.header("2. Text Preprocessing")
    data['clean_text'] = data['sentence'].apply(clean_text)
    st.write("Sample cleaned text:", data[['sentence', 'clean_text']].head())


    st.header("3. Select Target Column")
    target_column = st.selectbox("Select Target Column", 
                                 options=['fuel', 'machine', 'others', 'part', 'price', 'service'],
                                 index=0)
    y = data[target_column]


    st.header("4. Feature Extraction")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['clean_text'])
    st.write(f"TF-IDF matrix shape: {X.shape}")


    st.header("5. Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


    st.header("6. Model Training (Cross Validation)")


    models = {
        'SVM_C1_OvR': OneVsRestClassifier(SVC(C=1, kernel='linear', probability=True)),
        'SVM_C10_OvO': OneVsOneClassifier(SVC(C=10, kernel='linear', probability=True)),
        'KNN_5_OvR': OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)),
        'KNN_7_OvO': OneVsOneClassifier(KNeighborsClassifier(n_neighbors=7)),
    }


    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']


    results = []

    for model_name, model in models.items():
        with st.spinner(f"Training {model_name}..."):
            scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, return_train_score=False)
            results.append({
                'Model': model_name,
                'Accuracy': np.mean(scores['test_accuracy']),
                'Precision': np.mean(scores['test_precision_macro']),
                'Recall': np.mean(scores['test_recall_macro']),
                'F1-Score': np.mean(scores['test_f1_macro'])
            })

    results_df = pd.DataFrame(results)
    st.subheader("Cross-Validation Results")
    st.dataframe(results_df)


    best_model_name = results_df.sort_values(by='F1-Score', ascending=False).iloc[0]['Model']
    st.success(f"Best Model Selected: {best_model_name}")


    final_model = models[best_model_name]
    final_model.fit(X_train, y_train)


    joblib.dump(final_model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


    st.header("7. Model Evaluation on Test Set")

    y_pred = final_model.predict(X_test)


    pred_df = pd.DataFrame({
        'True Label': y_test,
        'Predicted Label': y_pred
    })
    st.subheader("Sample Predictions")
    st.write(pred_df.head())


    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())


    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)
    st.pyplot(fig)


    st.header("8. Predict New Text")
    new_text = st.text_area("Enter text to analyze sentiment:")

    if st.button("Predict"):

        loaded_model = joblib.load("sentiment_model.pkl")
        loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

        cleaned_text = clean_text(new_text)
        text_vector = loaded_vectorizer.transform([cleaned_text])

        pred = loaded_model.predict(text_vector)

        st.success(f"Predicted sentiment: {pred[0]}")
