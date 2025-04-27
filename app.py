import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.title("Sentiment Analysis App - UTS Deployment")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# 1. Data Loading
st.header("1. Load Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='latin1')
    st.write("Sample data:", data.head())

    # 2. Text Preprocessing
    st.header("2. Text Preprocessing")
    data['clean_text'] = data['sentence'].apply(clean_text)
    st.write("Sample cleaned text:", data[['sentence', 'clean_text']].head())

    # 3. Target Column Selection
    st.header("3. Select Target Column")
    target_column = st.selectbox("Select Target Column", 
                               options=['fuel', 'machine', 'others', 'part', 'price', 'service'],
                               index=0)
    
    y = data[target_column]
    
    # 4. Feature Extraction (TF-IDF)
    st.header("4. Feature Extraction")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['clean_text'])
    st.write(f"TF-IDF matrix shape: {X.shape}")

    # 5. Model Training
    st.header("5. Model Training")
    model_option = st.selectbox("Select Model", 
                              ['Support Vector Machine', 'K-Nearest Neighbors', 'Naive Bayes'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_option == 'Support Vector Machine':
        model = SVC(C=10, kernel='linear')
    elif model_option == 'K-Nearest Neighbors':
        model = KNeighborsClassifier(n_neighbors=7, weights='uniform')
    else:  # Naive Bayes
        model = MultinomialNB(alpha=0.1)
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.success(f"Model trained! Accuracy: {acc:.2f}")
    
    # Show classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())
    
    # Show confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    st.pyplot(fig)
    
    # Save model and vectorizer
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    # 6. Prediction on New Text
    st.header("6. Predict New Text")
    new_text = st.text_area("Enter text to analyze sentiment:")
    
    if st.button("Predict"):
        # Load model and vectorizer
        loaded_model = joblib.load("sentiment_model.pkl")
        loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        
        # Clean and vectorize the new text
        cleaned_text = clean_text(new_text)
        text_vector = loaded_vectorizer.transform([cleaned_text])
        
        # Make prediction
        pred = loaded_model.predict(text_vector)
        pred_proba = loaded_model.predict_proba(text_vector)
        
        # Display results
        st.success(f"Predicted sentiment: {pred[0]}")
        
        # Show prediction probabilities
        st.subheader("Prediction Probabilities")
        proba_df = pd.DataFrame({
            'Class': loaded_model.classes_,
            'Probability': pred_proba[0]
        })
        st.bar_chart(proba_df.set_index('Class'))