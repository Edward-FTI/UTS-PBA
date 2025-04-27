import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer

# Function to lowercase and stem
stemmer = PorterStemmer()
def uncapitalize(doc):
    return doc.lower()

def normalize_document(doc):
    # Lowercase
    doc = uncapitalize(doc)
    # Remove non-alphabetic characters
    doc = re.sub(r"[^a-zA-Z\s]", "", doc, re.I | re.A)
    # Strip whitespace
    doc = doc.strip()
    # Tokenize by whitespace
    tokens = doc.split()
    # Stem each token
    tokens = [stemmer.stem(word) for word in tokens]
    # Rejoin to string
    return " ".join(tokens)

# Streamlit App
st.title('Text Classification with Naive Bayes')
st.write("Upload a CSV file containing 'sentence' and 'label' columns.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"]);
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader('Raw Data')
    st.write(df.head())

    # Check required columns
    if 'sentence' not in df.columns or 'label' not in df.columns:
        st.error("CSV must contain 'sentence' and 'label' columns.")
    else:
        # Preprocessing
        st.subheader('Preprocessing')
        df['clean_sentence'] = df['sentence'].astype(str).apply(normalize_document)
        st.write(df[['sentence', 'clean_sentence', 'label']].head())

        # Feature Extraction
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['clean_sentence'])
        y = df['label']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        # Model training
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.2f}")

        st.subheader('Classification Report')
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center')
        st.pyplot(fig)

        # Predict new data
        st.subheader('Predict New Sentence')
        user_input = st.text_input('Enter a sentence to classify')
        if st.button('Predict'):
            clean_input = normalize_document(user_input)
            vect_input = vectorizer.transform([clean_input])
            pred = model.predict(vect_input)[0]
            st.write(f'**Predicted Label:** {pred}')
else:
    st.info('Awaiting CSV file to be uploaded.')
