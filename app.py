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
    doc = uncapitalize(doc)
    doc = re.sub(r"[^a-zA-Z\s]", "", doc, re.I | re.A)
    doc = doc.strip()
    tokens = doc.split()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Streamlit App
st.title('Multi-Aspect Text Classification with Naive Bayes')
st.write("Upload a CSV file containing a 'sentence' column and one or more aspect columns.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader('Raw Data Preview')
    st.write(df.head())

    # Validate 'sentence' column
    if 'sentence' not in df.columns:
        st.error("CSV must contain a 'sentence' column.")
        st.stop()

    # Determine possible label columns (all except 'sentence')
    label_cols = [col for col in df.columns if col != 'sentence']
    if not label_cols:
        st.error("No aspect columns found. CSV must have at least one label column besides 'sentence'.")
        st.stop()

    # Let user select which column to use as label
    selected_label = st.sidebar.selectbox("Select aspect column to classify (label)", label_cols)
    st.write(f"**Using label column:** {selected_label}")
    st.write(df[['sentence', selected_label]].head())

    # Preprocessing
    st.subheader('Preprocessing')
    df['clean_sentence'] = df['sentence'].astype(str).apply(normalize_document)
    st.write(df[['sentence', 'clean_sentence', selected_label]].head())

    # Feature Extraction
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['clean_sentence'])
    y = df[selected_label]

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

    # Predict new sentence
    st.subheader('Predict New Sentence')
    user_input = st.text_input('Enter a sentence to classify')
    if st.button('Predict') and user_input:
        clean_input = normalize_document(user_input)
        vect_input = tfidf.transform([clean_input])
        pred = model.predict(vect_input)[0]
        st.write(f'**Predicted {selected_label}:** {pred}')
else:
    st.info('Awaiting CSV file to be uploaded.')
