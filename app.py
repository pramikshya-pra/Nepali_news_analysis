import streamlit as st
import joblib
import re


import nltk
from nltk.corpus import stopwords

### ------- Load the model & Vectorizer----------

svm_model = joblib.load("news_svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

## ------- Nepali Stopwords -----------


NEPALI_STOPWORDS = set(stopwords.words('nepali'))
print(NEPALI_STOPWORDS)

## ----------- Cleaning Process --------------

def clean_nepali_text(text):
    text = str(text)

    # remove English letters & digits
    text = re.sub(r"[a-zA-Z0-9]", "", text)

    # keep only Nepali unicode
    text = re.sub(r"[^\u0900-\u097F\s]", "", text)

    tokens = text.split()
    tokens = [w for w in tokens if w not in NEPALI_STOPWORDS]

    return " ".join(tokens)


## ----------- Prediction --------

def predict_news_category(news_text):
    cleaned_text = clean_nepali_text(news_text)
    vector = tfidf.transform([cleaned_text])
    prediction = svm_model.predict(vector)
    return prediction[0]


## ============ Streamlit =====================

st.title(" Nepali News Classification App")
st.write("Paste Nepali news text below and click **Predict**")

news_input = st.text_area(" Enter Nepali News Text")


if st.button("Predict Category"):
    if news_input.strip() == "":
        st.warning("Please enter some Nepali text.")
    else:
        prediction = predict_news_category(news_input)
        st.success(f"Predicted Category: **{prediction}**")



