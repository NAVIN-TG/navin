import os
import streamlit as st
import pickle

# -------------------------------
# Load model and vectorizer safely
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()

# -------------------------------
# UI
# -------------------------------

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")

news_text = st.text_area(
    "Enter News Text",
    height=200,
    placeholder="Paste news content here..."
)

if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("⚠ Please enter some news text")
    else:
        data = vectorizer.transform([news_text])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
