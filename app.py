import os
import streamlit as st
import pickle

MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

def load_artifacts():
    """Attempt to load model and vectorizer; show friendly Streamlit errors on failure."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        st.error(
            f"Model files not found. Expected '{MODEL_PATH}' and '{VECT_PATH}' in the app root."
        )
        st.info(
            "Push the model files to your repo or use external storage and download at startup."
        )
        return None, None

    try:
        with open(MODEL_PATH, "rb") as mf:
            model = pickle.load(mf)
        with open(VECT_PATH, "rb") as vf:
            vectorizer = pickle.load(vf)
        return model, vectorizer
    except Exception as e:
        st.error("Failed to load model or vectorizer — check deployment logs for details.")
        st.exception(e)
        return None, None


model, vectorizer = load_artifacts()

st.title("📰 Fake News Detection System")

news_text = st.text_area("Enter News Text")

if st.button("Check News"):
    if model is None or vectorizer is None:
        st.warning("Model not loaded — cannot perform prediction.")
    elif news_text.strip() == "":
        st.warning("Please enter news text")
    else:
        data = vectorizer.transform([news_text])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
 