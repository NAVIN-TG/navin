import os
import streamlit as st
import pickle

MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

MODEL_URL = os.environ.get("MODEL_URL")
VECT_URL = os.environ.get("VECT_URL")


def _download_file(url, dest_path):
    try:
        import requests
    except Exception:
        return False, "requests not installed"

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)


@st.cache_resource
def load_artifacts():
    # Try to download if URLs provided and files missing
    if not os.path.exists(MODEL_PATH) and MODEL_URL:
        ok, err = _download_file(MODEL_URL, MODEL_PATH)
        if not ok:
            st.warning(f"Could not download model: {err}")

    if not os.path.exists(VECT_PATH) and VECT_URL:
        ok, err = _download_file(VECT_URL, VECT_PATH)
        if not ok:
            st.warning(f"Could not download vectorizer: {err}")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        return None, None

    try:
        with open(MODEL_PATH, "rb") as mf:
            model = pickle.load(mf)
        with open(VECT_PATH, "rb") as vf:
            vectorizer = pickle.load(vf)
        return model, vectorizer
    except Exception as e:
        st.error("Failed to load model or vectorizer; see logs for details.")
        st.exception(e)
        return None, None


def main():
    st.title("📰 Fake News Detection System")

    model, vectorizer = load_artifacts()

    if model is None or vectorizer is None:
        st.info("Model not loaded. Provide `MODEL_URL` and `VECT_URL` or place model files in the repo.")

    news_text = st.text_area("Enter News Text", key="news_text", height=200)

    if st.button("Check News"):
        if model is None or vectorizer is None:
            st.warning("Model unavailable — cannot perform prediction.")
            return

        if not news_text or news_text.strip() == "":
            st.warning("Please enter news text")
            return

        try:
            data = vectorizer.transform([news_text])
            prediction = model.predict(data)
            if int(prediction[0]) == 1:
                st.success("✅ REAL NEWS")
            else:
                st.error("❌ FAKE NEWS")
        except Exception as e:
            st.error("Prediction failed — check logs for details.")
            st.exception(e)


if __name__ == "__main__":
    main()
import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection System")

news_text = st.text_area("Enter News Text")

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter news text")
    else:
        data = vectorizer.transform([news_text])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
 