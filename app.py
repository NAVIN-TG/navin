import os
import streamlit as st
import pickle

MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

MODEL_URL = os.environ.get("MODEL_URL")
VECT_URL = os.environ.get("VECT_URL")

def _download_file(url, dest_path):
    """Download a file from a public URL to dest_path. Returns True on success."""
    try:
        import requests
    except Exception:
        st.warning("`requests` not installed; cannot download model files automatically.")
        return False

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        st.warning(f"Download failed for {url}: {e}")
        return False


def load_artifacts():
    """Attempt to load or download model and vectorizer; show friendly Streamlit messages on failure."""
    # If files missing, try downloading from configured URLs
    if not os.path.exists(MODEL_PATH) and MODEL_URL:
        st.info(f"Downloading model from configured URL...")
        _download_file(MODEL_URL, MODEL_PATH)

    if not os.path.exists(VECT_PATH) and VECT_URL:
        st.info(f"Downloading vectorizer from configured URL...")
        _download_file(VECT_URL, VECT_PATH)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        st.error(
            f"Model files not found. Expected '{MODEL_PATH}' and '{VECT_PATH}' in the app root, or set `MODEL_URL` and `VECT_URL` environment variables."
        )
        st.info(
            "You can upload model files to your repository, use Git LFS, or host them at a public URL and set the environment variables."
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
 