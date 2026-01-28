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
 