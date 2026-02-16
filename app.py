import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="Letterboxd Sentiment Detective", layout="centered")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

classifier = load_model()

st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🎬 Letterboxd Sentiment Detective</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)

review = st.text_area("Paste your Letterboxd review here:", height=200)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(review)[0]
            label = result["label"]
            score = result["score"]

            stars = int(label.split()[0])

            if stars >= 4:
                sentiment = "Positive 😊"
            elif stars == 3:
                sentiment = "Neutral 😐"
            else:
                sentiment = "Negative 😠"

            st.success(f"Sentiment: {sentiment}")
            st.info(f"Model Confidence: {round(score * 100, 2)}%")
