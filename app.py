import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer
import os

# Set page config
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea > div > div > textarea {
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .stButton > button {
        width: 100%;
        font-size: 1.2rem;
        background-color: #4CAF50;
        color: white;
        padding: 0.8rem 1rem;
        border: none;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        text-align: center;
        font-size: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .review-example {
        font-style: italic;
        color: #666;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Movie Review Sentiment Analysis")
st.markdown("Enter your movie review to analyze its sentiment.")
st.markdown("""
    <div class="review-example">
    Example_1: "This film was an absolute masterpiece. The storytelling was emotionally gripping, 
    the performances were outstanding, and the cinematography was breathtaking."
    </div>
    <div class="review-example">
    Example_2: "The film was poorly paced, with weak performances and a confusing plot. It failed to keep my interest, and I wouldn't recommend it."
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Load the saved model and tokenizer
    model_path = "sentiment_model"
    tokenizer_path = "distilbert-tokenizer"
    
    if not os.path.exists(model_path):
        st.error("Model not found! Please make sure the sentiment_model directory exists.")
        st.stop()
    
    if not os.path.exists(tokenizer_path):
        st.error("Tokenizer not found! Please make sure the distilbert-tokenizer directory exists.")
        st.stop()
    
    try:
        # Load the tokenizer from local directory
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        # Load the TensorFlow model
        model = tf.saved_model.load(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def predict_sentiment(text, model, tokenizer):
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    
    # Get prediction
    outputs = model(inputs)
    # Extract logits from the output dictionary
    logits = outputs['logits']
    predictions = tf.nn.softmax(logits, axis=1).numpy()
    
    # Get the predicted class and confidence
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    sentiment_map = {0: "Negative", 1: "Positive"}
    
    return sentiment_map[predicted_class], confidence

# Load the model
model, tokenizer = load_model()

# Text input with larger area for reviews
user_input = st.text_area(
    "Enter your movie review here:",
    height=200,
    placeholder="Type your review here... (e.g., 'This film was an absolute masterpiece...')"
)

# Submit button
if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing your review..."):
            # Get prediction
            sentiment, confidence = predict_sentiment(user_input, model, tokenizer)
            
            # Display result with appropriate color and confidence
            color = "#4CAF50" if sentiment == "Positive" else "#f44336"
            st.markdown(f"""
                <div class="result-box" style="background-color: {color}20; border: 1px solid {color};">
                    Sentiment: <strong>{sentiment}</strong><br>
                    Confidence: <strong>{confidence:.2%}</strong>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a review to analyze.")
