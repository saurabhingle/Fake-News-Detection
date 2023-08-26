# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:31:38 2023

@author: saura
"""

import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# Load the pipeline model
with open('decision_tree_pipeline.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Set page title and favicon
st.set_page_config(
    page_title="News Classifier App",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for styling
st.markdown(
    f"""
    <style>
        .stButton button {{
            background-color: #3366cc;
            color: white;
            border-color: #3366cc;
            border-radius: 5px;
            padding: 8px 20px;
            font-size: 16px;
        }}
        .stTextInput textarea {{
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }}
        .real-prediction {{
            color: green;
        }}
        .fake-prediction {{
            color: red;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Header and title
st.title('News Classifier')
st.markdown("---")

# Text input
text = st.text_area('Enter News Text:', height=200)

# Classify button
if st.button('Classify'):
    prediction = pipeline.predict([text])[0]
    prediction_probabilities = pipeline.predict_proba([text])[0]

    real_news_probability = prediction_probabilities[1] * 100
    fake_news_probability = prediction_probabilities[0] * 100

    prediction_text = "Real" if prediction == 1 else "Fake"
    prediction_class = "real-prediction" if prediction == 1 else "fake-prediction"

    st.markdown("---")
    st.subheader('Prediction:')
    st.markdown(f'<p class="{prediction_class}">{prediction_text}</p>', unsafe_allow_html=True)

    st.subheader('Prediction Probabilities:')
    st.write(f"Real News Probability: {real_news_probability:.2f}%")
    st.write(f"Fake News Probability: {fake_news_probability:.2f}%")

    # Display word cloud for real news
    if prediction == 1:
        st.subheader('Word Cloud for Real News')
        wordcloud_real = WordCloud(width=800, height=800, background_color='white').generate(text)
        img_buf = BytesIO()
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud_real)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(img_buf, format='png')
        st.image(img_buf.getvalue(), use_column_width=True)

    # Display word cloud for fake news
    else:
        st.subheader('Word Cloud for Fake News')
        wordcloud_fake = WordCloud(width=800, height=800, background_color='white').generate(text)
        img_buf = BytesIO()
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud_fake)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(img_buf, format='png')
        st.image(img_buf.getvalue(), use_column_width=True)






