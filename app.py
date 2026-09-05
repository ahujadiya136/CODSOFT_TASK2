import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬", layout="centered")

st.title("🎬 Movie Rating Prediction App")
st.write("Enter the details of a movie to predict its rating using Machine Learning.")

# Load the saved trained model and target encoder
@st.cache_resource
def load_assets():
    model = joblib.load('movie_rating_model.pkl')
    encoder = joblib.load('target_encoder.pkl')
    return model, encoder

try:
    model, encoder = load_assets()
    
    # User Inputs
    genre = st.text_input("Genre", value="Action")
    director = st.text_input("Director Name", value="Christopher Nolan")
    actor = st.text_input("Main Actor Name", value="Leonardo DiCaprio")

    # Prediction Button
    if st.button("Predict Rating"):
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Genre': genre,
            'Director': director,
            'Actor 1': actor
        }])
        
        # Encode features and predict
        input_encoded = encoder.transform(input_data)
        prediction = model.predict(input_encoded)[0]
        
        # Display Result
        st.success(f"⭐ Estimated Rating: **{prediction:.1f} / 10**")

except FileNotFoundError:
    st.error("Model files not found! Please run your training script to generate `movie_rating_model.pkl` and `target_encoder.pkl` first.")