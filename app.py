import streamlit as st
import pandas as pd
import joblib
import os


st.write("""
# Wine Quality and Type Prediction App
This app predicts the wine quality and it's type 
""")

st.sidebar.header('Wine input parameters')
file_dir = os.path.dirname(__file__)
CLASSIFICATION_MODEL = joblib.load(os.path.join(file_dir, 'LogisticRegression.pkl'))
type_labels = {
    0: 'White',
    1: 'Red'
}


def user_input_features():
    """"
    Handling user input features
    """
    fixed_acidity = st.sidebar.slider('Fixed acidity', 1, 10, 6)
    volatile_acidity = st.sidebar.slider('Volatile acidity', 0.0, 1.0, 0.3)
    citric_acid = st.sidebar.slider('Citric acid', 0.0, 1.0, 0.2)
    residual_sugar = st.sidebar.slider('Residual sugar', 1, 25, 7)
    chlorides = st.sidebar.slider('Chlorides', 0.01, 0.1, 0.02)
    free_sulfur_dioxide = st.sidebar.slider('Free sulfur dioxide', 1, 100, 13)
    total_sulfur_dioxide = st.sidebar.slider('Total sulfur dioxide', 50, 250, 63)
    ph = st.sidebar.slider('pH', 3.0, 4.0, 3.14)
    sulphates = st.sidebar.slider('Sulphates', 0.4, 0.9, 0.49)
    alcohol = st.sidebar.slider('Alcohol', 5.0, 20.0, 6.3)
    density = st.sidebar.slider('Density', 0.800, 1.100, 0.993)

    data = {
        'Fixed acidity': fixed_acidity,
        'Volatile acidity': volatile_acidity,
        'Citric acid ': citric_acid,
        'Residual sugar': residual_sugar,
        'Chlorides': chlorides,
        'Free sulfur dioxide': free_sulfur_dioxide,
        'Total sulfur dioxide': total_sulfur_dioxide,
        'Density': density,
        'pH': ph,
        'Sulphates': sulphates,
        'Alcohol': alcohol,
    }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('Wine Input parameters')
st.write(df)

st.subheader('Wine Type')
st.write(
    type_labels.get(CLASSIFICATION_MODEL.predict(df)[0])
)

