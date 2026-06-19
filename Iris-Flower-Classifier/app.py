import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Load model
model = joblib.load("models/iris_model.joblib")

# Species names
species = ["setosa", "versicolor", "virginica"]

# Title
st.title("🌸 Iris Flower Classifier")

st.write(
    "Enter flower measurements using the sliders below."
)

# Input sliders
sepal_length = st.slider(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.8
)

sepal_width = st.slider(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=4.5,
    value=3.0
)

petal_length = st.slider(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=4.0
)

petal_width = st.slider(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=2.5,
    value=1.2
)

# Prediction button
if st.button("🔮 Predict Species"):

    features = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[0]

    st.success(
        f"Predicted Species: {species[prediction[0]].title()} 🌼"
    )

    st.subheader("Confidence Scores:")

    for i, sp in enumerate(species):
        st.write(f"{sp}: {probabilities[i] * 100:.1f}%")
        st.progress(float(probabilities[i]))

# Sidebar
st.sidebar.header("About")

st.sidebar.info(
    """
    Iris Flower Classification App

    Model: Random Forest Classifier

    Dataset: Iris Dataset

    Features:
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width
    """
)

# Current values
st.sidebar.subheader("Current Input Values")

st.sidebar.write(
    {
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width
    }
)