
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide"
)

st.title("Wine Quality Predictor")
st.write(
    "Legg inn kjemiske verdier for vinen du ønsker"
    "å predikere verdien for ved hjelp av maskinlæring"
)

st.info(
    "perdiksjonen er basert på kjemiske verdier og historiske"
    "vinkvalitets vurderinger"
)
# choosing wine type
wine_type = st.radio(
    "Choose wine type",
    ["Red wine", "White wine"],
    horizontal=True
)

if wine_type == "Red wine":
    model_path = "models/red_model.joblib"
else:
    model_path = "models/white_model.joblib"

model = joblib.load(model_path)

# Input form
with st.form("wine_form"):
    st.subheader("Wine properties")

    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.number_input(
            "Fixed acidity", min_value=0.0,
            value=7.4, step=0.1
        )
        volatile_acidity = st.number_input(
            "Volatile acidity", min_value=0.0,
            value=0.5, step=0.01
        )
        citric_acid = st.number_input(
            "Citric acid", min_value=0.0,
            value=0.3, step=0.01
        )
        residual_sugar = st.number_input(
            "Residual sugar", min_value=0.0,
            value=2.0, step=0.1
        )
        chlorides = st.number_input(
            "Chlorides", min_value=0.0,
            value=0.07, step=0.001,
            format="%.3f"
        )
        free_sulfur_dioxide = st.number_input(
            "Free sulfur dioxide", min_value=0.0,
            value=15.0, step=1.0
        )

    with col2:
        total_sulfur_dioxide = st.number_input(
            "Total sulfur dioxide", min_value=0.0,
            value=46.0, step=1.0
        )
        density = st.number_input(
            "Density", min_value=0.0,
            value=0.996, step=0.0001,
            format="%.4f"
        )
        pH = st.number_input(
            "pH", min_value=0.0,
            value=3.3, step=0.01
        )
        sulphates = st.number_input(
            "Sulphates", min_value=0.0,
            value=0.6, step=0.01
        )
        alcohol = st.number_input(
            "Alcohol (%)", min_value=0.0,
            max_value=100.0, value=10.0, step=0.1
        )

    submitted = st.form_submit_button(
        "Predict wine quality",
        type="primary"
    )

if submitted:
    # Use the same feature names and order as during training
    input_data = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Predikert vinkvalitet")

    st.metric(
    label="Predikert vinkvalitet",
    value= f"{prediction:.2f} /10"
    )

    st.progress(
        min(max(float(prediction) / 10, 0.0), 1.0)
    )


