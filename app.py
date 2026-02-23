import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

st.set_page_config(page_title="Bike Price Prediction + SHAP", layout="wide")
st.title("🏍 Used Bike Price Prediction with Explainable AI")

def force_numeric_df(df, cols):
    df = df.copy()
    df = df.reindex(columns=cols, fill_value=0)

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.astype(np.float64)

    return df


@st.cache_resource
def load_artifacts():
    model = XGBRegressor()
    model.load_model("bike_price_xgb.json")

    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

    with open("shap_background.pkl", "rb") as f:
        background = pickle.load(f)

    if isinstance(background, np.ndarray):
        background = pd.DataFrame(background, columns=model_columns)

    background = force_numeric_df(background, model_columns)

    return model, model_columns, background


model, model_columns, background = load_artifacts()


def extract_categories(prefix):
    return sorted([
        col.replace(prefix, "")
        for col in model_columns
        if col.startswith(prefix)
    ])

brand_options = extract_categories("Brand_")
bike_type_options = extract_categories("Bike Type_")
district_options = extract_categories("District_")

@st.cache_resource
def build_explainer(_model, _background, _cols):

    def predict_fn(data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=_cols)
        else:
            data = data.copy()

        data = force_numeric_df(data, _cols)
        return _model.predict(data)

    masker = shap.maskers.Independent(_background)
    return shap.Explainer(predict_fn, masker)


explainer = build_explainer(model, background, model_columns)


st.sidebar.header("Enter Bike Details")

year = st.sidebar.number_input("Year", 1980, 2026, 2018)
mileage = st.sidebar.number_input("Mileage (km)", 0, 300000, 20000)
capacity = st.sidebar.number_input("Engine Capacity (cc)", 50, 2000, 150)

brand = st.sidebar.selectbox("Brand", brand_options)
bike_type = st.sidebar.selectbox("Bike Type", bike_type_options)
district = st.sidebar.selectbox("District", district_options)

CURRENT_YEAR_FOR_FEATURES = 2024
vehicle_age = max(CURRENT_YEAR_FOR_FEATURES - int(year), 0)

mileage_per_year = mileage / (vehicle_age + 1)
cc_age_interaction = capacity * vehicle_age
cc_mileage_interaction = capacity * mileage

base_features = {
    "Year": year,
    "Mileage": mileage,
    "Capacity": capacity,
    "Vehicle_Age": vehicle_age,
    "Mileage_per_Year": mileage_per_year,
    "CC_Age_Interaction": cc_age_interaction,
    "CC_Mileage_Interaction": cc_mileage_interaction,
}

input_df = pd.DataFrame([base_features])
input_df = force_numeric_df(input_df, model_columns)

brand_col = f"Brand_{brand}"
bike_type_col = f"Bike Type_{bike_type}"
district_col = f"District_{district}"

if brand_col in input_df.columns:
    input_df[brand_col] = 1
if bike_type_col in input_df.columns:
    input_df[bike_type_col] = 1
if district_col in input_df.columns:
    input_df[district_col] = 1


if st.button("Predict Price"):

    pred_log = float(model.predict(input_df)[0])
    pred_price = float(np.expm1(pred_log))

    st.success(f"💰 Predicted Price: Rs {pred_price:,.0f}")

    st.markdown("---")
    st.subheader("🔎 Explainability (SHAP)")

    shap_values = explainer(input_df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Waterfall Plot")
        fig1 = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False, max_display=12)
        st.pyplot(fig1, clear_figure=True)

    with col2:
        st.markdown("### Top Feature Contributions")

        vals = np.abs(shap_values.values[0])
        top_idx = np.argsort(vals)[-12:][::-1]

        top_features = np.array(model_columns)[top_idx]
        top_vals = vals[top_idx]

        fig2 = plt.figure()
        plt.barh(top_features[::-1], top_vals[::-1])
        plt.xlabel("Absolute SHAP Value")
        plt.title("Top 12 Features")
        st.pyplot(fig2, clear_figure=True)