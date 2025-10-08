import os
os.system('pip install streamlit pandas lightgbm scikit-learn joblib')

import streamlit as st
import pandas as pd
import pickle
import lightgbm
import numpy as np
from sklearn.preprocessing import RobustScaler

# ===============================================================
# CONFIGURACIÓN DE LA APP
# ===============================================================
st.set_page_config(page_title="Predicción Automática", page_icon="🤖", layout="wide")

st.markdown(
    """
    <h1 style="background-color:#004aad; color:white; padding:10px; text-align:center; border-radius:10px;">
        Sistema de Predicción Automática
    </h1>
    """,
    unsafe_allow_html=True
)

st.info("Introduce los valores de entrada y obtén una predicción automática con los modelos entrenados (clasificación y regresión).")
st.caption("💡 Si deseas realizar otra predicción, simplemente cambia los valores y presiona nuevamente el botón correspondiente.")

# ===============================================================
# CARGA DE MODELOS Y SCALER
# ===============================================================

@st.cache_resource
def load_classification_model():
    with open("modelo_clasificacion.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_regression_model():
    with open("modelo_regresion.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler_y():
    """Carga el scaler que se usó para escalar la variable objetivo (y)."""
    try:
        with open("scaler_y.pkl", "rb") as f:
            scaler = pickle.load(f)
        st.success("✅ Scaler de salida cargado correctamente.")
        return scaler
    except Exception as e:
        st.warning(f"⚠️ No se encontró scaler_y.pkl o no se pudo cargar: {e}")
        return None

try:
    clas_model = load_classification_model()
    reg_model = load_regression_model()
    scaler_y = load_scaler_y()
    st.success("✅ Modelos cargados correctamente.")
except Exception as e:
        st.error(f"❌ Error al realizar la predicción de regresión: {e}")

