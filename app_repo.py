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
st.caption("Si deseas realizar otra predicción, simplemente cambia los valores y presiona nuevamente el botón correspondiente.")

# ===============================================================
# CARGA DE MODELOS
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

try:
    clas_model = load_classification_model()
    reg_model = load_regression_model()
    st.success("Modelos cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")

# ===============================================================
# CREACIÓN INTERNA DEL SCALER Y FUNCIÓN DE INVERSIÓN
# ===============================================================

# Parámetros base para recrear un RobustScaler inverso
# (puedes ajustarlos si sabes los rangos típicos de tu variable objetivo)
internal_center = 0.0
internal_scale = 1.0

internal_scaler = RobustScaler()
internal_scaler.center_ = np.array([internal_center])
internal_scaler.scale_ = np.array([internal_scale])
internal_scaler.n_features_in_ = 1

def safe_inverse_transform(value):
    """
    Aplica un inverse_transform interno para desescalar la predicción.
    Si el valor parece ya en escala original, lo devuelve tal cual.
    """
    try:
        # Detección simple: si el valor parece "escalado" (pequeño rango)
        if abs(value) < 10:
            value_array = np.array(value).reshape(-1, 1)
            inv_value = internal_scaler.inverse_transform(value_array)[0][0]
            return inv_value
        else:
            return value
    except Exception:
        return value

# ===============================================================
# DEFINICIÓN DE VARIABLES
# ===============================================================
feature_specs = [
    {"name": "Age", "type": "int", "unit":"años", "description":"Edad del individuo"},
    {"name": "Heart_Rate", "type": "float", "unit":"lpm", "description":"Latidos por minutos del individuo"},
    {"name": "Duration", "type": "float", "unit":"minutos"m "description":"Duración de la actividad física"},
    {"name": "Weight", "type": "float", "unit":"kg", "description":"Peso corporal del individuo"}  # Solo usada para la regresión
]

# ===============================================================
# FORMULARIO DE ENTRADA
# ===============================================================
st.subheader("🧮 Formulario de datos de entrada")
user_input = {}

with st.form("formulario_prediccion"):
    cols = st.columns(2)
    for i, spec in enumerate(feature_specs):
        with cols[i % 2]:
            user_input[spec["name"]] = st.number_input(
                f"{spec['name']}",
                value=0.0 if spec["type"] == "float" else 0,
                key=spec["name"]
            )
    submitted = st.form_submit_button("🔍 Obtener Predicciones")

# ===============================================================
# PROCESO DE PREDICCIÓN
# ===============================================================
if submitted:
    input_df = pd.DataFrame([user_input])
    st.write("**Datos ingresados:**")
    st.dataframe(input_df)

    # ---------------- Clasificación ----------------
    try:
        input_class = input_df[["Age", "Heart_Rate", "Duration"]]
        pred_class = clas_model.predict(input_class)[0]

        if hasattr(clas_model, "predict_proba"):
            prob = clas_model.predict_proba(input_class)[0][1]
            st.success(f"🔹 Predicción (Clasificación): **{pred_class}** — Probabilidad positiva: **{prob:.2%}**")
        else:
            st.success(f"🔹 Predicción (Clasificación): **{pred_class}**")

    except Exception as e:
        st.error(f"❌ Error en la predicción de clasificación: {e}")

    # ---------------- Regresión ----------------
    try:
        if hasattr(reg_model, "feature_names_in_"):
            expected_features = reg_model.feature_names_in_
            input_df = input_df.reindex(columns=expected_features, fill_value=0)
            st.caption(f"📋 Columnas reordenadas según el modelo: {list(expected_features)}")

        st.write("**Datos enviados al modelo de regresión:**")
        st.dataframe(input_df)

        pred_reg = reg_model.predict(input_df)
        pred_reg_value = pred_reg[0] if isinstance(pred_reg, (list, np.ndarray)) else pred_reg

        # Aplicar inverse scaling manual si el valor parece escalado
        final_value = safe_inverse_transform(pred_reg_value)

        st.info(f"🔸 Predicción (Regresión): **{final_value:.3f}**")
        st.caption("Si deseas otra predicción, modifica los valores y presiona el botón nuevamente.")    

    except Exception as e:
        st.error(f"❌ Error al realizar la predicción de regresión: {e}")

