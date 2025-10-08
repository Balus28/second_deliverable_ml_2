import os
os.system('pip install streamlit pandas lightgbm scikit-learn joblib')

import streamlit as st
import pandas as pd
import pickle
import lightgbm
import numpy as np

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
    st.success("✅ Modelos cargados correctamente.")
except Exception as e:
    st.error(f"⚠️ Error al cargar los modelos: {e}")

# ===============================================================
# DEFINICIÓN DE VARIABLES
# ===============================================================
feature_specs = [
    {"name": "Age", "type": "int"},
    {"name": "Heart_Rate", "type": "float"},
    {"name": "Duration", "type": "float"},
    {"name": "Weight", "type": "float"}  # Solo usada para la regresión
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
        input_class = input_df[["Age", "Heart_Rate", "Duration"]]  # subset para clasificación
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
        # Reordenar columnas según las esperadas por el modelo
        if hasattr(reg_model, "feature_names_in_"):
            expected_features = reg_model.feature_names_in_
            input_df = input_df.reindex(columns=expected_features, fill_value=0)
            st.caption(f"📋 Columnas reordenadas según el modelo: {list(expected_features)}")
        else:
            st.warning("⚠️ El modelo no contiene metadatos de columnas. Se usará el orden actual.")

        # Mostrar al usuario los datos que se enviarán al modelo
        st.write("**Datos enviados al modelo de regresión:**")
        st.dataframe(input_df)

        # Realizar predicción
        pred_reg = reg_model.predict(input_df)

        # Si el resultado es una lista o array, tomar el primer valor
        pred_reg_value = pred_reg[0] if isinstance(pred_reg, (list, np.ndarray)) else pred_reg

        # Mostrar resultado al usuario
        st.info(f"🔸 Predicción (Regresión): **{pred_reg_value:.3f}**")
        st.caption("Si deseas otra predicción, modifica los valores y presiona el botón nuevamente.")    

    except Exception as e:
        st.error(f"❌ Error al realizar la predicción de regresión: {e}")
