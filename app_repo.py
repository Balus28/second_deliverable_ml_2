import os
os.system('pip install streamlit pandas joblib lightgbm scikit-learn')
import streamlit as st
import pandas as pd
import pickle

################################################################
# CONFIGURACIÓN DE LA APP
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

################################################################
# CARGA DE MODELOS

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

clas_model = load_classification_model()
regr_model = load_regression_model()

################################################################
# DEFINICIÓN DE VARIABLES
feature_specs = [
    {"name": "Age", "type": "int"},
    {"name": "Heart_Rate", "type": "float"},
    {"name": "Duration", "type": "float"},
    {"name": "Weight", "type": "float"}
]

################################################################
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
for spec in feature_specs:
    if spec["name"] not in st.session_state:
        st.session_state[spec["name"]] = 0.0 if spec["type"] == "float" else 0

################################################################
# FORMULARIO DE ENTRADA

st.subheader("Formulario de datos de entrada")
user_input = {}

with st.form("formulario_prediccion"):
    cols = st.columns(2)
    for i, spec in enumerate(feature_specs):
        with cols[i % 2]:
            if spec["type"] in ["int", "float"]:
                user_input[spec["name"]] = st.number_input(
                    f"{spec['name']}",
                    value=st.session_state[spec["name"]],
                    key=spec["name"]
                )

    submitted = st.form_submit_button("Obtener Predicciones")

# 🔄 BOTÓN DE REINICIO (fuera del formulario)
if st.button("🔄 Reiniciar valores"):
    for spec in feature_specs:
        st.session_state[spec["name"]] = 0.0 if spec["type"] == "float" else 0
    st.rerun()

################################################################
# PROCESO DE PREDICCIÓN

if submitted:
    input_df = pd.DataFrame([user_input])
    st.write("**Datos ingresados:**")
    st.dataframe(input_df)

    input_df_class = input_df[["Age", "Heart_Rate", "Duration"]]

    try:
        ######################################
        # SECCIÓN 1: CLASIFICACIÓN
        ######################################
        st.markdown("### 🔹 Predicción de Clasificación")
        class_pred = clas_model.predict(input_df_class)[0]
        class_prob = clas_model.predict_proba(input_df_class)[0][1]

        st.success(f"Predicción (Clase): **{class_pred}**")
        st.write(f"Probabilidad estimada de clase positiva: **{class_prob:.2%}**")

        ######################################
        # SECCIÓN 2: REGRESIÓN
        ######################################
        st.markdown("### 🔹 Predicción de Regresión")
        regr_pred = regr_model.predict(input_df)[0]

        st.success(f"Valor predicho (Regresión): **{regr_pred:.2f}**")

    except Exception as e:
        st.error(f"Ocurrió un error al realizar la predicción: {e}")
