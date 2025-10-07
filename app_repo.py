import os
os.system('pip install streamlit pandas joblib')
import streamlit as st
import pandas as pd
import joblib

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

st.info("Introduce los valores de entrada y obtén una predicción automática con el modelo entrenado.")

################################################################
# CARGA DEL MODELO

@st.cache_resource
def load_model():
    model = joblib.load("modelo_clasificacion.joblib")  # Cambia el nombre si tu modelo se llama diferente
    return model

model = load_model()

################################################################
# DEFINICIÓN DE VARIABLES

feature_specs = [
    {"name": "Age", "type": "int"},
    {"name": "Heart_Rate", "type": "float"},
    {"name": "Duration", "type": "float"}
]

######################################################################
# FORMULARIO DE ENTRADA

st.subheader("Formulario de datos de entrada")
user_input = {}

with st.form("formulario_prediccion"):
    cols = st.columns(2)
    for i, spec in enumerate(feature_specs):
        with cols[i % 2]:
            if spec["type"] in ["int", "float"]:
                user_input[spec["name"]] = st.number_input(
                    f"{spec['name']}", value=0.0 if spec["type"] == "float" else 0, key=spec["name"]
                )
            elif spec["type"] == "cat":
                user_input[spec["name"]] = st.selectbox(spec["name"], spec["options"], key=spec["name"])

    submitted = st.form_submit_button("Obtener Predicción")

#####################################################################
# PROCESO DE PREDICCIÓN

if submitted:
    input_df = pd.DataFrame([user_input])
    st.write("**Datos ingresados:**")
    st.dataframe(input_df)

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.success(f"Predicción: **{prediction}**")
        st.write(f"Probabilidad estimada de clase positiva: **{probability:.2%}**")
    except Exception as e:
        st.error(f"Ocurrió un error al realizar la predicción: {e}")
