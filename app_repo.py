import os
os.system('pip install streamlit pandas lightgbm scikit-learn joblib')

import streamlit as st
import pandas as pd
import pickle
import lightgbm
import numpy as np
from sklearn.preprocessing import RobustScaler
from PIL import Image, ImageOps
import base64
import io

###################################################################################################
# CONFIGURACIÓN DE LA APP
st.set_page_config(page_title="Predicción Automática", page_icon="assets/logo.png", layout="wide")

###################################################################################################
# CONFIGURACIÓN PARA EL USO DE IMÁGENES

ASSETS_DIR = "assets"  # Carpeta local con las imágenes

# Se carga la imagen al cache (memoria temporal de rápido acceso)
@st.cache_resource
def load_image(path):
    return Image.open(path)

# Rutas de las imágenes
logo_path = os.path.join(ASSETS_DIR, "logo.png")
icon_class_path = os.path.join(ASSETS_DIR, "icon_class.png")
icon_reg_path = os.path.join(ASSETS_DIR, "icon_reg.png")

# Validación de existencia
logo_img = load_image(logo_path) if os.path.exists(logo_path) else None
icon_class = load_image(icon_class_path) if os.path.exists(icon_class_path) else None
icon_reg = load_image(icon_reg_path) if os.path.exists(icon_reg_path) else None

###################################################################################################
# ENCABEZADO UNIFICADO CON LOGO Y TÍTULO PRINCIPAL

if logo_img is not None:
    col_logo, col_title = st.columns([1, 9])
    with col_logo:
        st.image(logo_img, width=120, use_container_width=False)
    with col_title:
        st.markdown(
            """
            <h1 style="background-color:#004aad; color:white; padding:10px; 
                       text-align:center; border-radius:10px; margin-bottom:0;">
                Sistema de Predicción Automática
            </h1>
            """,
            unsafe_allow_html=True
        )
else:
    st.markdown(
        """
        <h1 style="background-color:#004aad; color:white; padding:10px; 
                   text-align:center; border-radius:10px; margin-bottom:0;">
            Sistema de Predicción Automática
        </h1>
        """,
        unsafe_allow_html=True
    )

# Mensajes introductorios
st.info("Introduce los valores de entrada y obtén una predicción automática con los modelos entrenados (clasificación y regresión).")
st.caption("Si deseas realizar otra predicción, simplemente cambia los valores y presiona nuevamente el botón correspondiente.")

##############################################################################################
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

try:
    clas_model = load_classification_model()
    reg_model = load_regression_model()
    st.success("Modelos cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")

################################################################################################
# CREACIÓN INTERNA DEL SCALER Y FUNCIÓN DE INVERSIÓN

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
        if abs(value) < 10:
            value_array = np.array(value).reshape(-1, 1)
            inv_value = internal_scaler.inverse_transform(value_array)[0][0]
            return inv_value
        else:
            return value
    except Exception:
        return value

#################################################################################################
# DEFINICIÓN DE VARIABLES

feature_specs = [
    {"name": "Edad", "type": "int", "unit": "años", "description": "Edad del individuo"},
    {"name": "Ritmo cardiáco", "type": "float", "unit": "lpm", "description": "Latidos por minuto del individuo"},
    {"name": "Duración", "type": "float", "unit": "minutos", "description": "Duración de la actividad física"},
    {"name": "Peso", "type": "float", "unit": "kg", "description": "Peso corporal del individuo"}  # Solo usada para la regresión
]

################################################################################################
# FORMULARIO DE ENTRADA

st.subheader("🧮 Formulario de datos de entrada")
user_input = {}

with st.form("formulario_prediccion"):
    cols = st.columns(2)
    for i, spec in enumerate(feature_specs):
        with cols[i % 2]:
            label = f"{spec['name']} ({spec['unit']})"
            tooltip = spec['description']
            user_input[spec["name"]] = st.number_input(
                label,
                value=0.0 if spec["type"] == "float" else 0,
                key=spec["name"],
                help=tooltip
            )
    submitted = st.form_submit_button("🔍 Obtener Predicciones")

################################################################################################
# PROCESO DE PREDICCIÓN

if submitted:
    input_df = pd.DataFrame([user_input])
    st.write("**Datos ingresados:**")
    st.dataframe(input_df)

    # Mapeo de nombres al inglés (para compatibilidad con el modelo)
    rename_map = {
        "Edad": "Age",
        "Ritmo cardiáco": "Heart_Rate",
        "Duración": "Duration",
        "Peso": "Weight"
    }
    input_df.rename(columns=rename_map, inplace=True)

    # ---------------- Clasificación ----------------
    try:
        input_class = input_df[["Age", "Heart_Rate", "Duration"]]
        pred_class = clas_model.predict(input_class)[0]

        if hasattr(clas_model, "predict_proba"):
            prob = clas_model.predict_proba(input_class)[0][1]

            col_icon, col_text = st.columns([0.6, 9])
            with col_icon:
                if icon_class is not None:
                    _icon = icon_class.copy()
                    _icon.thumbnail((48, 48))
                    st.image(_icon, width=48, use_container_width=False)
            with col_text:
                st.success(f"🔹 Predicción (Clasificación): **{pred_class}** — Probabilidad positiva: **{prob:.2%}**")

                if prob > 0.7:
                    st.caption("Interpretación: Alta probabilidad de quemar más de 80 calorías durante la sesión.")
                elif prob > 0.4:
                    st.caption("Interpretación: Probabilidad moderada de quemar más de 80 calorías durante la sesión.")
                else:
                    st.caption("Interpretación: Baja probabilidad de quemar más de 80 calorías durante la sesión.")
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

        final_value = safe_inverse_transform(pred_reg_value)

        col_icon, col_text = st.columns([0.6, 9])
        with col_icon:
            if icon_reg is not None:
                _icon = icon_reg.copy()
                _icon.thumbnail((48, 48))
                st.image(_icon, width=48, use_container_width=False)
        with col_text:
            st.info(f"🔸 Predicción (Regresión): **{final_value:.3f}**")
            st.caption("** Este valor representa una estimación de las calorías quemadas durante la sesión, basadas en los datos del usuario.")
            st.caption("Si deseas otra predicción, modifica los valores y presiona el botón nuevamente.")    

    except Exception as e:
        st.error(f"❌ Error al realizar la predicción de regresión: {e}")













