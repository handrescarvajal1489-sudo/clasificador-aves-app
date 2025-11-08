import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd

# ==========================
# CONFIGURACIÓN GENERAL
# ==========================
st.set_page_config(
    page_title="Clasificador de Aves 🦜",
    page_icon="🦜",
    layout="wide",
)

# ==========================
# ESTILO PERSONALIZADO
# ==========================
st.markdown("""
<style>
/* Fondo general: vino + amarillo pero más suave y claro */
.stApp {
    background: linear-gradient(
        180deg,
        #8b2b2b 0%,
        #8b2b2b 40%,
        #ffe766 40%,
        #fff9c4 100%
    );
}

/* Contenedor principal más transparente */
.block-container {
    background-color: rgba(0, 0, 0, 0.04);
    padding: 2rem 2rem 3rem 2rem;
    border-radius: 16px;
    margin-top: 1rem;
}

/* Títulos en el contenido principal */
h1, h2, h3, h4 {
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar más llamativa pero suave */
[data-testid="stSidebar"] {
    background-color: #181b26;
    border-right: 2px solid #8b2b2b;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFD700;
    font-family: 'Segoe UI', sans-serif;
}

/* Texto del sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #f5f5f5;
    font-size: 14px;
}

/* Botones */
div.stButton > button:first-child {
    background-color: #8b2b2b;
    color: #FFD700;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    padding: 0.5em 1.1em;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
div.stButton > button:first-child:hover {
    background-color: #6A0000;
    color: #ffffff;
    transform: scale(1.03);
}

/* Tablas */
.dataframe {
    background-color: #ffffff;
    border-radius: 10px;
    overflow: hidden;
}

/* Caja de resultado principal */
.result-box {
    background-color: rgba(0, 0, 0, 0.35);
    border: 2px solid #FFD700;
    border-radius: 15px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #ffffff;
}

/* Marca de agua con tu nombre y universidad */
.watermark {
    position: fixed;
    right: 20px;
    bottom: 10px;
    font-size: 13px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.45);
    z-index: 9999;
    pointer-events: none;
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Marca de agua
st.markdown(
    "<div class='watermark'>Hollman Carvajal - Universidad Cooperativa</div>",
    unsafe_allow_html=True
)

# ==========================
# TÍTULO PRINCIPAL
# ==========================
st.title("🦜 Clasificador de Aves")
st.markdown(
    "Sube una imagen de un ave y deja que el modelo de *Deep Learning* "
    "prediga la especie."
)

# ==========================
# SELECCIÓN DE MODELO
# ==========================
model_options = {
    "VGG16": os.path.join("modelos", "dataset_vgg16.keras"),
    "NASNetMobile": os.path.join("modelos", "dataset_nasnetmobile.keras"),
}

st.sidebar.title("⚙️ Configuración del modelo")
model_choice = st.sidebar.selectbox(
    "Selecciona el modelo a utilizar:",
    list(model_options.keys())
)

MODEL_PATH = model_options[model_choice]
CLASS_NAMES_PATH = "class_names.txt"

# Tamaño de imagen según modelo
if model_choice == "VGG16":
    IMG_SIZE = (224, 224)
else:
    IMG_SIZE = (224, 224)

# ==========================
# FUNCIONES AUXILIARES
# ==========================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_names(num_classes: int):
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) >= num_classes:
            return names[:num_classes]
        else:
            names += [f"Clase {i}" for i in range(len(names), num_classes)]
            return names
    return [f"Clase {i}" for i in range(num_classes)]

def preprocess_image(img: Image.Image, target_size=IMG_SIZE):
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def predict_image(model, img_array, class_names, top_k=3):
    preds = model.predict(img_array)[0]
    indices = np.argsort(preds)[::-1][:top_k]
    return [{"class_name": class_names[i], "prob": float(preds[i])} for i in indices]

# ==========================
# CARGA DE MODELO
# ==========================
try:
    model = load_model(MODEL_PATH)
    output_shape = model.output_shape
    num_classes = output_shape[-1]
    class_names = load_class_names(num_classes)

    st.sidebar.success(f"Modelo '{model_choice}' cargado correctamente ✅")
    st.sidebar.metric("Nº de clases", num_classes)
    st.sidebar.metric("Modelo activo", model_choice)

    # CONTEXTO DEL PROYECTO EN LA BARRA LATERAL
    st.sidebar.markdown("### ℹ️ Sobre el proyecto")
    st.sidebar.markdown(f"""
Este proyecto implementa un **clasificador de aves** usando redes neuronales
convolucionales (CNN).

- 🧠 Arquitecturas: `VGG16` y `NASNetMobile`  
- 🐦 Especies que puede reconocer: **{num_classes}**  
- 🧪 Uso: práctica y demostración de modelos de **Deep Learning**.  
- 🎓 Ideal para proyectos académicos, posters y presentaciones.
""")

    st.sidebar.markdown("### ✅ Consejos para mejores resultados")
    st.sidebar.markdown("""
- Procura que el ave esté **centrada** en la foto.  
- Evita imágenes muy oscuras o borrosas.  
- Prueba varias fotos de la misma especie y observa la estabilidad de la predicción.  
- Usa siempre formatos **JPG** o **PNG**.
""")

except Exception as e:
    st.error(f"Error al cargar modelo: {e}")
    st.stop()

# ==========================
# INTERFAZ PRINCIPAL
# ==========================
st.subheader("📸 Sube tu imagen")
uploaded_file = st.file_uploader(
    "Sube una imagen de un ave (JPG o PNG)",
    type=["jpg", "jpeg", "png"]
)

st.markdown(
    "Una vez cargues la imagen, pulsa **Clasificar ave** para ver las "
    "3 especies más probables."
)

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.subheader("Imagen subida")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Predicción")
        if st.button("🔍 Clasificar ave"):
            with st.spinner("Analizando imagen..."):
                img_array = preprocess_image(image)
                results = predict_image(model, img_array, class_names, top_k=3)

            if results:
                top_pred = results[0]
                name = top_pred["class_name"]
                prob = top_pred["prob"] * 100

                st.markdown(f"""
                <div class='result-box'>
                    <h3>🏆 Especie más probable</h3>
                    <h2>{name}</h2>
                    <p><b>Confianza:</b> {prob:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                df = pd.DataFrame({
                    "Especie": [r["class_name"] for r in results],
                    "Probabilidad (%)": [round(r["prob"]*100, 2) for r in results]
                })

                st.markdown("### 📊 Tabla de predicciones (Top 3)")
                st.dataframe(df, use_container_width=True)

                st.markdown("### 📈 Distribución de probabilidades")
                st.bar_chart(df.set_index("Especie"))
            else:
                st.warning("No se obtuvieron predicciones, revisa la imagen.")
else:
    st.info("👆 Sube una imagen para comenzar la clasificación.")



