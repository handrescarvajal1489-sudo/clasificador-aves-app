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
/* Fondo con degradado tipo selva tropical */
.stApp {
    background: linear-gradient(180deg, #dff6f0 0%, #b5e7a0 50%, #a7d5f2 100%);
    color: #1b3a4b;
}

/* Títulos */
h1, h2, h3, h4 {
    color: #084c61;
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
}

/* Botones */
div.stButton > button:first-child {
    background-color: #27ae60;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    padding: 0.5em 1em;
    transition: all 0.3s ease-in-out;
}
div.stButton > button:first-child:hover {
    background-color: #2ecc71;
    transform: scale(1.05);
}

/* Tablas */
.dataframe {
    border-radius: 10px;
    background-color: #f8fbf8;
    color: #1b3a4b;
    font-size: 15px;
}

/* Cuadro de resultado */
.result-box {
    background-color: rgba(255, 255, 255, 0.75);
    border: 2px solid #2c7da0;
    border-radius: 15px;
    padding: 1em;
    margin-top: 1em;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# TÍTULO PRINCIPAL
# ==========================
st.title("🦜 Clasificador de Aves Tropicales")
st.markdown(
    "Sube una imagen y el modelo reconocerá la especie de ave. "
    "Inspirado en la diversidad de aves de Latinoamérica. 🇨🇴"
)

# ==========================
# DESCRIPCIONES DE AVES
# ==========================
BIRD_DESCRIPTIONS = {
    "Coereba_flaveola": "Pequeña ave tropical conocida como pinchaflor o mielero, de pico curvado y plumaje amarillo intenso.",
    "Icterus_nigrogularis": "El turpial, ave emblemática de vivos tonos naranjas y negros, con canto melodioso.",
    "Oryzoborus_angolensis": "Semillero robusto de pico fuerte, común en zonas húmedas y con vegetación densa.",
}
DEFAULT_DESCRIPTION = "Descripción no disponible aún. Puedes agregar más especies en el código."

# ==========================
# SELECCIÓN DE MODELO
# ==========================
model_options = {
    "VGG16": os.path.join("modelos", "dataset_vgg16.keras"),
    "NASNetMobile": os.path.join("modelos", "dataset_nasnetmobile.keras"),
}

st.sidebar.header("⚙️ Configuración del modelo")
model_choice = st.sidebar.selectbox(
    "Selecciona el modelo a utilizar:",
    list(model_options.keys())
)

MODEL_PATH = model_options[model_choice]
CLASS_NAMES_PATH = "class_names.txt"

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
    class_names = load_class_names(output_shape[-1])
    st.sidebar.success(f"Modelo '{model_choice}' cargado correctamente ✅")
except Exception as e:
    st.error(f"Error al cargar modelo: {e}")
    st.stop()

# ==========================
# INTERFAZ
# ==========================
st.sidebar.header("📸 Instrucciones")
st.sidebar.markdown("""
1. Elige el modelo.  
2. Sube una foto del ave.  
3. Presiona **Clasificar ave**.  
4. Verás la especie probable, su descripción y una tabla de resultados.
""")

uploaded_file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.image(image, caption="📷 Imagen subida", use_container_width=True)

    with col2:
        st.subheader("🔎 Resultado de clasificación")
        if st.button("🟢 Clasificar ave"):
            with st.spinner("Analizando imagen..."):
                img_array = preprocess_image(image)
                results = predict_image(model, img_array, class_names, top_k=3)

            top_pred = results[0]
            name = top_pred["class_name"]
            prob = top_pred["prob"] * 100
            desc = BIRD_DESCRIPTIONS.get(name, DEFAULT_DESCRIPTION)

            st.markdown(f"""
            <div class='result-box'>
                <h3>🏆 Especie más probable</h3>
                <h4 style='color:#05668d;'>{name}</h4>
                <p><b>Confianza:</b> {prob:.2f}%</p>
                <p><b>Descripción:</b> {desc}</p>
            </div>
            """, unsafe_allow_html=True)

            df = pd.DataFrame({
                "Especie": [r["class_name"] for r in results],
                "Probabilidad (%)": [round(r["prob"]*100, 2) for r in results]
            })

            st.markdown("### 📊 Tabla de predicciones")
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("Especie"))
else:
    st.info("👆 Sube una imagen para comenzar.")

