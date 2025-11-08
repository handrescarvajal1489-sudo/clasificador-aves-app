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
/* Fondo general */
.stApp {
    background: linear-gradient(
        180deg,
        #8b2b2b 0%,
        #8b2b2b 35%,
        #fff176 35%,
        #fffde7 100%
    );
}

/* Contenedor */
.block-container {
    background-color: rgba(0, 0, 0, 0.03);
    padding: 2rem 2rem 3rem 2rem;
    border-radius: 16px;
}

/* Títulos */
h1 { color: #ffffff !important; }
h2, h3, h4, label, p, span, li { color: #111111 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161921;
    border-right: 3px solid #8b2b2b;
}

/* Títulos del sidebar en blanco */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Texto del sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span {
    color: #f5f5f5;
    font-size: 14px;
}

/* Botones */
div.stButton > button:first-child {
    background-color: #8b2b2b;
    color: #FFD700;
    border-radius: 10px;
    font-size: 16px;
    padding: 0.5em 1.1em;
    font-weight: 600;
}
div.stButton > button:first-child:hover {
    background-color: #6A0000;
    color: #ffffff;
}

/* Caja resultado */
.result-box {
    background-color: rgba(0, 0, 0, 0.35);
    border: 2px solid #FFD700;
    border-radius: 15px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #ffffff;
}

/* Texto dentro del uploader */
section[data-testid="stFileUploader"] div {
    color: white !important;
}

/* Marca de agua */
.watermark {
    position: fixed;
    left: 50%;
    bottom: 20px;
    transform: translateX(-50%);
    font-size: 15px;
    font-weight: 600;
    color: rgba(0, 0, 0, 0.8);
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

# Marca de agua
st.markdown(
    "<div class='watermark'>Hollman Carvajal - Universidad Cooperativa</div>",
    unsafe_allow_html=True
)

# ==========================
# DATOS DE ESPECIES
# ==========================
species_info = {
    "Amazilia cyaninfrons": ("Colibrí Gorriiazul", "🌄 Zonas andinas y subandinas."),
    "Anthocephala berlepschi": ("Colibrí Cabecicastaño Andino", "🌲 Endémico: Bosques andinos y subandinos (Ibagué, Villahermosa)."),
    "Atlapetes flaviceps": ("Pinzón Cabeciamarillo", "🌲 Bosques y bordes de bosque (Tolima Central)."),
    "Bolborhynchus ferrugineifrons": ("Periquito de los Nevados", "🏔️ Páramos y zonas altas (PNN Los Nevados, Murillo)."),
    "Crax alberti": ("Paujil Colombiano", "💧 Bosques húmedos del Magdalena medio (Norte del Tolima)."),
    "Euphonia concinna": ("Eufonia del Magdalena", "🌞 Valle del río Magdalena, zonas bajas y cálidas."),
    "Hapalopsittaca fuertesi": ("Loro Coroniazul", "🌫️ Bosques de niebla, Andes Centrales (límites con Quindío)."),
    "Leptotila conoveri": ("Paloma Montaraz de Tolima", "🌲 Endémico: Bosques andinos y subandinos (El Líbano, Roncesvalles)."),
    "Ognorhynchus icterotis": ("Loro Orejiamarillo", "🌴 Bosques de Palma de Cera (PNN Los Nevados)."),
    "Pyrocephalus rubinus": ("Atrapamoscas Pechirrojo", "🏞️ Zonas abiertas cerca de agua (Flandes, Espinal).")
}

# ==========================
# FUNCIONES
# ==========================
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(img, size=(224, 224)):
    img = img.convert("RGB").resize(size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def predict_image(model, img_array, class_names, top_k=3):
    preds = model.predict(img_array)[0]
    indices = np.argsort(preds)[::-1][:top_k]
    return [{"class_name": class_names[i], "prob": float(preds[i])} for i in indices]

# ==========================
# CONFIGURACIÓN DE MODELO
# ==========================
model_options = {
    "VGG16": os.path.join("modelos", "dataset_vgg16.keras"),
    "NASNetMobile": os.path.join("modelos", "dataset_nasnetmobile.keras"),
}
CLASS_NAMES_PATH = "class_names.txt"

st.sidebar.title("⚙️ Configuración del modelo")
model_choice = st.sidebar.selectbox("Selecciona el modelo a utilizar:", list(model_options.keys()))
MODEL_PATH = model_options[model_choice]

try:
    model = load_model(MODEL_PATH)
    num_classes = model.output_shape[-1]
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    st.sidebar.success(f"Modelo '{model_choice}' cargado correctamente ✅")
    st.sidebar.metric("Nº de clases", num_classes)
    st.sidebar.metric("Modelo activo", model_choice)

    st.sidebar.markdown("### 🧠 Sobre el proyecto")
    st.sidebar.markdown(f"""
Proyecto académico que implementa un **clasificador de aves colombianas** mediante **Deep Learning (CNN)**.

- 🧬 Arquitecturas: `VGG16` y `NASNetMobile`  
- 🐦 Especies reconocibles: **{num_classes}**  
- 🎓 Autor: *Hollman Carvajal - Universidad Cooperativa*  
- 🧪 Enfoque: Procesamiento de imágenes y predicción visual.
""")

    st.sidebar.markdown("### 🪶 Consejos de uso")
    st.sidebar.markdown("""
- Usa imágenes claras, con el ave centrada.  
- Evita sombras o fondos muy oscuros.  
- Formatos admitidos: **JPG / PNG**.  
""")

except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# ==========================
# INTERFAZ PRINCIPAL
# ==========================
st.subheader("📸 Sube tu imagen")
uploaded_file = st.file_uploader("Sube una imagen de un ave (JPG o PNG)", type=["jpg", "jpeg", "png"])

st.markdown("Una vez cargues la imagen, pulsa **Clasificar ave** para ver las 3 especies más probables.")

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.subheader("📷 Imagen subida")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔎 Predicción")
        if st.button("🔍 Clasificar ave"):
            with st.spinner("Analizando imagen..."):
                img_array = preprocess_image(image)
                results = predict_image(model, img_array, class_names, top_k=3)

            top_pred = results[0]
            name = top_pred["class_name"]
            prob = top_pred["prob"] * 100

            # Buscar información
            info = species_info.get(name, ("Especie desconocida", "Sin información disponible."))
            common_name, habitat = info

            st.markdown(f"""
            <div class='result-box'>
                <h3>🏆 Especie más probable</h3>
                <h2>{common_name}</h2>
                <p><b>Nombre científico:</b> <i>{name}</i></p>
                <p><b>Confianza:</b> {prob:.2f}%</p>
                <p><b>Hábitat:</b> {habitat}</p>
            </div>
            """, unsafe_allow_html=True)

            # Tabla de resultados
            df = pd.DataFrame({
                "Especie": [r["class_name"] for r in results],
                "Probabilidad (%)": [round(r["prob"]*100, 2) for r in results]
            })
            st.markdown("### 📊 Tabla de predicciones (Top 3)")
            st.dataframe(df, use_container_width=True)

            st.markdown("### 📈 Distribución de probabilidades")
            st.bar_chart(df.set_index("Especie"))
else:
    st.info("👆 Sube una imagen para comenzar la clasificación.")




