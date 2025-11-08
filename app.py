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
# ESTILOS PERSONALIZADOS
# ==========================
st.markdown("""
<style>

/* --- Fondo general amarillo claro --- */
.stApp {
    background-color: #fffde7;
}

/* --- Encabezado fijo vino tinto --- */
.header-fixed {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 110px;
    background-color: #8b2b2b;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    font-weight: 800;
    font-family: 'Segoe UI', sans-serif;
    z-index: 1000;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}

/* --- Ajuste del contenido principal para que no quede debajo del header --- */
.block-container {
    margin-top: 130px;
    background-color: rgba(0, 0, 0, 0.02);
    padding: 2rem 2rem 3rem 2rem;
    border-radius: 16px;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background-color: #161921;
    border-right: 3px solid #8b2b2b;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}

/* --- Títulos y textos --- */
h1, h2, h3, h4, p, span, li, label {
    font-family: 'Segoe UI', sans-serif;
    color: #111111;
}
h1 {
    color: #8b2b2b !important;
}

/* --- Botones principales --- */
div.stButton > button:first-child,
section[data-testid="stFileUploader"] button {
    background-color: #8b2b2b !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    border: 1.5px solid #5c1a1a !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    text-shadow: 0px 0px 3px rgba(0,0,0,0.4);
    box-shadow: 1px 2px 5px rgba(0,0,0,0.4);
    transition: all 0.25s ease-in-out !important;
}
div.stButton > button:first-child:hover,
section[data-testid="stFileUploader"] button:hover {
    background-color: #6A0000 !important;
    color: #FFD700 !important;
    transform: scale(1.05);
}

/* --- Caja de resultados --- */
.result-box {
    background-color: rgba(0, 0, 0, 0.6);
    border: 2px solid #FFD700;
    border-radius: 15px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #ffffff;
}

/* --- Marca de agua inferior --- */
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

# --- Header fijo ---
st.markdown(
    "<div class='header-fixed'>🦜 Clasificador de Aves Colombianas</div>",
    unsafe_allow_html=True,
)

# --- Marca de agua ---
st.markdown(
    "<div class='watermark'>Hollman Carvajal - Universidad Cooperativa</div>",
    unsafe_allow_html=True,
)

# ==========================
# DATOS DE ESPECIES
# ==========================
species_table = {
    "Especie científica": [
        "Amazilia cyaninfrons",
        "Anthocephala berlepschi",
        "Atlapetes flaviceps",
        "Bolborhynchus ferrugineifrons",
        "Crax alberti",
        "Euphonia concinna",
        "Hapalopsittaca fuertesi",
        "Leptotila conoveri",
        "Ognorhynchus icterotis",
        "Pyrocephalus rubinus",
    ],
    "Nombre común": [
        "Colibrí Gorriiazul",
        "Colibrí Cabecicastaño Andino",
        "Pinzón Cabeciamarillo / Gorrión de Anteojos",
        "Lorito Cadillero / Periquito de los Nevados",
        "Paujil Colombiano",
        "Eufonia del Magdalena",
        "Loro Coroniazul",
        "Paloma Montaraz de Tolima / Caminera Tolimense",
        "Loro Orejiamarillo",
        "Atrapamoscas Pechirrojo",
    ],
    "Hábitat": [
        "Zonas andinas y subandinas.",
        "Endémico: Bosques andinos y subandinos (Ibagué, Villahermosa).",
        "Endémico: Bosques y bordes de bosque (Villa Hermosa, Tolima Central).",
        "Páramos y zonas altas (PNN Los Nevados, Murillo).",
        "Especie en peligro. Bosques húmedos del Magdalena medio (Norte del Tolima).",
        "Valle del río Magdalena, zonas bajas y cálidas.",
        "Bosques de niebla, Andes Centrales (límites con Quindío).",
        "Endémico: Bosques andinos y subandinos (El Líbano, Roncesvalles).",
        "Bosques de Palma de Cera (PNN Los Nevados).",
        "Zonas abiertas cerca de agua (Flandes, Espinal).",
    ],
}
df_species = pd.DataFrame(species_table)
species_info = {
    row["Especie científica"]: (row["Nombre común"], row["Hábitat"])
    for _, row in df_species.iterrows()
}

# ==========================
# FUNCIONES AUXILIARES
# ==========================
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)

@st.cache_data
def load_class_names(num_classes: int, path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) >= num_classes:
            return names[:num_classes]
        else:
            names += [f"Clase {i}" for i in range(len(names), num_classes)]
            return names
    return [f"Clase {i}" for i in range(num_classes)]

def preprocess_image(img: Image.Image, size=(224, 224)):
    img = img.convert("RGB").resize(size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def predict_image(model, img_array, class_names, top_k=3):
    preds = model.predict(img_array, verbose=0)[0]
    indices = np.argsort(preds)[::-1][:top_k]
    return [{"class_name": class_names[i], "prob": float(preds[i])} for i in indices]

# ==========================
# CONFIGURACIÓN DE MODELOS
# ==========================
model_options = {
    "VGG16": os.path.join("modelos", "dataset_final_defini.keras"),
    "NASNetMobile": os.path.join("modelos", "dataset_nasnetmobile.keras"),
}
CLASS_NAMES_PATH = "class_names.txt"

st.sidebar.title("⚙ Configuración del modelo")
model_choice = st.sidebar.selectbox("Selecciona el modelo a utilizar:", list(model_options.keys()))
MODEL_PATH = model_options[model_choice]

try:
    model = load_model(MODEL_PATH)
    num_classes = model.output_shape[-1]
    class_names = load_class_names(num_classes, CLASS_NAMES_PATH)
    st.sidebar.success(f"Modelo '{model_choice}' cargado correctamente ✅")
    st.sidebar.metric("Nº de clases", num_classes)
    st.sidebar.metric("Modelo activo", model_choice)
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# ==========================
# INTERFAZ PRINCIPAL
# ==========================
st.title("🦜 Clasificador de Aves")
st.markdown("Sube una imagen y deja que el modelo prediga la especie del ave.")

uploaded_file = st.file_uploader("📸 Sube tu imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_display = np.array(image)
    except Exception as e:
        st.error(f"No se pudo abrir la imagen. Detalle: {e}")
        st.stop()

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.markdown("<h4 style='color:#8b2b2b;'>📸 Imagen subida</h4>", unsafe_allow_html=True)
        st.image(img_display, use_column_width=True)

    with col2:
        st.markdown("<h4 style='color:#8b2b2b;'>🔍 Predicción</h4>", unsafe_allow_html=True)
        if st.button("Clasificar ave"):
            with st.spinner("Analizando imagen..."):
                img_array = preprocess_image(image)
                results = predict_image(model, img_array, class_names)
            top_pred = results[0]
            sci_name = top_pred["class_name"].replace("_", " ")
            prob = top_pred["prob"] * 100
            common_name, habitat = species_info.get(
                sci_name, ("Nombre común no disponible", "Hábitat no disponible.")
            )
            st.markdown(f"""
            <div class='result-box'>
                <h3>🏆 Especie más probable</h3>
                <h2>{common_name}</h2>
                <p><b>Nombre científico:</b> <i>{sci_name}</i></p>
                <p><b>Confianza:</b> {prob:.2f}%</p>
                <p><b>Hábitat:</b> {habitat}</p>
            </div>
            """, unsafe_allow_html=True)

            df_pred = pd.DataFrame({
                "Especie (modelo)": [r["class_name"] for r in results],
                "Probabilidad (%)": [round(r["prob"] * 100, 2) for r in results],
            })
            st.markdown("### 📊 Resultados de predicción")
            st.dataframe(df_pred, use_container_width=True)
            st.bar_chart(df_pred.set_index("Especie (modelo)"))

else:
    st.info("👆 Sube una imagen para comenzar la clasificación.")
