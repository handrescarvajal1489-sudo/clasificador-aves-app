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

/* Título principal */
h1 {
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}
h2, h3, h4, label, p, span, li {
    color: #111111;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161921;
    border-right: 3px solid #8b2b2b;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Botones rojos principales (Predicción, Clasificar, Imagen subida) */
.btn-red {
    background-color: #8b2b2b;
    color: #ffffff !important;
    padding: 10px 18px;
    border-radius: 10px;
    display: inline-block;
    font-weight: bold;
    font-size: 18px;
    box-shadow: 1px 1px 4px rgba(0,0,0,0.3);
    margin-bottom: 10px;
    border: none;
    text-align: center;
    transition: all 0.25s ease-in-out;
}
.btn-red:hover {
    background-color: #6A0000;
    color: #FFD700 !important;
    transform: scale(1.05);
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

/* Botón del uploader (Browse files) - SIEMPRE LEGIBLE */
section[data-testid="stFileUploader"] button {
    color: #ffffff !important;
    border: 2px solid #FFD700 !important;
    background-color: #8b2b2b !important;
    font-weight: 600 !important;
    transition: all 0.3s ease-in-out !important;
}
section[data-testid="stFileUploader"] button:hover {
    background-color: #6A0000 !important;
    border-color: #FFD700 !important;
    color: #FFD700 !important;
}

/* Texto del uploader */
section[data-testid="stFileUploader"] * {
    color: #ffffff !important;
}

/* Botón principal (Clasificar ave) igual al de Predicción */
div.stButton > button:first-child {
    background-color: #8b2b2b !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border: none !important;
    padding: 10px 20px !important;
    box-shadow: 1px 1px 4px rgba(0,0,0,0.3) !important;
    transition: all 0.25s ease-in-out !important;
}
div.stButton > button:first-child:hover {
    background-color: #6A0000 !important;
    color: #FFD700 !important;
    transform: scale(1.05);
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
# FUNCIONES
# ==========================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    return tf.keras.models.load_model(model_path)

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
    class_names = load_class_names(num_classes, CLASS_NAMES_PATH)

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

    st.sidebar.markdown("### 🐥 Especies clasificadas")
    st.sidebar.dataframe(df_species, use_container_width=True)

except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# ==========================
# INTERFAZ PRINCIPAL
# ==========================
st.title("🦜 Clasificador de Aves")
st.markdown("Sube una imagen de un ave y deja que el modelo de *Deep Learning* prediga la especie.")

st.subheader("📸 Sube tu imagen")
uploaded_file = st.file_uploader(
    "Sube una imagen de un ave (JPG o PNG)",
    type=["jpg", "jpeg", "png"]
)

st.markdown("Una vez cargues la imagen, pulsa **Clasificar ave** para ver las 3 especies más probables.")

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.markdown('<div class="btn-red">📸 Imagen subida</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="btn-red">🔍 Predicción</div>', unsafe_allow_html=True)

        if st.button("🔍 Clasificar ave", key="predict_button"):
            with st.spinner("Analizando imagen..."):
                img_array = preprocess_image(image)
                results = predict_image(model, img_array, class_names, top_k=3)

            top_pred = results[0]
            sci_name = top_pred["class_name"].replace("_", " ")
            prob = top_pred["prob"] * 100

            common_name, habitat = species_info.get(
                sci_name,
                ("Nombre común no disponible", "Hábitat no disponible.")
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
            st.markdown("### 📊 Tabla de predicciones (Top 3)")
            st.dataframe(df_pred, use_container_width=True)
            st.markdown("### 📈 Distribución de probabilidades")
            st.bar_chart(df_pred.set_index("Especie (modelo)"))
else:
    st.info("👆 Sube una imagen para comenzar la clasificación.")






