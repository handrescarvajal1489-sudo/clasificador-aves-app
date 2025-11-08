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
/* Fondo vino tinto + amarillo más suave */
.stApp {
    background: linear-gradient(
        180deg,
        #8b2b2b 0%,
        #8b2b2b 35%,
        #fff176 35%,
        #fffde7 100%
    );
}

/* Contenedor principal transparente */
.block-container {
    background-color: rgba(0, 0, 0, 0.03);
    padding: 2rem 2rem 3rem 2rem;
    border-radius: 16px;
    margin-top: 1rem;
}

/* Títulos */
h1, h2, h3, h4 {
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161921;
    border-right: 3px solid #8b2b2b;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFD700;
    font-family: 'Segoe UI', sans-serif;
}

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

/* Caja resultado */
.result-box {
    background-color: rgba(0, 0, 0, 0.35);
    border: 2px solid #FFD700;
    border-radius: 15px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #ffffff;
}

/* Marca de agua centrada */
.watermark {
    position: fixed;
    left: 50%;
    bottom: 20px;
    transform: translateX(-50%);
    font-size: 15px;
    font-weight: 600;
    color: rgba(0, 0, 0, 0.4);
    z-index: 9999;
    pointer-events: none;
    font-family: 'Segoe UI', sans-serif;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# Marca de agua centrada
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
    "prediga la especie con base en redes convolucionales (CNN)."
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

# Tamaño de imagen
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

    # CONTEXTO DEL PROYECTO
    st.sidebar.markdown("### ℹ️ Sobre el proyecto")
    st.sidebar.markdown(f"""
Proyecto académico desarrollado como demostración de un sistema de **clasificación de aves colombianas**
mediante modelos de **Deep Learning (CNN)**.

- 🧠 Arquitecturas: `VGG16` y `NASNetMobile`
- 🐦 Especies reconocibles: **{num_classes}**
- 🎓 Autor: *Hollman Carvajal - Universidad Cooperativa*
- 🧪 Enfoque: Procesamiento de imágenes y predicción visual.
""")

    st.sidebar.markdown("### ✅ Consejos de uso")
    st.sidebar.markdown("""
- Usa imágenes claras, con el ave centrada.  
- Formatos admitidos: **JPG** y **PNG**.  
- Ideal para análisis visual o presentaciones científicas.
""")

    # ==========================
    # TABLA DE ESPECIES EN EL LATERAL
    # ==========================
    st.sidebar.markdown("### 🐥 Especies clasificadas")

    data = {
        "Especie científica": [
            "Amazilia cyaninfrons", "Anthocephala berlepschi", "Atlapetes flaviceps",
            "Bolborhynchus ferrugineifrons", "Crax alberti", "Euphonia concinna",
            "Hapalopsittaca fuertesi", "Leptotila conoveri", "Ognorhynchus icterotis",
            "Pyrocephalus rubinus"
        ],
        "Nombre común": [
            "Colibrí Gorriiazul", "Colibrí Cabecicastaño Andino", "Pinzón Cabeciamarillo",
            "Periquito de los Nevados", "Paujil Colombiano", "Eufonia del Magdalena",
            "Loro Coroniazul", "Paloma Montaraz de Tolima", "Loro Orejiamarillo",
            "Atrapamoscas Pechirrojo"
        ],
        "Hábitat": [
            "Zonas andinas y subandinas", "Bosques andinos (Ibagué, Villahermosa)",
            "Bordes de bosque (Tolima Central)", "Páramos (PNN Los Nevados, Murillo)",
            "Bosques húmedos del Magdalena medio", "Valle del río Magdalena",
            "Bosques de niebla (Andes Centrales)", "Bosques andinos (El Líbano, Roncesvalles)",
            "Bosques de Palma de Cera (PNN Los Nevados)", "Zonas abiertas cerca de agua"
        ]
    }

    df_species = pd.DataFrame(data)
    st.sidebar.dataframe(df_species, use_container_width=True)

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

st.markdown("Una vez cargues la imagen, pulsa **Clasificar ave** para ver las 3 especies más probables.")

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





