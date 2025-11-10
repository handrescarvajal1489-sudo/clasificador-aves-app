import os
import unicodedata
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd

# ==========================
# CONFIGURACIÓN GENERAL
# ==========================
st.set_page_config(
    page_title="DEEP LEARNING🦜",
    page_icon="🦜",
    layout="wide",
)

# ==========================
# ESTILO PERSONALIZADO
# ==========================
st.markdown(
    """
<style>
/* Fondo general amarillo clásico */
.stApp {
    background-color: #FCDD09;
}

/* Contenedor principal (zona central) */
.block-container {
    background-color: #6D090D;
    padding: 2rem 2rem 3rem 2rem;
    border-radius: 16px;
}

/* Botones de título */
.title-button {
    background-color: #FCDD09;
    color: #6D090D !important;
    padding: 20px 60px;                 /* 🔹 Más grande */
    border-radius: 20px;
    display: block;                     /* 🔹 Permite centrar */
    margin: 30px auto;                  /* 🔹 Centrado horizontal */
    font-weight: 900;
    font-size: 28px;                    /* 🔹 Tamaño más grande */
    text-align: center;                 /* 🔹 Texto centrado */
    font-family: 'Segoe UI', sans-serif;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    box-shadow: 2px 3px 10px rgba(0,0,0,0.5);
    border: 3px solid #5c1a1a;
    transition: all 0.3s ease-in-out;
}
.title-button:hover {
    background-color: #6A0000;
    color: #FFD700 !important;
    transform: scale(1.05);
}

/* Subtítulo mantiene tamaño más pequeño */
.subtitle-button {
    background-color: #FCDD09;
    color: #6D090D !important;
    padding: 10px 25px;
    border-radius: 12px;
    display: inline-block;
    font-weight: 800;
    font-size: 18px;
    font-family: 'Segoe UI', sans-serif;
    text-shadow: 0px 0px 3px rgba(0,0,0,0.4);
    box-shadow: 1px 2px 5px rgba(0,0,0,0.4);
    border: 2px solid #5c1a1a;
    margin: 10px 0;
    transition: all 0.25s ease-in-out;
}
.subtitle-button:hover {
    background-color: #6A0000;
    color: #FFD700 !important;
    transform: scale(1.05);
}


/* Textos generales */
h1, h2, h3, h4, label, p, span, li {
    color: #FBDDAB;
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
/* Botón principal (Predecir especie) – igual a los botones amarillos */
div.stButton > button:first-child {
    background-color: #FCDD09 !important;      /* amarillo */
    color: #6D090D !important;                 /* texto vino tinto */
    font-weight: 800 !important;
    font-size: 18px !important;
    font-family: 'Segoe UI', sans-serif !important;
    border: 2px solid #5c1a1a !important;
    border-radius: 12px !important;
    padding: 10px 25px !important;
    text-shadow: 0px 0px 3px rgba(0,0,0,0.4);
    box-shadow: 1px 2px 5px rgba(0,0,0,0.4) !important;  /* misma sombra */
    margin: 10px 0 !important;
    transition: all 0.25s ease-in-out !important;
}

/* Forzamos también el color del texto interno (span) */
div.stButton > button:first-child span {
    color: #6D090D !important;
}

/* Hover igual que "Carga la imagen" */
div.stButton > button:first-child:hover {
    background-color: #6A0000 !important;      /* vino tinto */
    color: #FFD700 !important;                 /* texto dorado */
    transform: scale(1.05);
}

div.stButton > button:first-child:hover span {
    color: #FFD700 !important;                 /* texto dorado en hover */
}


/* Botón de carga (Browse files) */
section[data-testid="stFileUploader"] button {
    color: #FCDD09 !important;
    font-weight: 800 !important;
    border: 2px solid #5c1a1a !important;
    background-color: #8b2b2b !important;
    border-radius: 10px !important;
    text-shadow: 0px 0px 3px rgba(0,0,0,0.4);
    box-shadow: 1px 2px 5px rgba(0,0,0,0.4);
    transition: all 0.3s ease-in-out;
    font-size: 16px !important;
}
section[data-testid="stFileUploader"] button:hover {
    background-color: #6A0000 !important;
    color: #FFD700 !important;
    border-color: #FFD700 !important;
}

/* Texto del uploader */
section[data-testid="stFileUploader"] * {
    color: #FBDDAB !important;
    font-weight: 600;
}

/* Caja resultado */
.result-box {
    background-color: #6D090D;         /* fondo vino tinto */
    border: 2px solid #FFD700;         /* borde dorado */
    border-radius: 15px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #FCDD09;                    /* texto amarillo */
}
.result-box h2, 
.result-box h3, 
.result-box p, 
.result-box b, 
.result-box i {
    color: #FCDD09 !important;         /* todos los textos amarillos */
}

/* Marca de agua centrada abajo */
.watermark {
    position: fixed;
    left: 50%;
    bottom: 5px;                       /* bien abajo */
    transform: translateX(-50%);
    font-size: 15px;
    font-weight: 600;
    color: rgba(0, 0, 0, 0.8);
    z-index: 9999;
    text-align: center;
    width: 100%;
}
</style>
""",
    unsafe_allow_html=True,
)

# Marca de agua
st.markdown(
    "<div class='watermark'>Hollman Carvajal - Universidad Cooperativa</div>",
    unsafe_allow_html=True,
)

# ==========================
# FUNCIÓN DE NORMALIZACIÓN
# ==========================
def normalizar(texto: str) -> str:
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto

# ==========================
# DATOS DE ESPECIES
# ==========================
species_table = {
    "Especie científica": [
        "Amazilia cyanifrons",  
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
        "Bosques andinos y subandinos (Ibagué, Villahermosa).",
        "Bosques y bordes de bosque (Tolima Central).",
        "Páramos y zonas altas (PNN Los Nevados, Murillo).",
        "Bosques húmedos del Magdalena medio (Norte del Tolima).",
        "Valle del río Magdalena, zonas bajas y cálidas.",
        "Bosques de niebla, Andes Centrales.",
        "Bosques andinos y subandinos (El Líbano, Roncesvalles).",
        "Bosques de Palma de Cera (PNN Los Nevados).",
        "Zonas abiertas cerca de agua (Flandes, Espinal).",
    ],
}
df_species = pd.DataFrame(species_table)

species_info = {
    normalizar(row["Especie científica"]): (row["Nombre común"], row["Hábitat"])
    for _, row in df_species.iterrows()
}

# ==========================
# FUNCIONES AUXILIARES
# ==========================
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)

@st.cache_data
def load_class_names(num_classes: int, path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        return names[:num_classes]
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
# CONFIGURACIÓN DE MODELOS (CON SIDEBAR COMPLETO)
# ==========================
model_options = {
    "VGG16": os.path.join("modelos", "dataset_final_defini.keras"),
    "NASNetMobile": os.path.join("modelos", "dataset_nasnetmobile.keras"),
}
CLASS_NAMES_PATH = "class_names.txt"

st.sidebar.title("⚙ Configuración del modelo")
model_choice = st.sidebar.selectbox("Selecciona el modelo:", list(model_options.keys()))
MODEL_PATH = model_options[model_choice]

try:
    model = load_model(MODEL_PATH)
    num_classes = model.output_shape[-1]
    class_names = load_class_names(num_classes, CLASS_NAMES_PATH)

    st.sidebar.success(f"Modelo '{model_choice}' cargado correctamente ✅")
    st.sidebar.metric("Nº de clases", num_classes)
    st.sidebar.metric("Modelo activo", model_choice)

    st.sidebar.markdown("### 🧠 Sobre el proyecto")
    st.sidebar.markdown(
        f"""
Clasificación inteligente de aves del Tolima usando redes neuronales
con arquitecturas *VGG16* y *NASNetMobile*.

- 🧬 Tipo de modelo: CNN  
- 🐦 Especies entrenadas: *{num_classes}* clases  
- 🎓 Autor: Hollman Carvajal  
- 🏫 Universidad Cooperativa de Colombia – Sede Ibagué  
"""
    )

    st.sidebar.markdown("### 🪶 Consejos de uso")
    st.sidebar.markdown(
        """
- Usa imágenes claras con el ave centrada.  
- Evita fondos muy oscuros o desenfoques extremos.  
- Formatos admitidos: *JPG* y *PNG*.  
"""
    )

    st.sidebar.markdown("### 🐥 Especies incluidas")
    st.sidebar.dataframe(df_species, use_container_width=True)

except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# ==========================
# INTERFAZ PRINCIPAL
# ==========================
st.markdown(
    "<div class='title-button'>🦜 Clasificación inteligente de aves del Tolima</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "Sube una imagen de un ave y deja que el modelo prediga su especie basada en el entrenamiento con aves del Tolima."
)

st.markdown("<div class='subtitle-button'>📸 Carga la imagen</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_display = np.array(image)

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.markdown(
            "<div class='subtitle-button'>📸 Imagen cargada correctamente</div>",
            unsafe_allow_html=True,
        )
        st.image(img_display, use_column_width=True)

    with col2:
        if st.button("🔍 Predecir especie"):
            with st.spinner("Analizando imagen..."):
                img_array = preprocess_image(image)
                results = predict_image(model, img_array, class_names, top_k=3)

            top_pred = results[0]
            sci_name = top_pred["class_name"].replace("_", " ").strip()
            prob = top_pred["prob"] * 100

            normalized = normalizar(sci_name)
            common_name, habitat = next(
                (
                    (v[0], v[1])
                    for k, v in species_info.items()
                    if normalized in k or k in normalized
                ),
                ("Nombre común no disponible", "Hábitat no disponible."),
            )

            st.markdown(
                f"""
                <div class='result-box'>
                    <h3>🏆 Especie más probable</h3>
                    <h2>{common_name}</h2>
                    <p><b>Nombre científico:</b> <i>{sci_name}</i></p>
                    <p><b>Confianza:</b> {prob:.2f}%</p>
                    <p><b>Hábitat:</b> {habitat}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tabla de Top 3
            df_pred = pd.DataFrame(
                {
                    "Especie (modelo)": [r["class_name"] for r in results],
                    "Probabilidad (%)": [round(r["prob"] * 100, 2) for r in results],
                }
            )
            st.markdown("### 📊 Tabla de predicciones (Top 3)")
            st.dataframe(df_pred, use_container_width=True)

            # 📈 Gráfica de barras de las 3 especies
            st.markdown("### 📈 Distribución de probabilidades (Top 3)")
            st.bar_chart(df_pred.set_index("Especie (modelo)"))
else:
    st.info("👆 Sube una imagen para comenzar la detección.")
















