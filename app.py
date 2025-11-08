import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd  # 👈 para la tabla y la gráfica

# ==========================
# CONFIGURACIÓN GENERAL
# ==========================
st.set_page_config(
    page_title="Clasificador de Aves 🦜",
    page_icon="🦜",
    layout="centered",
)

st.title("🦜 Clasificador de Aves")
st.write(
    "Sube una imagen de un ave y selecciona el modelo entrenado "
    "para predecir a qué especie pertenece."
)

# ==========================
# DESCRIPCIONES DE AVES
# ==========================
# 👉 Aquí puedes completar las descripciones reales de tus especies.
BIRD_DESCRIPTIONS = {
    "Coereba_flaveola": "Pequeña ave tropical conocida como pinchaflor o mielero, frecuente en jardines y zonas urbanas.",
    "Icterus_nigrogularis": "Conocido como turpial, ave de colores vivos muy común en zonas abiertas y arboladas.",
    "Oryzoborus_angolensis": "Semillero robusto, de pico grueso, asociado a áreas con vegetación densa.",
    # Agrega aquí todas las especies de tu class_names.txt...
    # "Nombre_de_tu_clase": "Descripción del ave..."
}

DEFAULT_DESCRIPTION = (
    "Descripción no disponible aún para esta especie. "
    "Puedes actualizar el diccionario BIRD_DESCRIPTIONS en el código."
)

# ==========================
# SELECCIÓN DE MODELO
# ==========================
model_options = {
    "VGG16": os.path.join("modelos", "dataset_vgg16.keras"),
    "NASNetMobile": os.path.join("modelos", "dataset_nasnetmobile.keras"),
}

st.sidebar.header("Configuración del modelo")
model_choice = st.sidebar.selectbox(
    "Selecciona el modelo a utilizar:",
    list(model_options.keys()),
)

MODEL_PATH = model_options[model_choice]
CLASS_NAMES_PATH = "class_names.txt"

# Ajuste automático de tamaño de imagen según modelo
if model_choice == "VGG16":
    IMG_SIZE = (224, 224)
elif model_choice == "NASNetMobile":
    IMG_SIZE = (224, 224)  # Cambia a (331, 331) si tu modelo NASNet usa esa entrada
else:
    IMG_SIZE = (224, 224)


# ==========================
# CARGA DEL MODELO Y CLASES
# ==========================
@st.cache_resource
def load_model(model_path):
    """Carga el modelo seleccionado desde su ruta"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


@st.cache_data
def load_class_names(num_classes: int):
    """
    Lee los nombres de clase desde class_names.txt.
    Si faltan nombres, completa con 'Clase i'.
    Si sobran, recorta.
    """
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]

        if len(names) >= num_classes:
            return names[:num_classes]
        else:
            extra = [f"Clase {i}" for i in range(len(names), num_classes)]
            return names + extra

    # Si no hay archivo, inventa nombres genéricos
    return [f"Clase {i}" for i in range(num_classes)]


def preprocess_image(img: Image.Image, target_size=IMG_SIZE):
    """
    Preprocesa la imagen IGUAL que un ImageDataGenerator(rescale=1./255):
    - Convierte a RGB
    - Redimensiona
    - Escala a rango [0, 1]
    """
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0  # << importante
    img_array = np.expand_dims(img_array, axis=0)        # (1, h, w, 3)
    return img_array


def predict_image(model, img_array, class_names, top_k=3):
    """Devuelve las top-k clases con mayor probabilidad."""
    preds = model.predict(img_array)[0]  # (num_clases,)
    indices = np.argsort(preds)[::-1][:top_k]

    results = []
    for i in indices:
        results.append(
            {
                "class_name": class_names[i],
                "prob": float(preds[i]),
            }
        )
    return results


# ==========================
# INICIALIZACIÓN DEL MODELO
# ==========================
try:
    model = load_model(MODEL_PATH)
    output_shape = model.output_shape  # por ejemplo (None, 10)
    num_classes = output_shape[-1]
    class_names = load_class_names(num_classes)
    st.success(f"✅ Modelo '{model_choice}' cargado correctamente.")
except Exception as e:
    st.error(f"⚠️ Error al cargar el modelo: {e}")
    st.stop()


# ==========================
# INTERFAZ DE USUARIO
# ==========================
st.sidebar.header("Instrucciones")
st.sidebar.markdown(
    """
1. Selecciona el **modelo** a usar.  
2. Haz clic en **Examinar archivos** y elige una foto de un ave (JPG o PNG).  
3. La imagen se mostrará en pantalla.  
4. Pulsa el botón **🔍 Clasificar ave**.  
5. Verás las 3 especies más probables con su porcentaje de confianza, una tabla y una descripción.
"""
)

uploaded_file = st.file_uploader(
    "Sube una imagen de un ave (JPG o PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Imagen subida")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔎 Predicción")
        if st.button("🔍 Clasificar ave"):
            with st.spinner("Analizando imagen..."):
                img_array = preprocess_image(image)
                results = predict_image(model, img_array, class_names, top_k=3)

            if not results:
                st.warning("No se obtuvieron predicciones.")
            else:
                # ----------------------------
                # Predicción principal (TOP 1)
                # ----------------------------
                top_pred = results[0]
                top_name = top_pred["class_name"]
                top_prob = top_pred["prob"] * 100
                description = BIRD_DESCRIPTIONS.get(top_name, DEFAULT_DESCRIPTION)

                st.markdown(
                    f"""
                    ### 🏆 Predicción principal
                    **Especie:** `{top_name}`  
                    **Confianza:** **{top_prob:.2f}%**

                    **Descripción:**  
                    {description}
                    """
                )

                # Barra de progreso para la predicción principal
                st.progress(min(max(top_pred["prob"], 0.0), 1.0))

                # ----------------------------
                # Tabla y gráfica de predicciones (Top 3)
                # ----------------------------
                st.markdown("### 📊 Detalle de predicciones (Top 3)")
                df_results = pd.DataFrame(
                    [
                        {
                            "Posición": i + 1,
                            "Especie": r["class_name"],
                            "Probabilidad (%)": round(r["prob"] * 100, 2),
                        }
                        for i, r in enumerate(results)
                    ]
                )

                # Tabla interactiva
                st.dataframe(df_results, use_container_width=True)

                # Gráfica de barras
                st.markdown("#### Distribución de probabilidades")
                chart_data = df_results.set_index("Especie")["Probabilidad (%)"]
                st.bar_chart(chart_data)

                # Barra de progreso individual por especie
                with st.expander("Ver barras individuales"):
                    for r in results:
                        st.write(f"**{r['class_name']}** → {r['prob'] * 100:.2f}%")
                        st.progress(min(max(r["prob"], 0.0), 1.0))
else:
    st.info("👆 Sube una imagen para poder clasificarla.")


