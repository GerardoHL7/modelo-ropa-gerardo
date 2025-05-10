import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Configuración
st.set_page_config(page_title="Clasificador de Ropa", layout="centered")
st.title("🧥 Clasificador de Prendas de Ropa")
st.write("Sube una imagen de una prenda para predecir su categoría.")

# Cargar modelo
@st.cache_resource
def load_clothing_model():
    model = load_model("modeloRopa.h5")
    return model

model = load_clothing_model()

# Lista de clases (ajusta según tu modelo)
class_names = ['camisa', 'camiseta', 'chaqueta', 'pantalon', 'polo'] 
TARGET_SIZE = (128, 128)  # Tamaño esperado por el modelo

# Subir imagen
uploaded_file = st.file_uploader("📷 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convertir a escala de grises
        img = Image.open(uploaded_file).convert("L")
        st.image(img, caption="Imagen subida (escala de grises)", use_column_width=True)

        # Preprocesamiento
        img = img.resize(TARGET_SIZE)
        img_array = image.img_to_array(img) / 255.0  # (128, 128, 1)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 1)

        # Predicción
        predictions = model.predict(img_array)[0]
        pred_dict = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
        pred_label = class_names[np.argmax(predictions)]

        # Mostrar resultados
        st.subheader("🔍 Resultado:")
        st.write(f"**Predicción principal:** {pred_label}")
        st.bar_chart(pred_dict)

    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
