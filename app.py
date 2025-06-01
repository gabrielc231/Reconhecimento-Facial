import streamlit as st
import tensorflow as tf
from keras import models
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Reconhecimento de Emoções Faciais", layout="centered")

# === 1. Carregar o modelo ===
@st.cache_resource
def load_model(path="models/model6.h5"):
    return models.load_model(path)

model = load_model()

# === 2. Definir classes ===
classes = ['angry', 'happy', 'sad', 'surprise'] 
labels = {
    'angry': 'Raiva',
    'happy': 'Feliz',
    'sad': 'Triste',
    'surprise': 'Surpresa'
}

# === 3. Função de pré-processamento ===
def preprocess_image(image):
    image = image.convert('L')  # Converter para escala de cinza
    image = image.resize((48, 48))  # Redimensionar
    img_array = np.array(image) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=-1)  # Adicionar canal (48, 48, 1)
    img_array = np.expand_dims(img_array, axis=0)   # Adicionar batch (1, 48, 48, 1)
    return img_array

# === 4. Interface do Streamlit ===
st.title("Reconhecimento de Emoções em Expressões Faciais")
st.write("Envie uma image contendo um rosto humano com uma expressão facial clara.")

imagem_upada = st.file_uploader("📤 Escolha uma imagem (JPG ou PNG)", type=["jpg", "jpeg", "png"])

if imagem_upada:
    image = Image.open(imagem_upada)
    st.image(image, caption="Imagem enviada", use_container_width=True)

    if st.button("🔍 Classificar Emoção"):
        img_preprocessed = preprocess_image(image)
        st.image(img_preprocessed, caption="imagem preprocessada", use_container_width=True)
        prediction = model.predict(img_preprocessed)[0]

        indice = np.argmax(prediction)
        classe_predita = classes[indice]
        emocao = labels[classe_predita]
        confianca = prediction[indice] * 100

        st.success(f"Emoção reconhecida: **{emocao}** ({confianca:.2f}%)")

        # Mostrar todas as probabilidades (opcional)
        st.subheader("Distribuição das emoções:")
        for i, classe in enumerate(classes):
            bar_label = labels[classe]
            st.progress(float(prediction[i]), text=f"{bar_label}: {prediction[i]*100:.2f}%")