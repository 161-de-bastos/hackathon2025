import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://fastapi:8000/predict")
API_SECRET = os.getenv("API_SECRET_KEY", "")

st.title("Cliente Streamlit - Interfaz para el modelo GPU")

prompt = st.text_area("Escribe el texto que quieres enviar al modelo:")
if st.button("Enviar al modelo"):
    with st.spinner("Consultando modelo en GPU..."):
        payload = {
            "secret_key": API_SECRET,
            "data": {"prompt": prompt}  # lo que tu modelo espera
        }
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            if r.status_code == 200:
                st.subheader("Respuesta del modelo:")
                st.write(r.json())
            else:
                st.error(f"Error del servidor: {r.status_code}\n{r.text}")
        except Exception as e:
            st.error(f"No se pudo contactar al servidor: {e}")
