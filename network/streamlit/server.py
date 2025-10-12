import os, json, uuid, time
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
TIMEOUT = float(os.getenv("FRONT_TIMEOUT", "30"))

st.set_page_config(page_title="MalvadiToS", page_icon="⚖️", layout="wide")
st.title("MalvadiToS · Analizador de Términos y Condiciones")

with st.sidebar:
    st.caption("Back-end")
    st.write("API:", API_URL)
    if st.button("Probar /healthz"):
        try:
            r = requests.get(f"{API_URL}/healthz", timeout=10)
            st.json(r.json())
        except Exception as e:
            st.error(f"healthz falló: {e}")

st.markdown("Pega el texto de Términos y Condiciones y dale **Analizar**.")

with st.form("tos_form", clear_on_submit=False):
    text = st.text_area("Texto", height=300, placeholder="Pega tu ToS aquí…")

    with st.expander("Opciones avanzadas (opcional)"):
        col1, col2 = st.columns(2)
        with col1:
            pre_opts = st.text_area("preprocess_kwargs (JSON)", height=120, placeholder='{}')
        with col2:
            inf_opts = st.text_area("infer_kwargs (JSON)", height=120, placeholder='{}')

    submitted = st.form_submit_button("Analizar")

if submitted:
    if not text.strip():
        st.warning("Necesitas ingresar texto.")
        st.stop()

    try:
        preprocess_kwargs = json.loads(pre_opts.strip() or "{}")
        infer_kwargs = json.loads(inf_opts.strip() or "{}")
    except Exception as e:
        st.error(f"JSON inválido en opciones: {e}")
        st.stop()

    payload = {
        "text": text,
        "preprocess_kwargs": preprocess_kwargs,
        "infer_kwargs": infer_kwargs,
    }
    req_id = str(uuid.uuid4())
    headers = {"X-Request-ID": req_id}

    with st.spinner("Enviando al modelo…"):
        t0 = time.time()
        try:
            r = requests.post(f"{API_URL}/analyze", json=payload, headers=headers, timeout=TIMEOUT)
            latency = time.time() - t0
        except requests.exceptions.Timeout:
            st.error(f"Tiempo agotado tras {TIMEOUT}s (request-id {req_id})")
            st.stop()
        except Exception as e:
            st.error(f"Error de red: {e} (request-id {req_id})")
            st.stop()

    st.write(f"Request-ID: `{req_id}` · Latencia: **{latency:.2f}s**")

    if r.status_code >= 400:
        st.error(f"API respondió {r.status_code}: {r.text}")
        st.stop()

    try:
        data = r.json()
    except Exception:
        st.error("La API no devolvió JSON.")
        st.stop()

    st.subheader("Resultado")
    st.json(data)

    # Tabla automática si viene lista de dicts en alguna clave
    if isinstance(data, dict):
        for k, v in list(data.items()):
            if isinstance(v, list) and v and isinstance(v[0], dict):
                st.markdown(f"### {k}")
                st.dataframe(v, use_container_width=True)
