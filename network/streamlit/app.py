import os
import requests
import streamlit as st
import pandas as pd
from html import escape

# -------------------------------------------------------------------
# Configuración
# - En Streamlit Cloud define un Secret o variable de entorno API_URL
# - Ej.: API_URL = "https://tu-fastapi.tudominio.com"
# -------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Clasificador Justa/Injusta", layout="wide")

# Estado
if "results" not in st.session_state:
    st.session_state.results = []
if "focus_idx" not in st.session_state:
    st.session_state.focus_idx = None
if "notes" not in st.session_state:
    st.session_state.notes = {}

# Estilos
st.markdown("""
<style>
.sentence-chip {
  display: block;
  padding: 10px 12px;
  margin: 6px 0;
  border-radius: 10px;
  cursor: pointer;
  font-size: 0.98rem;
  border: 1px solid rgba(0,0,0,0.06);
}
.s-justa   { background: #e7f6ec; color: #0e5c2f; }
.s-injusta { background: #fde7e7; color: #7a0b0b; }
.detail-card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 10px;
  padding: 12px;
  background: #fff;
}
.small-muted { color: #6b7280; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("Clasificador de líneas: Justa / Injusta")

with st.container():
    left, right = st.columns([2,1], gap="large")

    with left:
        st.subheader("Ingresa el texto")
        demo = "Hola, ¿cómo estás? Vamos a aterrorizar a tus animales. Esto es solo un ejemplo."
        txt = st.text_area("Pega tu texto aquí", value=demo, height=180)
        if st.button("Analizar", type="primary", use_container_width=True):
            st.session_state.focus_idx = None
            st.session_state.notes = {}
            try:
                r = requests.post(f"{API_URL}/v1/predict", json={"text": txt}, timeout=60)
                r.raise_for_status()
                payload = r.json()
                st.session_state.results = payload.get("sentences", [])
                if not st.session_state.results:
                    st.info("No se detectaron oraciones en el texto.")
            except Exception as e:
                st.error(f"Error llamando a la API ({API_URL}). Detalle: {e}")

        # Render de resultados
        if st.session_state.results:
            st.subheader("Resultados por oración")
            for item in st.session_state.results:
                idx = item["idx"]
                label = item["label"]  # 0=justa, 1=injusta
                text = escape(item["text"])
                css = "s-injusta" if label == 1 else "s-justa"
                icon = "🔴" if label == 1 else "🟢"
                # Un botón por oración (permite click para ver detalles)
                if st.button(f"{icon} {text}", key=f"chip_{idx}", use_container_width=True):
                    st.session_state.focus_idx = idx

            with st.expander("Ver tabla y exportar"):
                df = pd.DataFrame(st.session_state.results)
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Descargar CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="resultados.csv",
                    mime="text/csv"
                )

    with right:
        st.subheader("Detalles")
        if st.session_state.focus_idx is None:
            st.markdown("<div class='detail-card small-muted'>Haz clic en una oración para ver sus detalles aquí.</div>", unsafe_allow_html=True)
        else:
            sel = next((r for r in st.session_state.results if r["idx"] == st.session_state.focus_idx), None)
            if sel:
                label_txt = "Injusta (1)" if sel["label"] == 1 else "Justa (0)"
                icon = "🔴" if sel["label"] == 1 else "🟢"
                st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
                st.markdown(f"**{icon} Oración #{sel['idx']}**")
                st.write(sel["text"])
                st.markdown("---")
                st.markdown(f"**Clase:** {label_txt}")
                st.markdown(f"**Probabilidad:** {sel['prob']:.3f}")
                sim = sel.get("similar_text") or ""
                if sim:
                    st.markdown("**Texto similar (disparador):**")
                    st.code(sim)
                st.markdown("---")
                st.markdown("**Comentario** (opcional)")
                note = st.text_area("Escribe tu comentario", key=f"note_{sel['idx']}", value=st.session_state.notes.get(sel["idx"], ""))
                if st.button("Guardar comentario", key=f"save_{sel['idx']}"):
                    st.session_state.notes[sel["idx"]] = note
                    st.success("Comentario guardado.")
                st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.notes:
            st.markdown("#### Comentarios guardados")
            for k, v in st.session_state.notes.items():
                st.markdown(f"- **#{k}**: {v if v else '_(sin texto)_'}")
