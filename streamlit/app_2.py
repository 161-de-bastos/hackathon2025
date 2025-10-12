import os
import requests
import streamlit as st
import pandas as pd
from html import escape

# -------------------------------------------------------------------
# Configuración
# Define en tu entorno: API_URL = "https://tu-fastapi.tudominio.com"
# -------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Detector de Cláusulas Injustas", layout="wide")

# Estado
if "results" not in st.session_state:
    st.session_state.results = []
if "focus_idx" not in st.session_state:
    st.session_state.focus_idx = None
if "notes" not in st.session_state:
    st.session_state.notes = {}

# -------------------------------------------------------------------
# Estilos
# -------------------------------------------------------------------
st.markdown("""
<style>
h1, h2, h3, h4 { color: #7a0b0b; }
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
.s-injusta { background: #fde7e7; color: #7a0b0b; font-weight: 600; }
.detail-card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 10px;
  padding: 12px;
  background: #fff;
}
.small-muted { color: #6b7280; font-size: 0.9rem; }
.divider { margin-top: 15px; margin-bottom: 15px; border-bottom: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Layout principal
# -------------------------------------------------------------------
st.title("⚖️ Clasificador de cláusulas: Justas / Injustas")

with st.container():
    left, right = st.columns([2,1], gap="large")

    # ------------------- COLUMNA IZQUIERDA -------------------
    with left:
        st.subheader("Texto de entrada")
        demo = """As such, the Services may change from time to time, at our discretion.
We may stop (permanently or temporarily) providing the Services or any features.
We also retain the right to create limits on use at our sole discretion.
We may remove content or terminate users without liability to you."""
        txt = st.text_area("Pega el texto a analizar", value=demo, height=180)

        if st.button("🔍 Analizar", type="primary", use_container_width=True):
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

        # ------------------- Resultados -------------------
        if st.session_state.results:
            st.subheader("Resultados por oración")

            for item in st.session_state.results:
                idx = item["idx"]
                label = item["label"]  # 0=justa, 1=injusta
                text = escape(item["text"])
                css = "s-injusta" if label == 1 else "s-justa"
                icon = "🔴" if label == 1 else "🟢"
                # Cada oración como botón
                if st.button(f"{icon} {text}", key=f"chip_{idx}", use_container_width=True):
                    st.session_state.focus_idx = idx

            # Tabla + Exportación
            with st.expander("📄 Ver tabla y exportar"):
                df = pd.DataFrame(st.session_state.results)
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "💾 Descargar CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="resultados.csv",
                    mime="text/csv"
                )

    # ------------------- COLUMNA DERECHA -------------------
    with right:
        st.subheader("Detalles de la oración")
        if st.session_state.focus_idx is None:
            st.markdown("<div class='detail-card small-muted'>Haz clic en una oración para ver detalles aquí.</div>", unsafe_allow_html=True)
        else:
            sel = next((r for r in st.session_state.results if r["idx"] == st.session_state.focus_idx), None)
            if sel:
                label_txt = "Injusta (1)" if sel["label"] == 1 else "Justa (0)"
                icon = "🔴" if sel["label"] == 1 else "🟢"

                st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
                st.markdown(f"**{icon} Oración #{sel['idx']}**")
                st.write(sel["text"])
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown(f"**Clase:** {label_txt}")
                st.markdown(f"**Probabilidad:** {sel['prob']:.3f}")

                sim = sel.get("similar_text") or ""
                if sim:
                    st.markdown("**Texto similar (disparador):**")
                    st.code(sim)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("**Comentario (opcional)**")
                note = st.text_area("Escribe tu comentario", key=f"note_{sel['idx']}", value=st.session_state.notes.get(sel["idx"], ""))
                if st.button("💬 Guardar comentario", key=f"save_{sel['idx']}"):
                    st.session_state.notes[sel["idx"]] = note
                    st.success("Comentario guardado.")
                st.markdown("</div>", unsafe_allow_html=True)

        # Mostrar comentarios guardados
        if st.session_state.notes:
            st.markdown("#### 🗒️ Comentarios guardados")
            for k, v in st.session_state.notes.items():
                st.markdown(f"- **#{k}**: {v if v else '_(sin texto)_'}")

