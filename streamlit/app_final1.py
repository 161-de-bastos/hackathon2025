import os
import streamlit as st
import pandas as pd
from html import escape
import random
import math

# -------------------------------------------------------------------
# Configuración general
# -------------------------------------------------------------------
st.set_page_config(page_title="Detector de Cláusulas Injustas", layout="wide")

# -------------------------------------------------------------------
# Estado de sesión
# -------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []
if "focus_idx" not in st.session_state:
    st.session_state.focus_idx = None
if "notes" not in st.session_state:
    st.session_state.notes = {}
if "page" not in st.session_state:
    st.session_state.page = 1
if "filter" not in st.session_state:
    st.session_state.filter = "Todas"

# -------------------------------------------------------------------
# Función dummy simulando la API FastAPI (/v1/predict)
# -------------------------------------------------------------------
def fake_predict(text: str):
    """Simula el comportamiento del endpoint /v1/predict"""
    sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
    simulated_results = []
    for i, s in enumerate(sentences):
        label = random.choice([0, 1])
        prob = round(random.uniform(0.4, 0.95), 3)
        similar_text = None
        if label == 1:
            similar_text = random.choice([
                "El servicio puede cambiar unilateralmente.",
                "El proveedor puede terminar sin previo aviso.",
                "El usuario no tiene derecho a reembolso.",
                "La empresa puede modificar las condiciones sin consentimiento."
            ])
        simulated_results.append({
            "idx": i,
            "text": s,
            "label": label,
            "prob": prob,
            "similar_text": similar_text
        })
    return {"sentences": simulated_results}

# -------------------------------------------------------------------
# Estilos visuales
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
    left, right = st.columns([2, 1], gap="large")

    # ------------------- COLUMNA IZQUIERDA -------------------
    with left:
        st.subheader("Texto de entrada")
        demo = """As such, the Services may change from time to time, at our discretion.
We may stop (permanently or temporarily) providing the Services or any features.
We also retain the right to create limits on use at our sole discretion.
We may remove content or terminate users without liability to you.
As such, the Services may change from time to time, at our discretion.
We may stop (permanently or temporarily) providing the Services or any features.
We also retain the right to create limits on use at our sole discretion.
We may remove content or terminate users without liability to you."""
        txt = st.text_area("Pega el texto a analizar", value=demo, height=180)

        if st.button("🔍 Analizar", type="primary", use_container_width=True):
            st.session_state.focus_idx = None
            st.session_state.notes = {}
            st.session_state.page = 1
            with st.spinner("Analizando texto..."):
                payload = fake_predict(txt)
                st.session_state.results = payload.get("sentences", [])
            if not st.session_state.results:
                st.info("No se detectaron oraciones en el texto.")

        # ------------------- Filtros y paginación -------------------
        if st.session_state.results:
            st.subheader("Resultados por oración")

            # --- Filtros horizontales ---
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Todas", use_container_width=True, type=("primary" if st.session_state.filter == "Todas" else "secondary")):
                    st.session_state.filter = "Todas"
                    st.session_state.page = 1
            with col2:
                if st.button("Justas 🟢", use_container_width=True, type=("primary" if st.session_state.filter == "Justas" else "secondary")):
                    st.session_state.filter = "Justas"
                    st.session_state.page = 1
            with col3:
                if st.button("Injustas 🔴", use_container_width=True, type=("primary" if st.session_state.filter == "Injustas" else "secondary")):
                    st.session_state.filter = "Injustas"
                    st.session_state.page = 1

            # --- Aplicar filtro ---
            filtered = st.session_state.results
            if st.session_state.filter == "Justas":
                filtered = [x for x in filtered if x["label"] == 0]
            elif st.session_state.filter == "Injustas":
                filtered = [x for x in filtered if x["label"] == 1]

            # --- Paginación ---
            n_por_pagina = st.number_input("Oraciones por página:", 5, 20, 8, step=1)
            total_items = len(filtered)
            total_paginas = max(1, math.ceil(total_items / n_por_pagina))

            if st.session_state.page > total_paginas:
                st.session_state.page = 1

            colp1, colp2, colp3 = st.columns(3)
            with colp1:
                if st.button("⬅️ Anterior", disabled=st.session_state.page <= 1):
                    st.session_state.page -= 1
            with colp2:
                st.markdown(
                    f"<div style='text-align:center;'>Página {st.session_state.page} de {total_paginas}</div>",
                    unsafe_allow_html=True
                )
            with colp3:
                if st.button("Siguiente ➡️", disabled=st.session_state.page >= total_paginas):
                    st.session_state.page += 1

            # --- Paginación efectiva ---
            start = (st.session_state.page - 1) * n_por_pagina
            end = start + n_por_pagina
            page_items = filtered[start:end]

            # --- Render de chips ---
            if page_items:
                for item in page_items:
                    idx = item["idx"]
                    label = item["label"]
                    text = escape(item["text"])
                    icon = "🔴" if label == 1 else "🟢"
                    if st.button(f"{icon} {text}", key=f"chip_{idx}", use_container_width=True):
                        st.session_state.focus_idx = idx
            else:
                st.info("No hay oraciones que coincidan con el filtro.")

            # --- Tabla + Exportación ---
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

