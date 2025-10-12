import os, requests, streamlit as st
from prometheus_client import Counter, generate_latest

API_URL = os.getenv("API_URL", "http://api:8000")
REQS = Counter("frontend_requests_total", "Frontend form submits")

st.set_page_config(page_title="MANN TOS Analyzer")
st.title("Analizador de Términos y Condiciones")

text = st.text_area("Pega el texto aquí", height=250)
if st.button("Procesar") and text.strip():
    REQS.inc()
    with st.spinner("Procesando..."):
        resp = requests.post(f"{API_URL}/process", json={"text": text})
        st.write(resp.json())

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            data = generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)
def serve():
    HTTPServer(("0.0.0.0", 9100), H).serve_forever()
threading.Thread(target=serve, daemon=True).start()