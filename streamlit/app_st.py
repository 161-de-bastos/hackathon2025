import streamlit as st
import pandas as pd
import numpy as np

# Título de la aplicación
st.title("🚀 Mi Primera App en Streamlit")

# Sidebar
st.sidebar.header("Panel de Control")
opcion = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Inicio", "Datos", "Gráficos", "Interactivo"]
)

# Sección Inicio
if opcion == "Inicio":
    st.header("¡Bienvenido!")
    st.write("Esta es una aplicación simple hecha con Streamlit.")
    
    nombre = st.text_input("¿Cómo te llamas?")
    if nombre:
        st.success(f"¡Hola {nombre}! 👋")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Usuarios", "1,234", "+12%")
    with col2:
        st.metric("Visitas", "5,678", "+8%")
    with col3:
        st.metric("Rating", "4.8", "+0.2")

# Sección Datos
elif opcion == "Datos":
    st.header("📊 Visualización de Datos")
    
    # Crear datos de ejemplo
    df = pd.DataFrame({
        'Producto': ['A', 'B', 'C', 'D', 'E'],
        'Ventas': np.random.randint(100, 500, 5),
        'Precio': np.random.uniform(10, 100, 5).round(2)
    })
    
    st.subheader("Tabla de Datos")
    st.dataframe(df, use_container_width=True)
    
    st.subheader("Estadísticas")
    st.write(df.describe())

# Sección Gráficos
elif opcion == "Gráficos":
    st.header("📈 Gráficos")
    
    # Gráfico de línea
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    st.line_chart(chart_data)
    
    # Gráfico de área
    st.subheader("Gráfico de Área")
    st.area_chart(chart_data)

# Sección Interactiva
elif opcion == "Interactivo":
    st.header("🎮 Elementos Interactivos")
    
    edad = st.slider("¿Cuántos años tienes?", 0, 100, 25)
    st.write(f"Tienes {edad} años")
    
    color = st.color_picker("Elige tu color favorito", "#00f900")
    st.write(f"Tu color favorito es: {color}")
    
    fecha = st.date_input("Selecciona una fecha")
    st.write(f"Fecha seleccionada: {fecha}")
    
    opciones = st.multiselect(
        "¿Qué te gusta hacer?",
        ["Leer", "Deportes", "Música", "Viajar", "Cocinar"]
    )
    if opciones:
        st.write(f"Te gusta: {', '.join(opciones)}")
    
    if st.button("¡Presiona aquí!"):
        st.balloons()
        st.success("¡Genial! 🎉")

# Footer
st.divider()
st.caption("Creado con Streamlit")