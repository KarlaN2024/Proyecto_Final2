import streamlit as st
import importlib

# Configuración de la página
st.set_page_config(page_title="Proyecto Final", layout="wide")

# Título de la página
st.title("Proyecto Final")
st.write("Bienvenido a la aplicación del proyecto final. Aquí exploraremos un dataset de Airbnb de Nueva York a través de diferentes módulos: Análisis Exploratorio, Pruebas de Hipótesis y Modelos de Predicción.")

# Menú de selección de páginas
st.sidebar.title("Menú")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["Análisis Exploratorio de Datos (EDA)", "Pruebas de Hipótesis", "Modelo Predictivo"]
)

# Condiciones para cargar el contenido según la página seleccionada
if page == "Análisis Exploratorio de Datos (EDA)":
    # Importar el módulo EDA dinámicamente
    eda_module = importlib.import_module("pages.1_💡_EDA.py")  # Mantener el nombre original del archivo

elif page == "Pruebas de Hipótesis":
    # Aquí importas y ejecutas tu código de Hipótesis
    hipotesis_module = importlib.import_module("pages.2_💡_Hipotesis.py")

elif page == "Modelo Predictivo":
    # Aquí importas y ejecutas tu código de Modelo
    modelo_module = importlib.import_module("pages.3_💡_Modelo.py")

# Agregar las imágenes como lo hacías antes si es necesario
from PIL import Image

@st.cache_resource
def load_images():
    eda_image = Image.open("utils/EDA.png")
    hipotesis_image = Image.open("utils/hipotesis.png")
    modelo_image = Image.open("utils/ML.png")
    return eda_image, hipotesis_image, modelo_image

eda_image, hipotesis_image, modelo_image = load_images()

# Sección: EDA
st.header("Análisis Exploratorio de Datos (EDA)")
with st.container():
    st.image(eda_image, caption="Exploración inicial del dataset.", width=250)
    st.write("La sección de EDA permite visualizar y analizar las características principales del dataset utilizado para este proyecto. Incluye estadísticas descriptivas y gráficas que ayudan a entender la estructura y patrones de los datos.")

# Sección: Hipótesis
st.header("Pruebas de Hipótesis")
with st.container():
    st.image(hipotesis_image, caption="Análisis estadístico avanzado.", width=250)
    st.write("En esta sección, se evalúan diferentes hipótesis basadas en los datos del dataset. Utilizamos técnicas estadísticas como ANOVA y pruebas t para validar supuestos y obtener conclusiones significativas.")

# Sección: Modelo
st.header("Modelo Predictivo")
with st.container():
    st.image(modelo_image, caption="Predicción de precios con Machine Learning.", width=250)
    st.write("La sección de modelo predictivo utiliza un modelo de Machine Learning para predecir el precio de los apartamentos basados en sus características. También se muestran las métricas de desempeño del modelo, como MAE, MSE y R² Score.")

# Mensaje final
st.markdown("---")
st.write("Explora cada sección a través de las opciones del menú para obtener una experiencia completa.")
