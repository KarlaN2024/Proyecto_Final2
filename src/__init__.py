"""
Módulo para análisis estadístico y pruebas de hipótesis basado en datos de Airbnb.
"""
# Configuración global para gráficos
import seaborn as sns
import matplotlib.pyplot as plt

# Importar funciones principales de hipótesis
from .hipotesis import (
    hipotesis1,
    hipotesis2,
    hipotesis3,
    hipotesis4,
    hipotesis5
)

# Importar funciones de análisis exploratorio de datos (EDA)
from .eda import (
    panoramaGeneral,
    resumenEstadistico,
    variablesNum,
    variablesCat,
    correlación
)

# Funciones para cargar y preparar el dataset (separado del tratamiento de datos)
def cargar_dataset():
    """
    Función para cargar el dataset base.
    """
    dataset_path = "data/dataset.csv"  # Ruta al dataset
    df = pd.read_csv(dataset_path)
    return df

# Función para tratar los valores nulos y los outliers
def tratar_datos(df):
    """
    Rellena valores nulos y recorta outliers en el dataset.
    """
    df = df.copy()
    df['name'] = df['name'].fillna('Desconocido')
    df['host_name'] = df['host_name'].fillna('Desconocido')
    df['last_review'] = df['last_review'].fillna('Sin reseñas')
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

    for col in ['price', 'minimum_nights', 'reviews_per_month', 'calculated_host_listings_count']:
        limite_superior = df[col].quantile(0.99)
        df[col] = np.clip(df[col], None, limite_superior)

    return df

# Cargar dataset automáticamente
try:
    df = cargar_dataset()
    df = tratar_datos(df)
    print("Dataset cargado y preparado.")
except Exception as e:
    print(f"Error al cargar el dataset: {e}")

# Configuración global para los gráficos
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
