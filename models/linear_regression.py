import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv("data/AB_NYC_2019.csv")
    return df

# Filtrar y preparar los datos
@st.cache_data
def prepare_data(df):
    # Filtrar precios válidos
    df = df[(df['price'] > 0) & (df['price'] <= 1000)]
    df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=True)
    
    X = df[['neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan', 
            'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island', 
            'room_type_Private room', 'room_type_Shared room']]
    y = df['price']
    
    return X, y

# Cargar los datos y prepararlos
df = load_data()
X, y = prepare_data(df)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model_lr, 'models/linear_regression.pkl')

# Evaluar el modelo en el conjunto de prueba
y_pred = model_lr.predict(X_test)

# Calcular las métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar las métricas de evaluación en Streamlit
st.title("Evaluación del Modelo de Regresión Lineal")
st.write("Evaluación del modelo de regresión lineal con los datos de prueba:")

st.write(f"**Error Absoluto Medio (MAE):** {mae:.2f}")
st.write(f"**Error Cuadrático Medio (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# Función para graficar la regresión
def plot_regression(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predicciones')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title('Valores Reales vs Predichos - Regresión Lineal')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Mostrar el gráfico
plot_regression(y_test, y_pred)


