from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Cargar los datos
df = pd.read_csv("data/AB_NYC_2019.csv")

# Filtrar y preparar los datos
df = df[(df['price'] > 0) & (df['price'] <= 1000)]
df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=True)

X = df[['neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan', 
        'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island', 
        'room_type_Private room', 'room_type_Shared room']]
y = df['price']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(model_rf, 'models/random_forest.pkl')

# Realizar predicciones
y_pred = model_rf.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar las métricas de evaluación
st.write(f"**Error Absoluto Medio (MAE):** {mae:.2f}")
st.write(f"**Error Cuadrático Medio (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# Gráfico: Valores Reales vs. Predicciones
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.title("Valores Reales vs. Predicciones")
plt.xlabel("Valores Reales de Precio")
plt.ylabel("Valores Predichos de Precio")
plt.grid(True)
st.pyplot(plt)

# Filtrar los valores por debajo de la media
mean_price = y_test.mean()
y_pred_below_mean = y_pred[y_pred < mean_price]
y_test_below_mean = y_test.loc[y_pred < mean_price]

# Calcular porcentaje de valores por debajo de la media
percentage_below_mean = len(y_pred_below_mean) / len(y_test) * 100

# Mostrar porcentaje de valores por debajo de la media
st.write(f"**Porcentaje de predicciones por debajo de la media del precio:** {percentage_below_mean:.2f}%")
