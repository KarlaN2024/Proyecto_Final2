from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import joblib

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

# Entrenar el modelo de regresión lineal
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(model_lr, 'models/linear_regression.pkl')

# Evaluación del modelo en el conjunto de prueba
y_pred = model_lr.predict(X_test)

# Calcular las métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar las métricas de evaluación
print(f"Evaluación del modelo de Regresión Lineal:")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")



