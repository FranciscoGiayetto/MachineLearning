# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Generar datos de ejemplo
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de árbol de decisión para regresión
regressor = DecisionTreeRegressor(max_depth=5)

# Entrenar el modelo
regressor.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = regressor.predict(X_test)

# Calcular el error cuadrático medio en el conjunto de prueba
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio en el conjunto de prueba: {mse:.4f}')

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, s=20, edgecolor="black", c="darkorange", label="datos reales")
plt.plot(X_test, y_pred, color="cornflowerblue", label="predicciones", linewidth=2)
plt.xlabel("Datos de entrada")
plt.ylabel("Variable de salida")
plt.title("Árbol de decisión para regresión")
plt.legend()
#plt.show()

print('Precisión del modelo: ', regressor.score(X_test, y_test))
