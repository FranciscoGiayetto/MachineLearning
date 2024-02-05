from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import matplotlib.pyplot as plt

# Datos de entrenamiento
variablesindependientesEntrenamiento = np.array([[1], [2], [3], [4]])
variablesdependientesEntrenamiento = np.array([2, 4, 5, 4])

# Datos de prueba
variablesindependientesPrueba = np.array([[5], [6], [7]])
variablesdependientesPrueba = np.array([5, 7, 8])

# Definir el modelo de regresión lineal
modelo = linear_model.LinearRegression()

# Entrenar el modelo
modelo.fit(variablesindependientesEntrenamiento, variablesdependientesEntrenamiento)

# Predecir con el conjunto de prueba
predicciones = modelo.predict(variablesindependientesPrueba)

# Obtener la pendiente (coeficiente) y la intersección (término independiente)
a_pendiente = modelo.coef_
b_interseccion = modelo.intercept_

# Calcular la precisión (coeficiente de determinación R^2)
precision = modelo.score(variablesindependientesPrueba, variablesdependientesPrueba)

# Calcular otras métricas de rendimiento, como el error cuadrático medio (MSE) y el coeficiente de correlación (R^2)
mse = mean_squared_error(variablesdependientesPrueba, predicciones)
r2 = r2_score(variablesdependientesPrueba, predicciones)

# Imprimir resultados
print(f"Pendiente (a): {a_pendiente}")
print(f"Intersección (b): {b_interseccion}")
print(f"Precisión (R^2): {precision}")
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}")

# Graficar los datos de entrenamiento
plt.scatter(variablesindependientesEntrenamiento, variablesdependientesEntrenamiento, color='blue', label='Datos de entrenamiento')

# Graficar la recta de regresión
plt.plot(variablesindependientesPrueba, predicciones, color='red', linewidth=3, label='Recta de regresión')

# Graficar los datos de prueba
plt.scatter(variablesindependientesPrueba, variablesdependientesPrueba, color='green', label='Datos de prueba')

# Etiquetas y leyenda
plt.xlabel('Variable Independiente')
plt.ylabel('Variable Dependiente')
plt.title('Regresión Lineal Simple')
plt.legend()

# Mostrar la gráfica
plt.show()