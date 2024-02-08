from sklearn import svm
import numpy as np

# Datos de entrenamiento
X_entrenamiento = np.array([[1], [2], [3], [4], [6]])
Y_entrenamiento = np.array([2, 4, 5, 4, 7])

# Datos de prueba
X_prueba = np.array([[5], [7], [8]])
Y_prueba = np.array([5, 8, 9])

algoritmo= svm.SVR()
# Entrena el algoritmo con los datos de entrada y salida
algoritmo.fit(X_entrenamiento,Y_entrenamiento)
algoritmo.predict(X_prueba)

import matplotlib.pyplot as plt

# Graficar los datos de entrenamiento
plt.scatter(X_entrenamiento, Y_entrenamiento, color='blue', label='Entrenamiento')

# Graficar los datos de prueba
plt.scatter(X_prueba, Y_prueba, color='red', label='Prueba')

# Graficar la predicción del modelo
y_pred = algoritmo.predict(X_prueba)
plt.plot(X_prueba, y_pred, color='green', label='Predicción')

# Configuraciones adicionales para la visualización
plt.title('Support Vector Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

presicion= algoritmo.score(X_prueba,Y_prueba)
print(presicion)
# Imprimir el Mean Squared Error (MSE)
#mse = mean_squared_error(Y_prueba, y_pred)
#print(f'Mean Squared Error (MSE): {mse}')
