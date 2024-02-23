import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Cargar el conjunto de datos de California Housing
california = datasets.fetch_california_housing()
X = california.data
Y = california.target

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Configurar el modelo LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

d_train = lgb.Dataset(x_train, label=y_train)

# Entrenar el modelo LightGBM
num_round = 300
bst = lgb.train(params, d_train, num_round)

# Realizar predicciones en el conjunto de prueba
y_pred = bst.predict(x_test, num_iteration=bst.best_iteration)

# Calcular la precisión del modelo
precision = np.corrcoef(y_pred, y_test)[0, 1]
print('Precisión del modelo (correlación):', precision)

# Visualizar las predicciones y los valores reales
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Valores Reales vs. Predicciones")
plt.show()
