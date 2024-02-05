from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

california = datasets.fetch_california_housing()

#print('Informacion del dataset: ', california.keys())

#print('Caracterizticas del dataset: ', california.DESCR)

print('Nombre de las columnas: ', california.feature_names)

X = california.data[:, 0:3]
Y = california.target
#print('Dimensiones de Y:', Y.shape)
# Utiliza train_test_split con argumento shuffle=False para asegurar la consistencia de los índices
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, shuffle=False)

# Asegúrate de que las características tengan la forma adecuada (2D)
x_train = x_train.reshape(-1, 3)
x_test = x_test.reshape(-1, 3)

lr = linear_model.LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

# Mala elección de modelo
print('Precisión del modelo: ', lr.score(x_test, y_test))

plt.scatter(X[:, 0], Y)
plt.plot(x_test[:, 0], y_pred, color='red', linewidth=3)

plt.show()
