from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()

#print('Informacion del dataset: ', diabetes.keys())

#print('Características del dataset: ', diabetes.DESCR)

print('Cantidad de datos: ', diabetes.data.shape)

print('Nombre de las columnas: ', diabetes.feature_names)

X = diabetes.data[:, np.newaxis, 8]
Y = diabetes.target

plt.scatter(X, Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

poli_reg = PolynomialFeatures(degree=2)

x_train_poli = poli_reg.fit_transform(x_train)
x_test_poli = poli_reg.transform(x_test)

lr = linear_model.LinearRegression()

lr.fit(x_train_poli, y_train)

y_pred = lr.predict(x_test_poli)

# Corregir cómo calculas la precisión
precision = lr.score(x_test_poli, y_test)
print('Precisión del modelo: ', precision)

plt.plot(x_test, y_pred, color='red', linewidth=3)
plt.show()
