from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

diabetes =  datasets.load_diabetes()
#print(diabetes)

#print('Informacion del dataset: ', diabetes.keys())

#print('Caracterizticas del dataset: ', diabetes.DESCR)

print('Cantidad de datos: ', diabetes.data.shape)

print('Nombre de las columnas: ', diabetes.feature_names)

X= diabetes.data[:,np.newaxis,2]
Y= diabetes.target



x_train,  x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

lr= linear_model.LinearRegression()

lr.fit(x_train, y_train)

y_pred=  lr.predict(x_test)

#Mala eleccion de modelo
print('Presicion del modelo: ',  lr.score(x_test, y_test))

plt.scatter(X,Y)
plt.plot(x_test,y_pred, color='red', linewidth=3)
plt.xlabel('Edad de la persona')
plt.ylabel('Valor medio')

plt.show()