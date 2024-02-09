import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

datos= pd.read_csv('./MarathonData.csv')

#Paso los datos a numeros
datos['Wall21']= pd.to_numeric(datos['Wall21'], errors='coerce') 

#print(datos.hist())
#print(datos.info())

# Elimino columnas inecesarias
datos=datos.drop(columns=['id'])
datos=datos.drop(columns=['Name'])
datos=datos.drop(columns=['Marathon'])
datos=datos.drop(columns=['CATEGORY'])

#print(datos)

#Muestro los datos nulos por columna
#print(datos.isna().sum())

#Cambio 74 daots por 0 y elimino el resto que son nulos
datos['CrossTraining']= datos['CrossTraining'].fillna(0)
datos = datos.dropna(how='any')

#print(datos.isna().sum())

#Tengo q pasar los datos de texto a numero
print(datos['CrossTraining'].unique())

valores_cross= {'CrossTraining': { 'ciclista 1h':1, 'ciclista 3h':2, 'ciclista 4h':3, 'ciclista 5h':4, 'ciclista 13h':5}}
datos.replace(valores_cross, inplace=True) 

#print(datos)

print(datos['Category'].unique())
valores_category= {'Category': { 'MAM':1, 'M45':2, 'M40':3, 'M50':4, 'M55':5, 'WAM':6}}
datos.replace(valores_category, inplace=True) 

print(datos)

plt.scatter(x= datos['Wall21'], y=datos['MarathonTime'])
plt.xlabel("Wall21")
plt.ylabel("MarathonTime")
plt.savefig('scatter_plot.png')

datos_entrenamiento= datos.sample(frac=0.8,random_state=0) #Separo entre entrenamiento y test
datos_test=datos.drop(datos_entrenamiento.index)

etiquetas_entrenamiento= datos_entrenamiento.pop('MarathonTime')
etiquetas_test= datos_test.pop('MarathonTime')

modelo= LinearRegression()
modelo.fit(datos_entrenamiento,etiquetas_entrenamiento)

prediccion=modelo.predict(datos_test)

precision = modelo.score(datos_test, etiquetas_test)
error= np.sqrt(mean_squared_error(etiquetas_test, prediccion))
print('Eror porcentual ', error*100, ' Presicion ', precision)