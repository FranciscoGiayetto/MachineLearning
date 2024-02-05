import pandas as pd
import numpy as np

#data= np.array([['','col1','col2'],['fila1',1,2],['fila2',3,4]])
#print( pd.DataFrame(data= data[1:,1:],index=data[1:,0] , columns=data[0,1:] ))

df= pd.DataFrame({'A':range(6), 'B':np.random.randint(5,size=6)})
print(df)

#forma del dataframe
print(df.shape)

#estadisticas
print(df.describe())

#media
print(df.mean())

#correlacion
print(df.corr())

#selecionar columna
print(df['A'])

#abrir archivos
#DF =  pd.read_csv('archivo.csv')

#verificar si hay datos nulos
print(df.isnull())