import numpy as np

# Crear matrices multidimensionales
array = np.array([[1,2,3],[4,5,6]])
print(array)

# El primer parametro indica el numero de filas y el segundo la cantidad de columnas
# Crear matrices completas con 1
unos= np.ones((3,4))
print("Matriz de unos: \n",unos)

# Crear matrices completas con 0
ceros= np.zeros((3,4))
print("Matriz de ceros: \n",ceros)

# Crear una matriz vacia
vacia = np.empty((0,3)) 
print("\n Matriz vacía:\n ", vacia)

print('suma: ',array +  array) # Suma de dos matrices
print('tamaño: ', array.dtype)
print ('min: ',array.min())
print ('max: ',array.max())
print ('suma: ',array.sum())