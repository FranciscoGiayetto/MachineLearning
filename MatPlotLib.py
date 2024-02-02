import matplotlib.pyplot as plt

#DIAGRAMA DE LINEAS
a = [1,2,3,4]
b = [5,6,7,8]
c = [5,6,7,8]
d = [1,2,3,4]

plt.plot(a,b,color='red', linewidth=5, label = 'linea1' )
plt.plot(c,d,color='blue', linewidth=5, label = 'linea2' )

plt.title('Ejemplo diagrama lineas')
plt.xlabel('Valores de x')
plt.ylabel('Valores de y')

# Agregar una leyenda a la izquierda superior
plt.legend()
# Mostrar el gráfico con los ejes etiquetados
plt.grid()
plt.show()

#DIAGRAMA DE BARRAS
a = [1,2,3,4]
b = [5,6,7,8]
c = [5,6,7,8]
d = [1,2,3,4]

plt.bar(a,b,color='red', linewidth=5, label = 'linea1' )
plt.bar(c,d,color='blue', linewidth=5, label = 'linea2' )

plt.title('Ejemplo diagrama lineas')
plt.xlabel('Valores de x')
plt.ylabel('Valores de y')

# Agregar una leyenda a la izquierda superior
plt.legend()
# Mostrar el gráfico con los ejes etiquetados
plt.grid()
plt.show()


#Graficos de dispercion con scatter