import numpy as np
import sklearn.metrics

rlist = np.random.randint(100, size=10)
rlist
from numpy.core.multiarray import array
avg = np.mean(rlist)  # Calcular el promedio usando la función mean de NumPy
print("Lista de números:", rlist)
print("Promedio:", avg)




v1 = np.array([1, 2, 3, 4])

# Calcula el logaritmo de cada elemento del vector
print("el logaritmo =", np.log(v1))

# Calcula la media de todos los elementos del vector
print("la media =", np.mean(v1))

# Multiplica todos los elementos de un vector por un escalar
print("por dos =", 2 * v1)

a = 2.0
type(v1), type(a)
# Imprime los tipos de datos de 'v1' y 'a'
print("Tipo de 'v1':", type(v1))
print("Tipo de 'a':", type(a))

#Task 01. Accuracy
import numpy as np
from sklearn.metrics import accuracy_score

# Generar datos aleatorios para las etiquetas reales y predichas
t1_actual = np.random.randint(2, size=20)
t1_predicted = np.abs(t1_actual * (np.random.random(size=20) > (np.random.random() * 0.9 + 0.05)).astype(int))

# Imprimir las etiquetas reales y predichas
print("Etiquetas reales:   ", ", ".join([str(i) for i in t1_actual]))
print("Etiquetas predichas:", ", ".join([str(i) for i in t1_predicted]))

# Calcular la precisión utilizando accuracy_score
accuracy = accuracy_score(t1_actual, t1_predicted)

# Imprimir la precisión calculada
print("Precisión:", accuracy)


#Task 2: Sensitivity
import numpy as np
from sklearn.metrics import recall_score

# Generar datos aleatorios para las etiquetas reales y predichas
t2_predicted = np.random.randint(2, size=20)
t2_actual = np.random.randint(2, size=20)
t2_predicted[np.argwhere(t2_actual == 1)[0][0]] = 0

# Imprimir las etiquetas reales y predichas
print("Etiquetas reales:   ", ", ".join([str(i) for i in t2_actual]))
print("Etiquetas predichas:", ", ".join([str(i) for i in t2_predicted]))

# Calcular la métrica de sensibilidad utilizando recall_score
tpr = recall_score(t2_actual, t2_predicted)

# Redondear la métrica a tres decimales
tpr = round(tpr, 3)

# Imprimir la métrica de sensibilidad calculada
print("Métrica de Sensibilidad:", tpr)


#Task 3: Evaluation in New York City Taxi Trip Duration Kaggle Competition