import numpy as np
rlist = np.random.randint(100, size=10)
rlist
from numpy.core.multiarray import array
avg = np.mean(rlist)  # Calcular el promedio usando la función mean de NumPy
print("Lista de números:", rlist)
print("Promedio:", avg)