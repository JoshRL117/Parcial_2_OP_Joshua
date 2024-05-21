import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#El fi de este codigo era el de desarrollar shekel de forma generica
def shekel(x, a=None, c=None):
    if a is None:# Esto lo hice para que el usuario pueda meter los pesos que guste, si no se ponen estos
        a = np.array([
            [4.0, 4.0, 4.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0, 8.0],
            [6.0, 6.0, 6.0, 6.0],
            [3.0, 7.0, 3.0, 7.0],
            [2.0, 9.0, 2.0, 9.0],
            [5.0, 5.0, 3.0, 3.0],
            [8.0, 1.0, 8.0, 1.0],
            [6.0, 2.0, 6.0, 2.0],
            [7.0, 3.6, 7.0, 3.6]
        ])
    if c is None:
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])#Lo mismo que con a
    
    m = len(c)
    s = 0
    for i in range(m):
        s -= 1 / (np.dot(x - a[i, :2], x - a[i, :2]) + c[i])#Esta es la sumatoria dado m, que seria el numero de terminos en la suma
    return s

x = np.linspace(0, 10, 400)
y = np.linspace(0, 10, 400)
X, Y = np.meshgrid(x, y)
Z = np.array([shekel(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
Z=Z*-1 #Es para que la grafica se parezca a la de la actividad
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('Shekel Function')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


