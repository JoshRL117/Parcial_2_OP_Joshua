import numpy as np 
import matplotlib.pyplot as plt
from enum import Enum
class restriction_fuctions:
    def __init__(self,data):
        self.data=data
        self.dim=len(data)
    
    def rosenbrooklinecubic(self,x): 
        return np.array([(1 - x[0])**2 + 100 * (x[1] - (x[0]**2))**2 ])

    def rosenbrooklinecubic_restriction(self,x):
        return (((x[0] - 1  )**3 - x[1] + 1))>= 0 and (x[0] + x[1] - 2)
        
    def rosenbrookdisk(self,x): 
        return np.array([(1 - x[0])**2 + 100 * (x[1] - (x[0]**2))**2])

    def rosenbrookdisk_restriction(self,x):
        return(x[0]**2 + x[1] ** 2 )

    def mishrabird(self,x):
        return np.sin(x[1]) * np.exp((1 - np.cos(x[0]))**2) + np.cos(x[0]) * np.exp((1 - np.sin(x[1]))**2) + (x[0] - x[1])**2

    def mishabird_restriction(self,x):
        return (x[0] + 5)**2 + (x[1] + 5)**2 < 25

    def townsendmod(self,x):
        return - (np.cos((x[0] - 0.1) * x[1]))**2 - x[0] * np.sin(3 * x[0] + x[1])

    def townsendmod_restriction(self,x):
        t = np.arctan2(x[1], x[0])
        op1 = x[0]**2 + x[1]**2
        op2 = (2 * np.cos(t) - 0.5 * np.cos(2 * t) - 0.25 * np.cos(3 * t) - 0.125 * np.cos(4 * t))**2 + (2 * np.sin(t))**2
        return op1 < op2

    def gomezandlevy(self,x):
        return 4 * x[0]**2 - 2.1 * x[0]**4 + (1 / 3) * x[0]**6 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4

    def gomezandlevi_restriction(self,x):
        return -np.sin(4 * np.pi * x[0]) + 2 * np.sin(2 * np.pi * x[1])**2 <= 1.5

    def simionescu(self,x):
        return 0.1 * (x[0] * x[1])

    def simionescu_restriction(self,x):
        r_T=1
        r_S=0.2
        n=8
        angulo = np.arctan2(x[1], x[0]) 
        cosine_term = np.cos(n * angulo)
        op = (r_T + r_S * cosine_term) ** 2
        return x[1]**2 + x[1]**2 - op
    
    def show_graph(self,name):
        if self.dim != 2: 
            return 0
        else:

            funciones={
                'rosenbrookcubic':[self.rosenbrooklinecubic,self.rosenbrooklinecubic_restriction],
                'rosenbrookdisk':[self.rosenbrookdisk,self.rosenbrookdisk_restriction],
                'misharbird':[self.mishrabird,self.mishabird_restriction],
                'townsed':[self.townsendmod,self.townsendmod_restriction],
                'gomezandlevi':[self.gomezandlevy,self.gomezandlevi_restriction],
                'simionescu':[self.simionescu,self.simionescu_restriction]
            }
            if name not in funciones:
                print('ERROR: {} is not in the package.'.format(name))
                return
            f,res= funciones[name]
            x = np.linspace(self.data[0][0], self.data[0][1], 400)
            y = np.linspace(self.data[1][0], self.data[1][1], 400)
            X, Y = np.meshgrid(x, y)
            Z = np.array([f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
            Z = np.empty_like(X)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if res([X[i, j], Y[i, j]]):
                        Z[i, j] = f([X[i, j], Y[i, j]])
                    else:
                        Z[i, j] = np.nan#Esto es para el espacio en blanco
            fig, ax = plt.subplots(figsize=(8, 8))
            contour = ax.contourf(X, Y, Z, cmap='viridis')
            fig.colorbar(contour, ax=ax, orientation='vertical')
            ax.set_title('Contorno 2D')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.suptitle(' funcion ' + ' ' + name)
            plt.show()
            return 0


if __name__ == "__main__":
    data = [[-1.25, 1.25], [-1.25, 1.25]]
    functions = restriction_fuctions(data)
    functions.show_graph('simionescu')