import numpy as np 
import matplotlib.pyplot as plt
''' 
def shekel(x,p,a,w):
    for n in range(len(p)):
'''
#class nombres(Enum)
#Esta clase es la de las funciones objetivo sin restricciones 
import numpy as np
import matplotlib.pyplot as plt

class objective_f_nr:
    def __init__(self, data):
        self.data = np.array(data)
        self.dim = len(data)
    
    def himmelblau(self, p):
        return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2
    
    def testfunction(self, x):
        return x[0]**2 + x[1]**2
    
    def sphere(self, x):
        return np.sum(np.square(x))

    def rastrigin(self, x, A=10):
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def rosenbrock(self, x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def beale(self, x):
        return ((1.5 - x[0] + x[0] * x[1])**2 +
                (2.25 - x[0] + x[0] * x[1]**2)**2 +
                (2.625 - x[0] + x[0] * x[1]**3)**2)
    
    def goldstein(self, x):
        part1 = (1 + (x[0] + x[1] + 1)**2 * 
                 (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2))
        part2 = (30 + (2 * x[0] - 3 * x[1])**2 * 
                 (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))
        return part1 * part2

    def boothfunction(self, x):
        return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

    def bunkinn6(self, x):
        return 100 * np.sqrt(np.abs(x[1] - 0.001 * x[0]**2)) + 0.01 * np.abs(x[0] + 10)

    def matyas(self, x):
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    def levi(self, x):
        part1 = np.sin(3 * np.pi * x[0])**2
        part2 = (x[0] - 1)**2 * (1 + np.sin(3 * np.pi * x[1])**2)
        part3 = (x[1] - 1)**2 * (1 + np.sin(2 * np.pi * x[1])**2)
        return part1 + part2 + part3
    
    def threehumpcamel(self, x):
        return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6) / 6 + x[0] * x[1] + x[1]**2

    def easom(self, x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

    def crossintray(self, x):
        op = np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))
        return -0.0001 * (op + 1)**0.1

    def eggholder(self, x):
        op1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47))))
        op2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
        return op1 + op2

    def holdertable(self, x):
        op = np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))
        return -op

    def mccormick(self, x):
        return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1

    def schaffern2(self, x):
        numerator = np.sin(x[0]**2 - x[1]**2)**2 - 0.5
        denominator = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
        return 0.5 + numerator / denominator

    def schaffern4(self, x):
        num = np.cos(np.sin(np.abs(x[0]**2 - x[1]**2))) - 0.5
        den = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
        return 0.5 + num / den

    def styblinskitang(self, x):
        return np.sum(x**4 - 16 * x**2 + 5 * x) / 2
    
    def show(self, name):
        if self.dim != 2:
            print('ERROR: Visualization is only supported for 2D functions.')
            return
        
        functions = {
            'himmelblau': self.himmelblau,
            'beale': self.beale,
            'goldstein': self.goldstein,
            'booth': self.boothfunction,
            'bunkinn6': self.bunkinn6,
            'matyas': self.matyas,
            'levi': self.levi,
            'threehumpcamel': self.threehumpcamel,
            'easom': self.easom,
            'crossintray': self.crossintray,
            'eggholder': self.eggholder,
            'holdertable': self.holdertable,
            'mccormick': self.mccormick,
            'schaffern2': self.schaffern2,
            'schaffern4': self.schaffern4
        }
        
        if name not in functions:
            print(f'ERROR: {name} is not in the package.')
            return
        
        func = functions[name]
        
        x = np.linspace(self.data[0][0], self.data[0][1], 400)
        y = np.linspace(self.data[1][0], self.data[1][1], 400)
        X, Y = np.meshgrid(x, y)
        Z = np.array([func([x, y]) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
        
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax.set_title('3D Surface Plot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(X, Y, Z, cmap='viridis')
        fig.colorbar(contour, ax=ax2, orientation='vertical')
        ax2.set_title('2D Contour Plot')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        plt.show()

# Example usage
data = [[-5, 5], [-5, 5]]
obj = objective_f_nr(data)
obj.show('beale')
