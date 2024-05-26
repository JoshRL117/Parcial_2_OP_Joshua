import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
#Estas funciones se hicieron para crear la animacion del punto extra
def init():
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.title('Nelder-Mead Simplex Movements')
    return []

def update(frame):
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    simplex = mov[frame]
    triangle = patches.Polygon(simplex, closed=True, fill=None, edgecolor='blue')
    ax.add_patch(triangle)
    for point in simplex:
        ax.plot(point[0], point[1], 'o', color='red')
    return [triangle]

# Funciones de prueba
def himmelblau(p):
    return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2 # (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2

def testfunction(x):
    return x[0]**2 + x[1]**2

def sphere(x):
    value = 0
    for i in range(len(x)):
        value += x[i]**2
    return value

def rastrigin(x, A=10):
    n = len(x)
    suma_ras = 0
    for i in range(n - 1):
        suma_ras += (x[i]**2) - (A * np.cos((2 * np.pi) * x[i]))
    return (A * n) - suma_ras

def rosenbrook(x):
    n = len(x)
    evaluacion = 0
    for i in range(n):
        evaluacion += (100 * (x[i] + 1 - (x[i]**2))**2) + (1 - x[i])**2
    return evaluacion

def delta1(N, scale):
    num = np.sqrt(N + 1) + N - 1
    den = N * np.sqrt(2)
    op = num / den
    return op * scale

def delta2(N, scale):
    num = np.sqrt(N + 1) - 1
    den = N * np.sqrt(2)
    op = num / den
    return op * scale

def stopcondition(simplex, xc, f):
    value = 0
    n = len(simplex)
    for i in range(n):
        value += (((f(simplex[i]) - f(xc))**2) / (n + 1))
    return np.sqrt(value)

def create_simplex(initial_point, scale=1.0):
    n = len(initial_point)
    simplex = [np.array(initial_point, dtype=float)] 
    d1 = delta1(n, scale)
    d2 = delta2(n, scale)
    for i in range(n):
        point = np.array(simplex[0], copy=True)  
        for j in range(n):
            if j == i: 
                point[j] += d1
            else:
                point[j] += d2
        simplex.append(point)
    simplex_final = np.array(simplex)
    return np.round(simplex_final, 4)

def findpoints(points, funcion):
    evaluaciones = [funcion(p) for p in points]
    worst = np.argmax(evaluaciones)
    best = np.argmin(evaluaciones)
    indices = list(range(len(evaluaciones)))
    indices.remove(worst)
    second_worst = indices[np.argmax([evaluaciones[i] for i in indices])]
    if second_worst == best:
        indices.remove(best)
        second_worst = indices[np.argmax([evaluaciones[i] for i in indices])]
    return best, second_worst, worst

def xc_calculation(x, indexs):
    m = x[indexs]
    centro = []
    for fila in m: 
        suma = np.sum(fila)
        v = suma / (len(fila))
        centro.append(v)
    return np.array(centro)

def neldermeadmead(gamma, beta, epsilon, initial_point, funcion):
    mov = []
    cont=1
    simplex = create_simplex(initial_point)
    print(simplex)
    mov.append(np.copy(simplex))
    best, secondworst, worst = findpoints(simplex, funcion)  # Indices de los 3 puntos
    indices = [best, secondworst, worst]
    indices.remove(worst)
    centro = xc_calculation(simplex, indices)
    x_r = (2 * centro) - simplex[worst]
    x_new = x_r
    if funcion(x_r) < funcion(simplex[best]): 
        x_new = ((1 - gamma) * centro) - (gamma * simplex[worst])
    elif funcion(x_r) >= funcion(simplex[worst]):
        x_new = ((1 - beta) * centro) + (beta * simplex[worst])
    elif funcion(simplex[secondworst]) < funcion(x_r) and funcion(x_r) < funcion(simplex[worst]):
        x_new = ((1 - beta) * centro) - (beta * simplex[worst])
    simplex[worst] = x_new
    mov.append(np.copy(simplex))#Esto me atoro 2 horas pipipipi 
    print(simplex)
    stop = stopcondition(simplex, centro, funcion)
    cont=1
    while stop > epsilon and cont < 100:
        stop = 0
        best, secondworst, worst = findpoints(simplex, funcion)  # Indices de los 3 puntos
        indices = [best, secondworst, worst]
        indices.remove(worst)
        centro = xc_calculation(simplex, indices)
        x_r = (2 * centro) - simplex[worst]
        x_new = x_r
        if funcion(x_r) < funcion(simplex[best]):  # Expansion gamma > 1 
            x_new = ((1 - beta) * centro) - (gamma * simplex[worst])
        elif funcion(x_r) >= funcion(simplex[worst]):  # Contraccion b < 0
            x_new = ((1 - beta) * centro) + (beta * simplex[worst])
        elif funcion(simplex[secondworst]) < funcion(x_r) and funcion(x_r) < funcion(simplex[worst]):  # Contraccion b > 0
            x_new = ((1 - beta) * centro) - (beta * simplex[worst])
        simplex[worst] = x_new
        stop = stopcondition(simplex, centro, funcion)
        print(stop)
        mov.append(np.copy(simplex))
        cont+=1
    print("iteraciones totales {}".format(cont))
    return simplex[best], mov

gamma_himmelblau = 1.1
beta_himmelblau = 0.1
initial_point_himmelblau = [0, 0]
epsilon = 0.5
print(himmelblau([0,0]))
best, mov = neldermeadmead(gamma_himmelblau, beta_himmelblau, epsilon, initial_point_himmelblau, himmelblau)
print( best)

# Crear el fondo de la función
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

# Graficar la animación de los simplex
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
contour_himmelblau = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour_himmelblau)
plt.title('Nelder-Mead Simplex Movements for Himmelblau')


ani_himmelblau = FuncAnimation(fig, update, frames=len(mov), init_func=init, blit=True, repeat=False)
#plt.show()

