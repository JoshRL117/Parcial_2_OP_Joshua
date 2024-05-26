import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#Funciones 
def himmelblau(p):
    return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2
def testfunction(x):
    return x[0]**2 + x[1]**2
def sphere(x):
    value=0
    for i in range(len(x)):
        value+= x[i]**2
    return value

def rastrigin(x, A=10):
    n = len(x)
    suma_ras = 0
    for i in range(n):
        suma_ras += x[i]**2 - A * np.cos(2 * np.pi * x[i])
    return A * n + suma_ras
def rosenbrook(x):
    n=len(x)#Esta es la dimension del problema
    evaluacion=0
    for i in range(n):
        evaluacion+=((100 * (x[i] +1 - (x[i]**2))**2) + (1 -x[i])**2)
    return evaluacion
import numpy as np 

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
    for i in range(len(m[0])):
        suma = sum(p[i] for p in m)
        v = suma / len(m)
        centro.append(v)
    return np.array(centro)
def stopcondition(simplex, xc, f):
    value = 0
    n = len(simplex)
    for i in range(n):
        value += (((f(simplex[i]) - f(xc))**2) / (n + 1))
    return np.sqrt(value)

# Función principal que implementa el algoritmo Nelder-Mead en n dimensiones
def neldermeadmead(gamma, beta, epsilon, initial_point, funcion):
    cont = 1
    mov = []
    simplex = create_simplex(initial_point)
    mov.append(simplex)
    best, secondworst, worst = findpoints(simplex, funcion)
    indices = [best, secondworst, worst]
    indices.remove(worst)
    centro = xc_calculation(simplex, indices)
    x_r = (2 * centro) - simplex[worst]
    x_new = x_r
    if funcion(x_r) < funcion(simplex[best]): 
        x_new = ((1 + gamma) * centro) - (gamma * simplex[worst])
    elif funcion(x_r) >= funcion(simplex[worst]):
        x_new = ((1 - beta) * centro) + (beta * simplex[worst])
    elif funcion(simplex[secondworst]) < funcion(x_r) and funcion(x_r) < funcion(simplex[worst]):
        x_new = ((1 - beta) * centro) - (beta * simplex[worst])
    simplex[worst] = x_new
    mov.append(np.copy(simplex))
    stop = stopcondition(simplex, centro, funcion)
    while stop >= epsilon:
        stop = 0
        best, secondworst, worst = findpoints(simplex, funcion)
        indices = [best, secondworst, worst]
        indices.remove(worst)
        centro = xc_calculation(simplex, indices)
        x_r = (2 * centro) - simplex[worst]
        x_new = x_r
        if funcion(x_r) < funcion(simplex[best]):
            x_new = ((1 + gamma) * centro) - (gamma * simplex[worst])
        elif funcion(x_r) >= funcion(simplex[worst]):
            x_new = ((1 - beta) * centro) + (beta * simplex[worst])
        elif funcion(simplex[secondworst]) < funcion(x_r) and funcion(x_r) < funcion(simplex[worst]):
            x_new = ((1 + beta) * centro) - (beta * simplex[worst])
        simplex[worst] = x_new
        stop = stopcondition(simplex, centro, funcion)
        print(stop)
        mov.append(np.copy(simplex))
        cont+=1
    return simplex[best], mov

test=np.array([[2,3],[3,2],[3.5,3.5]])
initialpoint=[2, 1.5, 3, -1.5, -2]
i_p2=[-2,-2,-2]
escalar=1
gamma=1.1
b=0.1
e=0.5
best, extra= (neldermeadmead(gamma,b,e,initialpoint,rosenbrook))
print(best)

