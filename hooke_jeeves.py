import numpy as np

def boothfunction(x):
    return ((x[0] + 2 * (x[1]) - 7) ** 2) + ((2 * x[0]) + x[1] - 5) ** 2

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

def movexploratory(basepoint, delta, f):
    nextpoint = []
    coordanatess = [basepoint]
    newvalue = True
    #Creacion de las coordenadas 
    for i in range(len(basepoint)):
        point = basepoint.copy()
        point2 = basepoint.copy()
        point[i] += delta[i]
        point2[i] -= delta[i]
        coordanatess.append(point)
        coordanatess.append(point2)
    
    #evaluacion de las coordenadas
    for coordenate in coordanatess:
        nextpoint.append(f(coordenate))
    
    #Busqueda del min 
    minum = np.argmin(nextpoint)
    if (coordanatess[minum] == basepoint).all():
        newvalue = False
    
    return coordanatess[minum], newvalue

def patternmove(currentbestpoint, lastbestpoint):
    basepoint = currentbestpoint + (currentbestpoint - lastbestpoint)
    return basepoint

def updatedelta(delta, alpha):
    new_delta = delta / alpha
    return new_delta
def hookejeeves(initialpoint, d, alpha, e, f):
    cont = 0
    x_inicial = np.array(initialpoint)
    delta = np.array(d)
    x_anterior = x_inicial
    x_mejor, flag = movexploratory(x_inicial, delta, f)
    print(x_mejor)
    while np.linalg.norm(delta) > e:
        if flag:
            x_point = patternmove(x_mejor, x_anterior)
            x_mejor_nuevo, flag = movexploratory(x_point, delta, f)
        else:
            delta = updatedelta(delta, alpha)
            x_mejor, flag = movexploratory(x_mejor, delta, f)
            x_point = patternmove(x_mejor, x_anterior)
            x_mejor_nuevo, flag = movexploratory(x_point, delta, f)
        #Son dos subprocersos
        if f(x_mejor_nuevo) < f(x_mejor):
            flag = True
            x_anterior = x_mejor
            x_mejor = x_mejor_nuevo
        else:
            flag = False

        cont += 1
        print(x_mejor_nuevo)
        print(f(x_mejor_nuevo))
    print("Num de iteraciones {}".format(cont))
    return x_mejor_nuevo
#main
x_inicial=([-5,-2.5])
delta=([0.5,0.25])
alpha=2
#xk,newv=(movexploratory(x_inicial,delta,boothfunction))
e=0.01
#print(patternmove(xk,x_inicial))
#print(hookejeeves(x_inicial,delta,alpha,e,boothfunction))

#print(hookejeeves([-1, 1.5],[0.5,0.5],alpha,e,sphere))#sphere
#print(hookejeeves([0, 0],[0.5,0.5],alpha,e,himmelblau))
#print(hookejeeves([-2, -2, -2] ,[0.5,0.5,0.5],alpha,e,rastrigin))
print(hookejeeves([2, 1.5, 3, -1.5, -2],[0.5,0.5,0.5,0.5,0.5],alpha,e,rosenbrook))