import numpy as np

def boothfunction(x):
    return ((x[0] + 2 * (x[1]) - 7) ** 2) + ((2 * x[0]) + x[1] - 5) ** 2

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
    delta = delta / alpha
    return delta

def hookejeeves(initialpoint, d, alpha, e, f):
    cont = 0
    x_inicial = np.array(initialpoint)
    delta = np.array(d)
    x_anterior = x_inicial
    x_mejor, flag = movexploratory(x_inicial, delta, f)

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
e=0.0001
#print(patternmove(xk,x_inicial))
print(hookejeeves(x_inicial,delta,alpha,e,boothfunction))