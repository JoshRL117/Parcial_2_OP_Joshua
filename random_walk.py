import numpy as np
#random_value = np.random.normal(mu, sigma)
def step_calculation(x):
    x_n=np.array(x)
    mu=0
    stddev=1
    random_value = np.random.normal(mu, stddev)
    return x_n + random_value

def testfunction(x):
    return x[0]**2 + x[1]**2
def randomwalk(x,f,stop):
    x=np.array(x)
    xmejor=x
    cont=0
    while(stop > cont):
        x_nuevo=step_calculation(x)
        if f(x_nuevo)< f(xmejor):
            xmejor=x_nuevo
        cont+=1
    return xmejor

def hillclimbing(x,f,stop):
    x=np.array(x)
    xactual=x
    climbing=[]
    climbing.append(xactual)
    cont=0
    while(stop > cont):
        x_nuevo=step_calculation(x)
        if f(x_nuevo)< f(xactual):
            xactual=x_nuevo
            climbing.append(xactual)
        cont+=1
    return xactual


if __name__ == "__main__": 

    # Definir valores de prueba para el punto inicial y el número de iteraciones
    x_inicial = [1, 0]  # Punto inicial
    stop = 100  # Número de iteraciones

    # Probar el algoritmo de randomwalk
    print("Random Walk:")
    print("Punto inicial:", x_inicial)
    print("Mejor punto encontrado:", randomwalk(x_inicial, testfunction, stop))

    # Probar el algoritmo de hillclimbing
    print("\nHill Climbing:")
    print("Punto inicial:", x_inicial)
    print("Mejor punto encontrado:", hillclimbing(x_inicial, testfunction, stop))