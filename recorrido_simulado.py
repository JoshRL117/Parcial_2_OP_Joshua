import numpy as np

#Diseño del algoritmo 

def probabilidad_rs(x,u,t,f): 
    num= f(np.array(u)) - f(np.array(x))
    return num / t

def tweak(x_actual):
    x=np.array(x_actual)
    u=x.copy()
    value=np.random.uniform(-1,1)
    if value == 0: 
        return u
    else: 
        return u + value

def u_generation()

def recorrido_simulado(x_inicial,f,w,t=100,alpha=0.3):#w es el numero de iteraciones, t es la temperatura 
    x_actual=x_inicial
    best=x_inicial
    cont=0
    while cont <= w:
        return 0
    
