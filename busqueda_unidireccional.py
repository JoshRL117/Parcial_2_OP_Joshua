import numpy as np
def funcionprueba(x1,x2):
    return x1 ** 2 + x2 **2

def funcion1(x1,x2):
    return ((x1- 10)**2 ) + ((x2 - 10)**2)

def himmelblau(p1,p2):
    return (p1**2 + p2 - 11)**2 + (p1 + p2**2 - 7)**2
#Busqueda unidireccional 
class optimizador: 
    def __init__(self,x_t,x_s,funcion,alpha):
        self.x_t=x_t
        self.x_s=x_s
        self.funcion=funcion
        self.alfa=alpha
    
    def primeraderivadanumerica(self, x_actual,f):
        delta=0.0001
        numerador= f(x_actual + delta) - f (x_actual - delta) 
        return (numerador / (2 * delta))

    def segundaderivadanumerica(self, x_actual,f):
        delta=0.0001
        numerado= f(x_actual + delta) - (2 * f (x_actual + f(x_actual- delta))) 
        return (numerado / (delta**2))

    def newton_raphson(self,x,e,f):
        cont=0
        x_actual=x
        xderiv=self.primeraderivadanumerica(x_actual,f)
        xderiv2=self.segundaderivadanumerica(x_actual,f)
        xsig= x_actual  - (xderiv/xderiv2)
        print("{} - {} / {}".format(x_actual,xderiv,xderiv2))
        print(xsig)
        x_actual=xsig
        while  (abs(self.primeraderivadanumerica(xsig,f)) > e):
            #print(" {} > {}".format(self.primeraderivadanumerica(xsig,f),e))
            xderiv=self.primeraderivadanumerica(x_actual,f)
            xderiv2=self.segundaderivadanumerica(x_actual,f)
            #print(xsig)
            xsig= x_actual  - (xderiv/xderiv2)
            #print("{} - {} / {}".format(x_actual,xderiv,xderiv2))
            x_actual=xsig
            #print("{} - {}".format(x_actual,(xderiv/xderiv2) ))
            cont+=1
            
        return xsig
    
    def evaluar_alpha(self,alpha):
        x_alpha=(self.x_t) + (alpha * self.x_s)
        x_nuevo=self.funcion(x_alpha[0],x_alpha[1])#Se evalua el punto
        return x_nuevo
    
    def busqueda_unidireccional(self,epsilon):
        alpha_nuevo=self.newton_raphson(self.alfa,epsilon,self.evaluar_alpha)
        x_alpha_new=(self.x_t) + (alpha_nuevo * self.x_s)
        return (x_alpha_new[0],x_alpha_new[1])



x_t=np.array([1,1])
x_s=np.array([1,0.5])
alpha=0.2

epsilon=0.1
optimi=optimizador(x_t,x_s,himmelblau,alpha)
print(optimi.busqueda_unidireccional(epsilon))



    









