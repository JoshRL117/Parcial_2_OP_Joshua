import numpy as np
def funcionprueba(x1,x2):
    return x1 ** 2 + x2 **2

def funcion1(x1,x2):
    return ((x1- 10)**2 ) + ((x2 - 10)**2)
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
        return numerador / (2 * delta)

    def segundaderivadanumerica(self, x_actual,f):
        delta=0.0001
        numerado= f(x_actual + delta) - (2 * f (x_actual + f(x_actual- delta))) 
        return numerado / (delta**2)

    def newton_raphson(self,x,e,f):
        x_actual=x
        xderiv=self.primeraderivadanumerica(x_actual,f)
        xderiv2=self.segundaderivadanumerica(x_actual,f)
        xsig= x_actual  - (xderiv/xderiv2)
        x_actual=xsig
        while (self.primeraderivadanumerica(xsig,f) > e):
            print(" {} > {}".format(self.primeraderivadanumerica(xsig,f),e))
            xderiv=self.primeraderivadanumerica(x_actual,f)
            xderiv2=self.segundaderivadanumerica(x_actual,f)
            xsig= x_actual  + (xderiv/xderiv2)
            x_actual=xsig
            #print("{} - {}".format(x_actual,(xderiv/xderiv2) ))
            
        return xsig
    
    def evaluar_alpha(self,alpha):
        x_alpha=(self.x_t) + (alpha * self.x_s)
        x_nuevo=self.funcion(x_alpha[0],x_alpha[1])
        return x_nuevo
    
    def busqueda_unidireccional(self,epsilon):
        alpha_nuevo=self.newton_raphson(self.alfa,epsilon,self.evaluar_alpha)
        x_alpha_new=(self.x_t) + (alpha_nuevo * self.x_s)
        return self.funcion(x_alpha_new[0],x_alpha_new[1])



x_t=np.array([2,1])
x_s=np.array([2,5])
alpha=0.5
epsilon=0.1
optimi=optimizador(x_t,x_s,funcion1,alpha)
print(optimi.busqueda_unidireccional(epsilon))



    









