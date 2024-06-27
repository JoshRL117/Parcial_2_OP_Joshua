import numpy as np
import math

def gradiente(f, x: np.ndarray, deltaX: float = 0.001):
    grad = []
    for i in range(len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i] + deltaX
        xn[i] = xn[i] - deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return grad

def regla_eliminacion(x1: float, x2: float, fx1: float, fx2: float, a: float, b: float):
    if fx1 < fx2:
        return x1, b
    if fx1 > fx2:
        return a, x2
    return x1, x2

def w_to_x(w: float, a: float, b: float) -> float:
    return w * (b - a) + a

def busquedaDorada(funcion, epsilon: float, a: float = 0, b: float = 1) -> float:
    PHI = (1 + math.sqrt(5)) / 2 - 1
    
    aw, bw = 0, 1
    Lw = 1
    k = 1
    
    while Lw > epsilon:
        w1 = aw + PHI * Lw
        w2 = bw - PHI * Lw
        fx1 = funcion(w_to_x(w1, a, b))
        fx2 = funcion(w_to_x(w2, a, b))
        
        aw, bw = regla_eliminacion(w1, w2, fx1, fx2, aw, bw)
        
        Lw = bw - aw
        k += 1

    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

def cauchy(funcion, x0: np.ndarray, epsilon1: float, epsilon2: float, M: int,a:float):
    terminar = False
    xk = x0
    k = 0
    
    while not terminar:
        grad = np.array(gradiente(funcion, xk))
        
        if np.linalg.norm(grad) < epsilon1 or k >= M:
            terminar = True
        else:
            def alpha_funcion(alpha):
                return funcion(xk - alpha * grad)
            
            alpha = a#busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0, b=1.0)
            xk_1 = xk - alpha * grad
            if np.linalg.norm(xk_1 - xk) / (np.linalg.norm(xk) + 0.0001) < epsilon2:
                terminar = True
            xk = xk_1
            k += 1
    print(k)
    return xk

def himmelblau(p):
    return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2

x0 = np.array([1.0,1.0])
resultado = cauchy(himmelblau, x0, epsilon1=0.001, epsilon2=0.001, M=100,a=0.01)
print(f"El mínimo se encuentra en x = {resultado}")
