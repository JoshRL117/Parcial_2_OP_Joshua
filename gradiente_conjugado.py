import numpy as np
import math
#Estos los utilice para el reporte
def gradiente(f, x: np.ndarray, deltaX: float = 0.001):
    grad = []
    for i in range(len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i] + deltaX
        xn[i] = xn[i] - deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return np.array(grad)

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

def cauchy(funcion, x0: np.ndarray, epsilon1: float, epsilon2: float, M: int):
    terminar = False
    xk = x0
    k = 0
    
    while not terminar:
        grad = gradiente(funcion, xk)
        
        if np.linalg.norm(grad) < epsilon1 or k >= M:
            terminar = True
        else:
            def alpha_funcion(alpha):
                return funcion(xk - alpha * grad)
            
            alpha = busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0, b=1.0)
            xk_1 = xk - alpha * grad
            if np.linalg.norm(xk_1 - xk) / (np.linalg.norm(xk) + 0.0001) < epsilon2:
                terminar = True
            xk = xk_1
            k += 1
    print(k)
    return xk

def s_sig_gradcon(gradiente_ac, gradiente_ant, s):
    beta = np.dot(gradiente_ac, gradiente_ac) / np.dot(gradiente_ant, gradiente_ant)
    return -gradiente_ac + beta * s

def gradiente_conjugado(funcion, x0: np.ndarray, epsilon2: float, epsilon3: float, max_iter: int,a):
    xk = x0
    grad = gradiente(funcion, xk)
    sk = -grad
    k = 1
    while (np.linalg.norm(grad) >= epsilon3) and (k <= max_iter):
        def alpha_funcion(alpha):
            return funcion(xk + alpha * sk)
        
        alpha =busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0, b=1.0)
        print(alpha)
        xk_1 = xk + alpha * sk
        
        if np.linalg.norm(xk_1 - xk) / np.linalg.norm(xk) < epsilon2:
            break
        
        grad_1 = gradiente(funcion, xk_1)
        sk = s_sig_gradcon(grad_1, grad, sk)
        
        xk = xk_1
        grad = grad_1
        k += 1
    
    print(k)
    return xk

def alpha_propuesto(funcion,alpha):
    randv=np.random.uniform(0,0.1)
    a_new=alpha+ randv
    f_act=funcion(alpha)
    f_new=funcion(a_new)
    if f_new < f_act:
        return a_new
    
    return alpha
    

def himmelblau(p):
    return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2

x0 = np.array([1.0, 1.0])
opt = gradiente_conjugado(himmelblau, x0, epsilon2=0.001, epsilon3=0.001, max_iter=100,a=0.001)
print(opt)
