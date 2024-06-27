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
    return np.array(grad)

def regla_eliminacion(x1: float, x2: float, fx1: float, fx2: float, a: float, b: float):
    if fx1 < fx2:
        return x1, b
    if fx1 > fx2:
        return a, x2
    return x1, x2

def w_to_x(w: float, a: float, b: float) -> float:
    return w * (b - a) + a

def alpha_propuesto(funcion, alpha, xk, sk,t):
    randv = np.random.uniform(0, 0.1)
    a_new = alpha + randv
    f_act = funcion(xk)
    f_new = funcion(xk + a_new * sk)
    if f_new < f_act:
        return a_new
    else:
        v = (f_new - f_act) / t
        if v >= np.random.uniform(0, 0.1):
            return alpha
        return alpha

def cauchy(funcion, x0: np.ndarray, epsilon1: float, epsilon2: float, M: int, alpha: float = 0.01):
    terminar = False
    xk = x0
    k = 0
    t = 0.5

    while not terminar:
        grad = gradiente(funcion, xk)
        
        if np.linalg.norm(grad) < epsilon1 or k >= M or t < epsilon2:
            terminar = True
        else:
            alpha = alpha_propuesto(funcion, alpha, xk, -grad, t)
            xk_1 = xk - alpha * grad
            if np.linalg.norm(xk_1 - xk) / (np.linalg.norm(xk) + 0.0001) < epsilon2:
                terminar = True
            xk = xk_1
            k += 1
            t *= alpha
    
    print(f"Iteraciones (Cauchy): {k}")
    return xk

def s_sig_gradcon(gradiente_ac, gradiente_ant, s):
    beta = np.dot(gradiente_ac, gradiente_ac) / np.dot(gradiente_ant, gradiente_ant)
    return -gradiente_ac + beta * s

def gradiente_conjugado(funcion, x0: np.ndarray, epsilon2: float, epsilon3: float, max_iter: int, alpha: float = 0.01):
    xk = x0
    grad = gradiente(funcion, xk)
    sk = -grad
    k = 1
    t = 0.5

    while (np.linalg.norm(grad) >= epsilon3) and (k <= max_iter) and (t >= epsilon2):
        alpha= alpha_propuesto(funcion, alpha, xk, sk,t)
        xk_1 = xk + alpha * sk
        
        if np.linalg.norm(xk_1 - xk) / np.linalg.norm(xk) < epsilon2:
            break
        
        grad_1 = gradiente(funcion, xk_1)
        sk = s_sig_gradcon(grad_1, grad, sk)
        
        xk = xk_1
        grad = grad_1
        k += 1
        t=t*alpha
    
    print(f"Iteraciones (Gradiente Conjugado): {k}")
    return xk

def himmelblau(p):
    return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2

x0 = np.array([1.0, 1.0])

# Ejemplo de uso de Cauchy con alpha propuesto
resultado_cauchy = cauchy(himmelblau, x0, epsilon1=0.001, epsilon2=0.001, M=100, alpha=0.5)
print(f"Mínimo con método de Cauchy se encuentra en x = {resultado_cauchy}")

# Ejemplo de uso de Gradiente Conjugado con alpha propuesto
resultado_gradiente_conjugado = gradiente_conjugado(himmelblau, x0, epsilon2=0.001, epsilon3=0.001, max_iter=100, alpha=0.01)
print(f"Mínimo con método de Gradiente Conjugado se encuentra en x = {resultado_gradiente_conjugado}")
