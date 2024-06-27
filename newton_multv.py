import numpy as np
import math

class Optimizador:
    def __init__(self, x, epsilon1: float, epsilon2: float, f, max_iter=100):
        self.variables = np.array(x, dtype=float)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.funcion = f
        self.max_iter = max_iter
    
    def alpha_propuesto(self, funcion, alpha, xk, sk, t):
        randv = np.random.uniform(0, 0.1)
        a_new = alpha + randv
        f_act = funcion(xk)
        f_new = funcion(xk + a_new * sk)
        if f_new < f_act:
            return a_new
        else:
            v = (f_new - f_act) / t
            if v >= np.random.uniform(0, 0.1):
                return a_new
        return alpha

    def hessian_matrix(self, f, x, deltaX):
        fx = f(x)
        N = len(x)
        H = []
        for i in range(N):
            hi = []
            for j in range(N):
                if i == j:
                    xp = x.copy()
                    xn = x.copy()
                    xp[i] = xp[i] + deltaX
                    xn[i] = xn[i] - deltaX
                    hi.append((f(xp) - 2 * fx + f(xn)) / (deltaX ** 2))
                else:
                    xpp = x.copy()
                    xpn = x.copy()
                    xnp = x.copy()
                    xnn = x.copy()
                    xpp[i] = xpp[i] + deltaX
                    xpp[j] = xpp[j] + deltaX
                    xpn[i] = xpn[i] + deltaX
                    xpn[j] = xpn[j] - deltaX
                    xnp[i] = xnp[i] - deltaX
                    xnp[j] = xnp[j] + deltaX
                    xnn[i] = xnn[i] - deltaX
                    xnn[j] = xnn[j] - deltaX
                    hi.append((f(xpp) - f(xpn) - f(xnp) + f(xnn)) / (4 * deltaX ** 2))
            H.append(hi)
        return np.array(H)

    def regla_eliminacion(self, x1, x2, fx1, fx2, a, b):
        if fx1 > fx2:
            return x1, b
        if fx1 < fx2:
            return a, x2
        return x1, x2 

    def w_to_x(self, w, a, b):
        return w * (b - a) + a 

    def gradiente(self, f, x, deltaX=0.001):
        grad = []
        for i in range(len(x)):
            xp = x.copy()
            xn = x.copy()
            xp[i] = xp[i] + deltaX
            xn[i] = xn[i] - deltaX
            grad.append((f(xp) - f(xn)) / (2 * deltaX))
        return np.array(grad)

    def newton(self, f, x0, epsilon1, epsilon2, M):
        terminar = False
        xk = x0
        k = 0
        t = 0.5
        alfa = 0.01
        while not terminar:
            gradiente = np.array(self.gradiente(f, xk))
            hessiana = self.hessian_matrix(f, xk, deltaX=0.001)
            hessian_inv = np.linalg.inv(hessiana)

            if np.linalg.norm(gradiente) < epsilon1 or k >= M or t < epsilon2:
                terminar = True
            else:
                def alpha_funcion(alpha):
                    return f(xk - alpha * np.dot(hessian_inv, gradiente))

                alfa = self.alpha_propuesto(alpha_funcion, alfa, xk, np.dot(hessian_inv, gradiente), t)
                x_k1 = xk - alfa * np.dot(hessian_inv, gradiente)

                if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 0.00001) <= epsilon2:
                    terminar = True
                else:
                    k += 1
                    xk = x_k1
                    t *= alfa

        print("Número de iteraciones:", k)
        return xk

def himmelblau(p):
    return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2

x0 = np.array([2.0, 3.0])
epsilon1 = 0.001
epsilon2 = 0.001
max_iter = 100

optimizer = Optimizador(x0, epsilon1, epsilon2, himmelblau, max_iter)
result = optimizer.newton(himmelblau, x0, epsilon1, epsilon2, max_iter)
print("Optimized variables:", result)
