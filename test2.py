import numpy as np
import math

class Optimizador:
    def __init__(self, x, epsilon1: float, epsilon2: float, f, max_iter=100):
        self.variables = np.array(x, dtype=float)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.funcion = f
        self.max_iter = max_iter

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

    def busquedaDorada(self, funcion, epsilon, a=None, b=None):
        phi = (1 + math.sqrt(5)) / 2 - 1
        aw, bw = 0, 1
        Lw = 1
        k = 1
        while Lw > epsilon:
            w2 = aw + phi * Lw
            w1 = bw - phi * Lw
            aw, bw = self.regla_eliminacion(w1, w2, funcion(self.w_to_x(w1, a, b)), funcion(self.w_to_x(w2, a, b)), aw, bw)
            k += 1
            Lw = bw - aw
        return (self.w_to_x(aw, a, b) + self.w_to_x(bw, a, b)) / 2

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

        while not terminar:
            grad = np.array(self.gradiente(f, xk))
            hessian = self.hessian_matrix(f, xk, deltaX=0.001)
            hessian_inv = np.linalg.inv(hessian)

            if np.linalg.norm(grad) < epsilon1 or k >= M:
                terminar = True
            else:
                def alpha_funcion(alpha):
                    return f(xk - alpha * np.dot(hessian_inv, grad))

                alpha = self.busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0.0, b=1.0)
                print(alpha)
                x_k1 = xk - alpha * np.dot(hessian_inv, grad)

                if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 0.00001) <= epsilon2:
                    terminar = True
                else:
                    k += 1
                    xk = x_k1

        return xk

# Example usage:
def himmelblau(p):
    return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2

x0 = np.array([2.0, 3.0])
epsilon1 = 0.001
epsilon2 = 0.001
max_iter = 100

optimizer = Optimizador(x0, epsilon1, epsilon2, himmelblau, max_iter)
result = optimizer.newton(himmelblau, x0, epsilon1, epsilon2, max_iter)
print("Optimized variables:", result)
