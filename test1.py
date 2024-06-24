import numpy as np

class Optimizador:
    def __init__(self, x, epsilon: float, f, iter=100):
        self.variables = np.array(x, dtype=float)
        self.epsilon = epsilon
        self.funcion = f
        self.puntoinicial = float(x[0])
        self.iteracion = iter
        self.gradiente = []

    def testalpha(self, alfa):
        return self.funcion(self.variables - (alfa * self.gradiente))

    def delta(self, a, b, n):
        return (b - a) / n

    def x2(self, x1, delta):
        return x1 + delta

    def x3(self, x2, delta):
        return x2 + delta

    def despejen(self, a, b, n):
        return int((2 * (b - a)) / n)

    def Exhaustivesearchmethod(self):
        a = self.puntoinicial
        b = max(self.variables)
        n = self.despejen(a, b, self.epsilon)
        d = self.delta(a, b, n)
        X1 = a
        X2 = self.x2(X1, d)
        X3 = self.x3(X2, d)
        FX1 = self.testalpha(X1)
        FX2 = self.testalpha(X2)
        FX3 = self.testalpha(X3)
        while (b >= X3):
            if FX1 >= FX2 and FX2 <= FX3:
                return X1, X3
            else:
                X1 = X2
                X2 = X3
                X3 = self.x3(X2, d)
                FX1 = self.testalpha(X1)
                FX2 = self.testalpha(X2)
                FX3 = self.testalpha(X3)
        return (X1 + X3) / 2

    def primeraderivadanumerica(self, x_actual, f):
        delta = 0.0001
        numerador = f([x_actual + delta]) - f([x_actual - delta])
        return numerador / (2 * delta)

    def segundaderivadanumerica(self, x_actual, f):
        delta = 0.0001
        numerador = f([x_actual + delta]) - (2 * f([x_actual])) + f([x_actual - delta])
        return numerador / (delta**2)

    def newton_raphson(self):
        cont = 0
        x_actual = self.puntoinicial
        xderiv = self.primeraderivadanumerica(x_actual, self.testalpha)
        xderiv2 = self.segundaderivadanumerica(x_actual, self.testalpha)
        xsig = x_actual - (xderiv / xderiv2)
        x_actual = xsig
        while abs(self.primeraderivadanumerica(xsig, self.testalpha)) > self.epsilon:
            xderiv = self.primeraderivadanumerica(x_actual, self.testalpha)
            xderiv2 = self.segundaderivadanumerica(x_actual, self.testalpha)
            xsig = x_actual - (xderiv / xderiv2)
            x_actual = xsig
            cont += 1
        return xsig

    def biseccionmethod(self):
        a = np.random.uniform(self.puntoinicial, max(self.variables))
        b = np.random.uniform(self.puntoinicial, max(self.variables))
        while self.primeraderivadanumerica(a, self.testalpha) > 0:
            a = np.random.uniform(self.puntoinicial, max(self.variables))

        while self.primeraderivadanumerica(b, self.testalpha) < 0:
            b = np.random.uniform(self.puntoinicial, max(self.variables))
        x1 = a
        x2 = b
        z = (x2 + x1) / 2
        while self.primeraderivadanumerica(z, self.testalpha) > self.epsilon:
            if self.primeraderivadanumerica(z, self.testalpha) < 0:
                x1 = z
                z = (x2 + x1) / 2
            elif self.primeraderivadanumerica(z, self.testalpha) > 0:
                x2 = z
                z = (x2 + x1) / 2
        return x1, x2

    def calculozensecante(self, x2, x1, f):
        numerador = self.primeraderivadanumerica(x2, f)
        denominador = (self.primeraderivadanumerica(x2, f) - self.primeraderivadanumerica(x1, f)) / (x2 - x1)
        op = numerador / denominador
        return x2 - op

    def metodosecante(self):
        a = np.random.uniform(self.puntoinicial, max(self.variables))
        b = np.random.uniform(self.puntoinicial, max(self.variables))
        while self.primeraderivadanumerica(a, self.testalpha) > 0:
            a = np.random.uniform(self.puntoinicial, max(self.variables))

        while self.primeraderivadanumerica(b, self.testalpha) < 0:
            b = np.random.uniform(self.puntoinicial, max(self.variables))
        x1 = a
        x2 = b
        z = self.calculozensecante(x2, x1, self.testalpha)
        while self.primeraderivadanumerica(z, self.testalpha) > self.epsilon:
            if self.primeraderivadanumerica(z, self.testalpha) < 0:
                x1 = z
                z = self.calculozensecante(x2, x1, self.testalpha)
            if self.primeraderivadanumerica(z, self.testalpha) > 0:
                x2 = z
                z = self.calculozensecante(x2, x1, self.testalpha)
        return x1, x2

    def findregions(self, rangomin, rangomax, x1, x2):
        if self.testalpha(x1) > self.testalpha(x2):
            rangomin = rangomin
            rangomax = x2
        elif self.testalpha(x1) < self.testalpha(x2):
            rangomin = x1
            rangomax = rangomax
        elif self.testalpha(x1) == self.testalpha(x2):
            rangomin = x1
            rangomax = x2
        return rangomin, rangomax

    def findregions_golden(self, fx1, fx2, rangomin, rangomax, x1, x2):
        if fx1 > fx2:
            rangomin = rangomin
            rangomax = x2
        elif fx1 < fx2:
            rangomin = x1
            rangomax = rangomax
        elif fx1 == fx2:
            rangomin = x1
            rangomax = x2
        return rangomin, rangomax

    def intervalstep3(self, b, x1, xm):
        if self.testalpha(x1) < self.testalpha(xm):
            b = xm
            xm = x1
            return b, xm, True
        else:
            return b, xm, False

    def intervalstep4(self, a, x2, xm):
        if self.testalpha(x2) < self.testalpha(xm):
            a = xm
            xm = x2
            return a, xm, True
        else:
            return a, xm, False

    def intervalstep5(self, b, a):
        l = b - a
        if abs(l) < self.epsilon:
            return False
        else:
            return True

    def intervalhalvingmethod(self):
        a = 0
        b = 1
        xm = (a + b) / 2
        l = b - a
        x1 = a + (l / 4)
        x2 = b - (l / 4)
        a, b = self.findregions(a, b, x1, x2)
        endflag = self.intervalstep5(a, b)
        l = b - a
        while endflag:
            x1 = a + (l / 4)
            x2 = b - l / 4
            b, xm, flag3 = self.intervalstep3(b, x1, xm)
            a, xm, flag4 = self.intervalstep4(a, x2, xm)
            if flag3 == True:
                endflag = self.intervalstep5(a, b)
            elif flag3 == False:
                a, xm, flag4 = self.intervalstep4(a, x2, xm)
            if flag4 == True:
                endflag = self.intervalstep5(a, b)
            elif flag4 == False:
                a = x1
                b = x2
                endflag = self.intervalstep5(a, b)
        return xm

    def fibonacci_iterativo(self, n):
        fibonacci = [0, 1]
        for i in range(2, n):
            fibonacci.append(fibonacci[i - 1] + fibonacci[i - 2])
        return fibonacci

    def fibonaccimethod(self):
        a = np.random.uniform(self.puntoinicial, max(self.variables))
        b = np.random.uniform(self.puntoinicial, max(self.variables))
        L = b - a
        n = (b - a) / self.epsilon
        n = int(n)
        fibonacci = self.fibonacci_iterativo(n)
        k = 0
        while k <= (n - 3):
            L = b - a
            x1 = a + (fibonacci[n - k - 2] / fibonacci[n - k]) * L
            x2 = a + (fibonacci[n - k - 1] / fibonacci[n - k]) * L
            if self.testalpha(x1) < self.testalpha(x2):
                b = x2
            else:
                a = x1
            k = k + 1
        x2 = x1 + self.epsilon
        if self.testalpha(x1) < self.testalpha(x2):
            return (a + x2) / 2
        else:
            return (x1 + b) / 2

    def findx1(self, rango1, rango2):
        return rango1 + (0.382 * (rango2 - rango1))

    def findx2(self, rango1, rango2):
        return rango2 - (0.382 * (rango2 - rango1))

    def busquedadorada(self):
        rango1 = np.random.uniform(self.puntoinicial, max(self.variables))
        rango2 = np.random.uniform(self.puntoinicial, max(self.variables))
        x1 = self.findx1(rango1, rango2)
        x2 = self.findx2(rango1, rango2)
        fx1 = self.testalpha(x1)
        fx2 = self.testalpha(x2)
        rango1, rango2 = self.findregions_golden(fx1, fx2, rango1, rango2, x1, x2)
        while abs(rango2 - rango1) > self.epsilon:
            x1 = self.findx1(rango1, rango2)
            x2 = self.findx2(rango1, rango2)
            fx1 = self.testalpha(x1)
            fx2 = self.testalpha(x2)
            rango1, rango2 = self.findregions_golden(fx1, fx2, rango1, rango2, x1, x2)
        return (rango2 + rango1) / 2

    def boundingphase(self):
        x0 = self.puntoinicial
        delta = 0.0001
        k = 0
        if self.testalpha(x0 + delta) >= self.testalpha(x0) and self.testalpha(x0 - delta) >= self.testalpha(x0):
            return (x0 - delta, x0 + delta)
        elif self.testalpha(x0 + delta) <= self.testalpha(x0) and self.testalpha(x0 - delta) <= self.testalpha(x0):
            return (x0 - delta, x0 + delta)
        if self.testalpha(x0 + delta) <= self.testalpha(x0) and self.testalpha(x0 - delta) >= self.testalpha(x0):
            x1 = x0 + delta
            x2 = x0 + (2 * delta)
            while self.testalpha(x1) > self.testalpha(x2):
                k = k + 1
                x0 = x1
                x1 = x2
                x2 = x2 + (2**k * delta)
            return (x0, x2)
        elif self.testalpha(x0 + delta) >= self.testalpha(x0) and self.testalpha(x0 - delta) <= self.testalpha(x0):
            x1 = x0 - delta
            x2 = x0 - (2 * delta)
            while self.testalpha(x1) > self.testalpha(x2):
                k = k + 1
                x0 = x1
                x1 = x2
                x2 = x2 - (2**k * delta)
            return (x0, x2)

    def optimizer(self, name):
        name = name.lower()
        if name == 'exhaustive':
            return self.Exhaustivesearchmethod
        elif name == 'bounding':
            return self.boundingphase
        elif name == 'interval':
            return self.intervalhalvingmethod
        elif name == 'golden':
            return self.busquedadorada
        elif name == 'newton':
            return self.newton_raphson
        elif name == 'secante':
            return self.metodosecante
        elif name == 'biseccion':
            return self.biseccionmethod
        else:
            raise ValueError("Optimizador no reconocido")

    def primeraderivadanumerica(self, x, f, i):
        delta = 0.0001
        x1 = np.array(x, dtype=float)
        x2 = np.array(x, dtype=float)
        x1[i] += delta
        x2[i] -= delta
        return (f(x1) - f(x2)) / (2 * delta)

    def segundaderivadanumerica(self, x, f, i):
        delta = 0.0001
        x1 = np.array(x, dtype=float)
        x2 = np.array(x, dtype=float)
        x1[i] += delta
        x2[i] -= delta
        return (f(x1) - 2 * f(x) + f(x2)) / (delta**2)

    def gradiente_calculation(self, x, delta=0.001):
        gradiente = []
        for i in range(len(x)):
            gradiente.append(self.primeraderivadanumerica(x, self.funcion, i))
        return np.array(gradiente)

    def hessian_matrix(self, x, delta=0.001):
        n = len(x)
        hessian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    hessian[i][j] = self.segundaderivadanumerica(x, self.funcion, i)
                else:
                    x1 = np.array(x, dtype=float)
                    x2 = np.array(x, dtype=float)
                    x3 = np.array(x, dtype=float)
                    x4 = np.array(x, dtype=float)
                    x1[i] += delta
                    x1[j] += delta
                    x2[i] += delta
                    x2[j] -= delta
                    x3[i] -= delta
                    x3[j] += delta
                    x4[i] -= delta
                    x4[j] -= delta
                    hessian[i][j] = (self.funcion(x1) - self.funcion(x2) - self.funcion(x3) + self.funcion(x4)) / (4 * delta**2)
        return hessian

    def newton_multvariable(self, e1, optimizador):
        stop = False
        opt = self.optimizer(optimizador)
        xk = self.variables
        k = 0
        while not stop: 
            gradiente = np.array(self.gradiente_calculation(xk))
            self.gradiente = gradiente
            hessiana = np.array(self.hessian_matrix(xk))
            if np.linalg.norm(gradiente) < e1 or k >= self.iteracion:
                stop = True 
            else:
                alfa = opt()
                x_k1 = xk - alfa * np.dot(np.linalg.inv(hessiana), gradiente)
                if np.linalg.norm((x_k1 - xk)) / (np.linalg.norm(xk) + 1e-8) <= self.epsilon:
                    stop = True 
                else:
                    k += 1
                    xk = x_k1
        return xk

if __name__ == "__main__":
    def himmelblau(p):
        return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2

    x = [1.5, 1.0]
    e = 0.001
    opt = Optimizador(x, e, himmelblau)
    print(opt.newton_multvariable(e, 'golden'))
