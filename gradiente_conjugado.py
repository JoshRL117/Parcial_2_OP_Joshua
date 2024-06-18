import numpy as np

class Optimizador:
    def __init__(self, x, epsilon: float, f, iter=100):
        self.variables =np.array(x,dtype=float)
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
        fibonaccie = [0, 1]
        for i in range(2, n):
            fibonaccie.append(fibonaccie[i - 1] + fibonaccie[i - 2])
        return fibonaccie

    def calculo_lk(self, fibonacci, n, k):
        indice1 = n - (k + 1)
        indice2 = n + 1
        return fibonacci[indice1] / fibonacci[indice2]

    def fibonaccisearch(self):
        a_inicial = 0
        b_inicial = 1
        l = b_inicial - a_inicial
        seriefibonacci = self.fibonacci_iterativo(self.iteracion * 10)
        k = 2
        lk = self.calculo_lk(seriefibonacci, self.iteracion, k)
        x1 = a_inicial + lk
        x2 = b_inicial - lk
        a = a_inicial
        b = b_inicial
        while k != self.iteracion:
            if k % 2 == 0:
                evalx1 = self.testalpha(x1)
                a, b = self.findregions(a, b, evalx1, x2)
            else:
                evalx2 = self.testalpha(x2)
                a, b = self.findregions(a, b, x1, evalx2)
            k += 1
        return a, b

    def boundingphase(self):
        k = 0
        x = self.puntoinicial
        delta = self.epsilon
        if x - abs(self.epsilon) >= self.testalpha(x) and self.testalpha(x) >= self.testalpha(x + abs(self.epsilon)):
            delta = abs(self.epsilon)
        elif x - abs(self.epsilon) <= self.testalpha(x) and self.testalpha(x) <= self.testalpha(x + abs(self.epsilon)):
            delta = delta

        x_nuevo = x + ((2**k) * delta)
        x_anterior = x
        x_ant = x_anterior
        while self.testalpha(x_nuevo) <= self.testalpha(x_anterior):
            k += 1
            if k >= self.iteracion:
                return x_ant, x_nuevo
            x_ant = x_anterior
            x_anterior = x_nuevo
            x_nuevo = x_anterior + ((2**k) * delta)
        return x_ant, x_nuevo

    def w_to_x(self, a, b, w: float) -> float:
        return w * (b - a) + a

    def busquedadorada(self) -> float:
        PHI = (1 + np.sqrt(5)) / 2 - 1
        a, b = 0,1
        aw, bw = 0, 1
        Lw = 1
        k = 1
        while Lw > self.epsilon:
            w2 = aw + PHI * Lw
            w1 = bw - PHI * Lw
            fx1 = self.w_to_x(aw, bw, w1)
            fx2 = self.w_to_x(aw, bw, w2)
            aw, bw = self.findregions_golden(fx1, fx2, w1, w2, aw, bw)
            k += 1
            Lw = bw - aw
        t=(bw * (b - a) + a)
        t2=(aw * (b - a) + a)
        print(t)
        return (t2 + t)/2

    def gradiente_calculation(self, x, delta=0.001):
        vector_f1_prim = []
        x_work = np.array(x)
        x_work_f = x_work.astype(np.float64)
        if isinstance(delta, int) or isinstance(delta, float):
            for i in range(len(x_work_f)):
                point = np.array(x_work_f, copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point, i, delta))
            return np.array(vector_f1_prim) 
        else:
            for i in range(len(x_work_f)):
                point = np.array(x_work_f, copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point, i, delta[i]))
            return np.array(vector_f1_prim) 

    def primeraderivadaop(self, x, i, delta):
        mof = x[i]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        nump1 = mof + delta
        nump2 = mof - delta
        p[i] = nump1
        p2[i] = nump2
        numerador = self.funcion(p) - self.funcion(p2)
        return numerador / (2 * delta)

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

    def cauchy(self, e1, optimizador):
        stop = False
        opt = self.optimizer(optimizador)
        xk = self.variables
        k = 0
        while not stop:
            gradiente = np.array(self.gradiente_calculation(xk))
            self.gradiente=gradiente
            if np.linalg.norm(gradiente) < e1 or k >= self.iteracion:
                stop = True
            else:
                alfa = opt()
                x_k1 = xk - (alfa * gradiente)
                if np.linalg.norm((x_k1 - xk)) / (np.linalg.norm(xk) + 0.0000001) <= self.epsilon:
                    stop = True
                else:
                    k += 1
                    xk = x_k1
        return xk
    def segundaderivadaop(self,x,i,delta):
        mof=x[i]
        p=np.array(x,copy=True)
        p2=np.array(x,copy=True)
        nump1=mof + delta
        nump2 =mof - delta
        p[i]= nump1
        p2[i]=nump2
        numerador=self.funcion(p) - (2 * self.funcion(x)) +  self.funcion(p2)
        return numerador / (delta**2) 

    def derivadadodadoop(self,x,index_principal,index_secundario,delta):
        mof=x[index_principal]
        mof2=x[index_secundario]
        p=np.array(x,copy=True)
        p2=np.array(x,copy=True)
        p3=np.array(x,copy=True)
        p4=np.array(x,copy=True)
        if isinstance(delta,int) or isinstance(delta,float):#Cuando delta es un solo valor y no un arreglo 
            mod1=mof + delta
            mod2=mof - delta
            mod3=mof2 + delta
            mod4=mof2 - delta
            p[index_principal]=mod1
            p[index_secundario]=mod3
            p2[index_principal]=mod1
            p2[index_secundario]=mod4
            p3[index_principal]=mod2
            p3[index_secundario]=mod3
            p4[index_principal]=mod2
            p4[index_secundario]=mod4
            numerador=((self.funcion(p)) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4*delta)
        else:#delta si es un arreglo 
            mod1=mof + delta[index_principal]
            mod2=mof - delta[index_principal]
            mod3=mof2 + delta[index_secundario]
            mod4=mof2 - delta[index_secundario]
            p[index_principal]=mod1
            p[index_secundario]=mod3
            p2[index_principal]=mod1
            p2[index_secundario]=mod4
            p3[index_principal]=mod2
            p3[index_secundario]=mod3
            p4[index_principal]=mod2
            p4[index_secundario]=mod4
            numerador=((self.funcion(p)) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4*delta)

        
    def hessian_matrix(self,x,delt= float(0.001)):# x es el vector de variables
        matrix_f2_prim=[([0]*len(x)) for i in range(len(x))]
        x_work=np.array(x)
        x_work_f=x_work.astype(np.float64)
        for i in range(len(x)):
            point=np.array(x_work_f,copy=True)
            for j in range(len(x)):
                if i == j:
                    matrix_f2_prim[i][j]=self.segundaderivadaop(point,i,delt)
                else:
                    matrix_f2_prim[i][j]=self.derivadadodadoop(point,i,j,delt)
        return matrix_f2_prim


    def newton_multvariable(self,e1,optimizador):#e son los epsilon y M es el numero de iteraciones 
        stop=False
        opt=self.optimizer(optimizador)
        xk=self.variables
        k=0
        while not stop: 
            gradiente=np.array(self.gradiente_calculation(xk))
            hessiana=self.hessian_matrix(xk)
            if np.linalg.norm(gradiente)< e1 or k >=self.iteracion:
                stop=True 
            else:
                #Es para que este epsilon sea el de los optimizadores  
                alfa=opt()
                x_k1= xk - (alfa * np.dot(np.linalg.inv(hessiana),gradiente))
                
                if np.linalg.norm((x_k1-xk))/ (np.linalg.norm(xk) + 0.0000001 )<= self.epsilon:
                    stop=True 
                else:
                    k+=1
                    xk=x_k1
        return xk
    
    def s_sig_gradcon(self, xac, xant, s):
        gradiente_ac = np.array(self.gradiente_calculation(xac))
        gradiente_ant = np.array(self.gradiente_calculation(xant))
        beta = np.dot(gradiente_ac, gradiente_ac) / np.dot(gradiente_ant, gradiente_ant)
        return -gradiente_ac + beta * s


    def grandiente_conjugado(self, e2, e3, optimizador):
        x_inicial = self.variables
        s_inicial = -self.gradiente_calculation(x_inicial)
        opt = self.optimizer(optimizador)
        alfa_inicial = opt()
        x_nuevo = x_inicial + alfa_inicial * s_inicial
        s_nuevo = self.s_sig_gradcon(x_nuevo, x_inicial, s_inicial)
        self.gradiente = s_nuevo 
        alfa_nuevo = opt()
        x_ant = x_nuevo
        x_nuevo = x_ant + alfa_nuevo * s_nuevo
        print(x_nuevo)
        k = 0

        while (np.linalg.norm(x_nuevo - x_ant) / np.linalg.norm(x_ant) >= e2) \
                or (np.linalg.norm(self.gradiente_calculation(x_nuevo)) >= e3) \
                or k < self.iteracion:
            s_ant = s_nuevo
            s_nuevo = self.s_sig_gradcon(x_nuevo, x_ant, s_ant)
            self.gradiente = s_nuevo
            alfa_nuevo = opt()
            x_ant = x_nuevo
            x_nuevo = x_ant + alfa_nuevo * s_nuevo
            print(x_nuevo)
            k += 1

        return x_nuevo

if __name__ == "__main__":
    def himmelblau(p):
        return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2
    
    x = [1.5,1]
    e = 0.001
    opt = Optimizador(x, e, himmelblau)
    #print(opt.cauchy(e, 'golden'))
    #print(opt.newton_multvariable(e,'golden'))
    print(opt.grandiente_conjugado(e,e,'golden'))
    #La salida debe ser de 3,2
