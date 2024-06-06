import numpy as np 

class cauchy_6_junio:#Esta es la clase nada mas para entregar la tarea a tiempo
    def __init__(self,a,b,epsilon,f,iter=100):
        self.a=a
        self.b=b
        self.epsilon=epsilon
        self.funcion=f
        self.puntoinicial=a
        self.iteracion=iter
    
    def delta(a,b,n):
     return (b-a)/n
    def x1(ai,delta):
        return ai*delta

    def x2(x1,delta):
        return x1+delta

    def x3(x2,delta):
        return x2+delta
    def despejen(a,b,n):
        return int((2*(b-a))/n)
    def Exhaustivesearchmethod(self,f,e:float,a:float=None, b:float=None):
        n=self.despejen(a,b,e)
        d=self.delta(a,b,n)
        X1=a
        X2=self.x2(X1,d)
        X3=self.x3(X2,d)
        FX1=f(X1)
        FX2=f(X2)
        FX3=f(X3)
        while (b>=X3):
            if FX1>=FX2 and FX2<=FX3:
                return X1,X3
            else:
                #print(" {} > {} < {}".format(FX1,FX2,FX3))
                X1=X2
                X2=X3
                X3=self.x3(X2,d)
                FX1=f(X1)
                FX2=f(X2)
                FX3=f(X3)
        return X1,X3
    
    def primeraderivadanumerica(x_actual,f):
        delta=0.0001
        numerador= f(x_actual + delta) - f (x_actual - delta) 
        return (numerador / (2 * delta))*-1

    def segundaderivadanumerica(x_actual,f):
        delta=0.0001
        numerado= f(x_actual + delta) - (2 * f (x_actual + f(x_actual- delta))) 
        return numerado / (delta**2)

    def newton_raphson(self,x,e,f):
        k=1
        x_actual=x[k]
        xderiv=self.primeraderivadanumerica(x_actual,f)
        xderiv2=self.segundaderivadanumerica(x_actual,f)
        xsig= x_actual  - (xderiv/xderiv2)
        while (self.primeraderivadanumerica(xsig,f) > e):
            k+=1
            x_actual=x[k]
            xderiv=self.primeraderivadanumerica(x_actual,f)
            xderiv2=self.segundaderivadanumerica(x_actual,f)
            xsig= x_actual  - (xderiv/xderiv2)
        return xsig

    def biseccionmethod(self,f,e:float,a_orginal:float=None, b_original:float=None):
        a = np.random.uniform(a_orginal, b_original)
        b = np.random.uniform(a_orginal, b_original)
        while(self.primeraderivadanumerica(a,f) > 0):
            a = np.random.uniform(a_orginal, b_original)
            print(self.primeraderivadanumerica(a,f))
        
        while (self.primeraderivadanumerica(b,f) < 0): 
            b = np.random.uniform(a_orginal, b_original)
        x1=a
        x2=b
        z = ((x2+x1)/2)
        #print(primeraderivadanumerica(x1,f))
        while(self.primeraderivadanumerica(z,f) > e):
            #print(z)
            if self.primeraderivadanumerica(z,f) < 0: 
                x1=z
                z=0
                z = int((x2+x1)/2)
            elif self.primeraderivadanumerica(z,f) > 0: 
                x2=z
                z=0
                z = ((x2+x1)/2)
        
        print("Listo!")
        return x1 , x2

    def calculozensecante(self,x2,x1,f):
        numerador=self.primeraderivadanumerica(x2,f)
        denominador=((self.primeraderivadanumerica(x2,f) - self.primeraderivadanumerica (x1,f)))/(x2-x1)
        op=numerador/denominador
        return x2 - op

    def metodosecante(self,f,e:float,a_orginal:float=None, b_original:float=None):
        a = np.random.randint(a_orginal, b_original)
        b = np.random.randint(a_orginal, b_original)
        while(self.primeraderivadanumerica(a,f) > 0):
            a = np.random.randint(a_orginal, b_original)
        
        while (self.primeraderivadanumerica(b,f) < 0): 
            b = np.random.randint(a_orginal, b_original)
        x1=a
        x2=b
        z = self.calculozensecante(x2,x1,f)
        while(self.primeraderivadanumerica(z,f) > e): 
            if self.primeraderivadanumerica(z,f) < 0: 
                x1=z
                z=0
                z = self.calculozensecante(x2,x1,f)
            if self.primeraderivadanumerica(z,f) > 0: 
                x2=z
                z=0
                z = self.calculozensecante(x2,x1,f)
        return x1 , x2
    
    def findregions(rangomin,rangomax,x1,x2,f):
        if f(x1)> f(x2):
            rangomin=rangomin
            rangomax=x2
        elif f(x1)< f(x2):
            rangomin=x1
            rangomax=rangomax
        elif f(x1)== f(x2):
            rangomin=x1
            rangomax=x2
        return rangomin,rangomax

    def intervalstep3(b,x1,xm,f):
        if f(x1)< f(xm):
            b=xm
            xm=x1
            return b,xm,True
        else:
            return b,xm,False

    def intervalstep4(a,x2,xm,f):
        if  f(x2)<f (xm):
            a=xm
            xm=x2
            return a,xm,True
        else:
            return a,xm,False

    def intervalstep5(b,a,e):
        l=b-a
        #print(" Valor actual de a y b = {} , {}".format(a,b))
        if abs(l) < e : 
            return False
        else:
            return True

    def intervalhalvingmethod(self,f,e:float,a:float=None, b:float=None):
        xm=(a+b)/2
        l=b-a
        x1=a + (l/4)
        x2=b - (l/4)
        a,b=self.findregions(a,b,x1,x2,f)
        #Validaciones
        endflag=self.intervalstep5(a,b,e)
        l=b-a
        while endflag:
            x1=a + (l/4)
            x2=b - l/4
            #Se obtiene las f(x) de x1 y x2 
            b,xm,flag3=self.intervalstep3(b,x1,xm,f)
            a,xm,flag4=self.intervalstep4(a,x2,xm,f)
            if flag3== True:
                endflag=self.intervalstep5(a,b,e)
            elif flag3==False:
                a,xm,flag4=self.intervalstep4(a,x2,xm,f)
            
            if flag4==True:
                endflag=self.intervalstep5(a,b,e)
            elif flag4==False: 
                a=x1
                b=x2
                endflag=self.intervalstep5(a,b,e)
        return xm
    def fibonacci_iterativo(n):
        fibonaccie = [0, 1]
        for i in range(2, n):
            fibonaccie.append(fibonaccie[i-1] + fibonaccie[i-2])
        return fibonaccie

    def calculo_lk(fibonacci,n,k):
        indice1=n - (k + 1)
        indice2= n + 1
        return fibonacci[indice1]/ fibonacci[indice2]

    def fibonaccisearch(self,a,b,n,f):
        l=b-a
        seriefibonacci=self.fibonacci_iterativo(n*10)
        #calculo de lk
        k=2
        lk=self.calculo_lk(seriefibonacci,n,k)
        x1=a+lk
        x2=b-lk
        while k != n:
            if k % 2 == 0:
                evalx1=f(x1)
                a,b=self.findregions(a,b,evalx1,x2,f)
                #print(" Valor actual de a y b = {} , {}".format(a,b))
            else:
                evalx2=f(x2)
                a,b=self.findregions(a,b,x1,evalx2,f)
                #print(" Valor actual de a y b = {} , {}".format(a,b))
            k+=1
        
        return a , b
    def boundingphase(x,delta,n,f):
        k=0
        if x - abs(delta)>= f(x) and f(x) >= f(x + abs(delta)):
            delta=abs(delta)
        elif x - abs(delta)<= f(x) and f(x) <= f(x + abs(delta)):
            delta=delta
        
        x_nuevo=x + ((2**k)* delta)
        x_anterior=x
        x_ant=x_anterior
        while f(x_nuevo) <= f(x_anterior):
            k+=1
            if k >= n:
                return x_ant,x_nuevo
            x_ant=x_anterior
            x_anterior=x_nuevo
            x_nuevo=x_anterior + ((2**k)* delta)
        
        print("Los puntos actuales son {} y {} ".format(x_anterior,x_nuevo))
        return x_ant,x_nuevo
        
    def w_to_x(w:float, a,b )-> float:
            return w * (b-a) + a

    def busquedadorada(self,f,e:float,a:float=None, b:float=None)->float:
        PHI=(1 + np.sqrt(5))/ 2-1
        aw,bw=0,1 
        Lw=1
        k=1# index del actual 
        while Lw> e: 
            w2= aw + PHI* Lw
            w1=bw - PHI * Lw
            aw,bw=self.findregions(w1,w2,aw,bw,f)
            k+=1 
            Lw=bw-aw
        
        return (f(aw,a,b) + f(bw,a,b))/2 
