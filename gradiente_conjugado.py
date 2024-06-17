import numpy as np 

class cauchy_6_junio:#Esta es la clase nada mas para entregar la tarea a tiempo
    def __init__(self,a,b,epsilon:float,f,iter=100):
        self.a=float(a)
        self.b=float(b)
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
    def Exhaustivesearchmethod(self):
        n=self.despejen(self.a,self.b,self.epsilon)
        d=self.delta(self.a,self.b,n)
        X1=self.a
        X2=self.x2(X1,d)
        X3=self.x3(X2,d)
        FX1=self.funcion(X1)
        FX2=self.funcion(X2)
        FX3=self.funcion(X3)
        while (self.b>=X3):
            if FX1>=FX2 and FX2<=FX3:
                return X1,X3
            else:
                #print(" {} > {} < {}".format(FX1,FX2,FX3))
                X1=X2
                X2=X3
                X3=self.x3(X2,d)
                FX1=self.funcion(X1)
                FX2=self.funcion(X2)
                FX3=self.funcion(X3)
        return X1,X3
    
    def primeraderivadanumerica(self, x_actual,f):
        delta=0.0001
        numerador= f(x_actual + delta) - f (x_actual - delta) 
        return (numerador / (2 * delta))

    def segundaderivadanumerica(self, x_actual,f):
        delta=0.0001
        numerado= f(x_actual + delta) - (2 * f (x_actual + f(x_actual- delta))) 
        return (numerado / (delta**2))

    def newton_raphson(self):
        cont=0
        x_actual=self.a
        xderiv=self.primeraderivadanumerica(x_actual,self.funcion)
        xderiv2=self.segundaderivadanumerica(x_actual,self.funcion)
        xsig= x_actual  - (xderiv/xderiv2)
        print("{} - {} / {}".format(x_actual,xderiv,xderiv2))
        print(xsig)
        x_actual=xsig
        while  (abs(self.primeraderivadanumerica(xsig,self.funcion)) > self.epsilon):
            #print(" {} > {}".format(self.primeraderivadanumerica(xsig,f),e))
            xderiv=self.primeraderivadanumerica(x_actual,self.funcion)
            xderiv2=self.segundaderivadanumerica(x_actual,self.funcion)
            #print(xsig)
            xsig= x_actual  - (xderiv/xderiv2)
            #print("{} - {} / {}".format(x_actual,xderiv,xderiv2))
            x_actual=xsig
            #print("{} - {}".format(x_actual,(xderiv/xderiv2) ))
            cont+=1
            
        return xsig

    def biseccionmethod(self):
        a = np.random.uniform(self.a, self.b)
        b = np.random.uniform(self.a, self.b)
        while(self.primeraderivadanumerica(a,self.funcion) > 0):
            a = np.random.uniform(self.a, self.b)
        
        while (self.primeraderivadanumerica(b,self.funcion) < 0): 
            b = np.random.uniform(self.a, self.b)
        x1=a
        x2=b
        z = ((x2+x1)/2)
        #print(primeraderivadanumerica(x1,f))
        while(self.primeraderivadanumerica(z,self.funcion) > self.epsilon):
            #print(z)
            if self.primeraderivadanumerica(z,self.funcion) < 0: 
                x1=z
                z=0
                z = int((x2+x1)/2)
            elif self.primeraderivadanumerica(z,self.funcion) > 0: 
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

    def metodosecante(self):
        a = np.random.randint(self.a, self.b)
        b =np.random.randint(self.a, self.b)
        while(self.primeraderivadanumerica(a,self.funcion) > 0):
            a = np.random.randint(self.a, self.b)
        
        while (self.primeraderivadanumerica(b,self.funcion) < 0): 
            b = np.random.randint(self.a, self.b)
        x1=a
        x2=b
        z = self.calculozensecante(x2,x1,self.funcion)
        while(self.primeraderivadanumerica(z,self.funcion) > self.epsilon): 
            if self.primeraderivadanumerica(z,self.funcion) < 0: 
                x1=z
                z=0
                z = self.calculozensecante(x2,x1,self.funcion)
            if self.primeraderivadanumerica(z,self.funcion) > 0: 
                x2=z
                z=0
                z = self.calculozensecante(x2,x1,self.funcion)
        return x1 , x2
    
    def findregions(self,rangomin,rangomax,x1,x2):
        if self.funcion(x1)> self.funcion(x2):
            rangomin=rangomin
            rangomax=x2
        elif self.funcion(x1)< self.funcion(x2):
            rangomin=x1
            rangomax=rangomax
        elif self.funcion(x1)== self.funcion(x2):
            rangomin=x1
            rangomax=x2
        return rangomin,rangomax

    def findregions_golden(self,fx1,fx2,rangomin,rangomax,x1,x2):
        if fx1 > fx2:
            rangomin=rangomin
            rangomax=x2
        elif fx1 < fx2:
            rangomin=x1
            rangomax=rangomax
        elif fx1== fx2:
            rangomin=x1
            rangomax=x2
        return rangomin,rangomax
    def intervalstep3(self,b,x1,xm):
        if self.funcion(x1)< self.funcion(xm):
            b=xm
            xm=x1
            return b,xm,True
        else:
            return b,xm,False

    def intervalstep4(self,a,x2,xm):
        if  self.funcion(x2)<self.funcion (xm):
            a=xm
            xm=x2
            return a,xm,True
        else:
            return a,xm,False

    def intervalstep5(self,b,a):
        l=b-a
        #print(" Valor actual de a y b = {} , {}".format(a,b))
        if abs(l) < self.epsilon : 
            return False
        else:
            return True

    def intervalhalvingmethod(self):
        xm=(self.a+self.b)/2
        l=self.b-self.a
        x1=self.a + (l/4)
        x2=self.b - (l/4)
        a,b=self.findregions(a,b,x1,x2)
        #Validaciones
        endflag=self.intervalstep5(a,b)
        l=b-a
        while endflag:
            x1=a + (l/4)
            x2=b - l/4
            #Se obtiene las f(x) de x1 y x2 
            b,xm,flag3=self.intervalstep3(b,x1,xm)
            a,xm,flag4=self.intervalstep4(a,x2,xm)
            if flag3== True:
                endflag=self.intervalstep5(a,b)
            elif flag3==False:
                a,xm,flag4=self.intervalstep4(a,x2,xm)
            
            if flag4==True:
                endflag=self.intervalstep5(a,b)
            elif flag4==False: 
                a=x1
                b=x2
                endflag=self.intervalstep5(a,b)
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

    def fibonaccisearch(self):
        l=self.b-self.a
        seriefibonacci=self.fibonacci_iterativo(self.iteracion*10)
        #calculo de lk
        k=2
        lk=self.calculo_lk(seriefibonacci,self.iteracion,k)
        x1=self.a+lk
        x2=self.b-lk
        a=self.a
        b=self.b
        while k != self.iteracion:
            if k % 2 == 0:
                evalx1=self.funcion(x1)
                a,b=self.findregions(a,b,evalx1,x2,self.funcion)
                #print(" Valor actual de a y b = {} , {}".format(a,b))
            else:
                evalx2=self.funcion(x2)
                a,b=self.findregions(a,b,x1,evalx2,self.funcion)
                #print(" Valor actual de a y b = {} , {}".format(a,b))
            k+=1
        
        return (a + b)/2
    def boundingphase(self):
        k=0
        x=self.a
        if x - abs(self.epsilon)>= self.funcion(x) and self.funcion(x) >= self.funcion(x + abs(self.epsilon)):
            delta=abs(self.epsilon)
        elif x - abs(self.epsilon)<= self.funcion(x) and self.funcion(x) <= self.funcion(x + abs(self.epsilon)):
            delta=delta
        
        x_nuevo=x + ((2**k)* delta)
        x_anterior=x
        x_ant=x_anterior
        while self.funcion(x_nuevo) <= self.funcion(x_anterior):
            k+=1
            if k >= self.iteracion:
                return x_ant,x_nuevo
            x_ant=x_anterior
            x_anterior=x_nuevo
            x_nuevo=x_anterior + ((2**k)* delta)
        
        print("Los puntos actuales son {} y {} ".format(x_anterior,x_nuevo))
        return (x_ant +x_nuevo)/2
        
    def w_to_x(self, w:float )-> float:
            return w * (self.b-self.a) + self.a

    def busquedadorada(self)->float:
        PHI=(1 + np.sqrt(5))/ 2-1
        aw,bw=0,1 
        Lw=1
        k=1# index del actual 
        while Lw> self.epsilon: 
            w2= aw + PHI* Lw
            w1=bw - PHI * Lw
            fx1=self.w_to_x(w1)
            fx2=self.w_to_x(w2)
            aw,bw=self.findregions_golden(fx1,fx2,w1,w2,aw,bw)
            k+=1 
            Lw=bw-aw
        
        return (self.w_to_x(aw) + self.w_to_x(bw))/2 
    def gradiente_calculation(self,x,delta=float(0.001)):
        vector_f1_prim=[]
        x_work=np.array(x)
        x_work_f=x_work.astype(np.float64)
        if isinstance(delta,int) or isinstance(delta,float):
            #print("X_no es arreglo")
            for i in range(len(x_work_f)):
                point=np.array(x_work_f,copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point,i,delta))
            return vector_f1_prim
        else:
            for i in range(len(x_work_f)):
                point=np.array(x_work_f,copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point,i,delta[i]))
            return vector_f1_prim

    def primeraderivadaop(self,x,i,delta):
        mof=x[i]
        p=np.array(x,copy=True)
        p2=np.array(x,copy=True)
        nump1=mof + delta
        nump2 =mof - delta
        p[i]= nump1
        p2[i]=nump2
        numerador=self.funcion(p) - self.funcion(p2)
        return numerador / (2 * delta) 
    
    def optimizer(self,name):
        name=name.lower()
        if name == 'exhaustive':#1
            return self.Exhaustivesearchmethod
        elif name== 'bounding':#2
            return self.boundingphase
        elif name == 'interval':#3 
            return self.intervalhalvingmethod
        elif name == 'golden':#4
            return self.busquedadorada
        elif name == 'newton':#5
            return self.newton_raphson
        elif name== 'secante':#6
            return self.metodosecante
        elif name == 'biseccion':#7
            return self.biseccionmethod 


    def cauchy(self,e1,optimizador):#e son los epsilon y M es el numero de iteraciones 
        stop=False
        opt=self.optimizer(optimizador)
        xk=np.array([self.a, self.b])
        k=0
        while not stop: 
            gradiente=np.array(self.gradiente_calculation(xk))
            if np.linalg.norm(gradiente)< e1 or k >=self.iteracion:
                stop=True 
            else:
                #Es para que este epsilon sea el de los optimizadores  
                alfa=opt()
                x_k1= xk - alfa * gradiente#Punto siguiente a xk
                if np.linalg.norm((x_k1-xk))/ (np.linalg.norm(xk) + 0.0000001 )<= self.epsilon:
                    stop=True 
                else:
                    k+=1
                    xk=x_k1
        return xk
    
    def alfa_dadopunto(self,initial_point,optimizador ='golden'):#Este retorna el alfa donde el punto sea un minimo para gradiente conjugado
        alfa=optimizador()
        return alfa
    def minormax(self,hessian):#Aqui encuentre que es el punto 
        det=np.linalg.det(hessian)
        f_xx=hessian[0][0]
        if det > 0 and f_xx > 0:
            return "Mínimo local"
        elif det > 0 and f_xx < 0:
            return "Máximo local"
        elif det < 0:
            return "Punto de silla"
        else:
            return "Criterio inconcluso"
    
    def grandiente_conjugado(self,e2,e3,optimizador):#e2 y e3 son los otros optimizadores 
        optimizador=self.optimizer(optimizador)
        x_inicial=np.array([self.a,self.b])
        gradiente=np.array(self.gradiente_calculation(x_inicial))
        s_inicial=gradiente * -1
        alfa_actual=self.alfa_dadopunto(x_inicial,optimizador)

        x_actual= x_inicial + (alfa_actual * s_inicial)
        v=(np.linalg.norm(self.gradiente_calculation(x_actual))**2/np.linalg.norm(self.gradiente_calculation(x_inicial))**2)
        #print(v*s_inicial)
        x_siguiente= ((-1 * np.array(self.gradiente_calculation(x_actual)))) + (((np.linalg.norm(self.gradiente_calculation(x_actual))**2)/
                                                              (np.linalg.norm(self.gradiente_calculation(x_inicial))**2)) * s_inicial)
        while ((np.linalg.norm(x_siguiente - x_actual))/np.linalg.norm(x_actual)) >= e2 or  np.linalg.norm(self.gradiente_calculation(x_siguiente)) >= e3:
            print("Sin terminar")
            #print((np.linalg.norm(x_siguiente - x_actual))/np.linalg.norm(x_actual))
            x_actual=x_siguiente
            x_siguiente=0
            gradiente=np.array(self.gradiente_calculation(x_actual))
            s_actual=gradiente * -1
            alfa_actual=self.alfa_dadopunto(x_actual,optimizador)
            x_actual_nuevo= x_actual + (alfa_actual * s_actual)
            #print(x_actual_nuevo)
            x_siguiente= (-1 * np.array(self.gradiente_calculation(x_actual_nuevo))) + ((((np.linalg.norm(self.gradiente_calculation(x_actual_nuevo))**2)/
                                                                (np.linalg.norm(self.gradiente_calculation(x_actual))**2))) * s_actual)
            x_actual= x_actual_nuevo
            print(x_siguiente)
            #print((np.linalg.norm(x_siguiente - x_actual))/np.linalg.norm(x_actual))
            print(np.linalg.norm(self.gradiente_calculation(x_siguiente)))
        return x_siguiente
    

        

if __name__== "__main__":
    import matplotlib.pyplot as plt
    def himmelblau(p):
        return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2
    
    a=0
    b=2
    e=0.001
    opt=cauchy_6_junio(a,b,e,himmelblau)
    print((opt.funcion))
    print(opt.grandiente_conjugado(e,e,'golden'))