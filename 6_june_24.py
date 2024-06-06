import numpy as np 
import central_differentation as cdif

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

def w_to_x(w:float, a,b )-> float:
    return w * (b-a) + a

def busquedadorada(f,e:float,a:float=None, b:float=None)->float:
    PHI=(1 + np.sqrt(5))/ 2-1
    aw,bw=0,1 
    Lw=1
    k=1# index del actual 
    while Lw> e: 
        w2= aw + PHI* Lw
        w1=bw - PHI * Lw
        aw,bw=findregions(w1,w2,aw,bw,f)
        k+=1 
        Lw=bw-aw
    
    return (f(aw,a,b) + f(bw,a,b))/2 

def gradiente_calculation(x,f,delta=0.001):
    vector_f1_prim=[]
    x_work=np.array(x)
    x_work_f=x_work.astype(np.float64)
    if isinstance(delta,int) or isinstance(delta,float):
        for i in range(len(x_work_f)):
            point=np.array(x_work_f,copy=True)
            vector_f1_prim.append(primeraderivadaop(point,i,delta,f))
        return vector_f1_prim
    else:
        for i in range(len(x_work_f)):
            point=np.array(x_work_f,copy=True)
            vector_f1_prim.append(primeraderivadaop(point,i,delta[i],f))
        return vector_f1_prim

def primeraderivadaop(x,i,delta,f):
    mof=x[i]
    p=np.array(x,copy=True)
    p2=np.array(x,copy=True)
    nump1=mof + delta
    nump2 =mof - delta
    p[i]= nump1
    p2[i]=nump2
    numerador=f(p) - f(p2)
    return numerador / (2 * delta) 

def cauchy(x_inicial,e1,e2,M,f,optimizador=busquedadorada):#e son los epsilon y M es el numero de iteraciones 
    stop=False
    xk=x_inicial
    k=0
    while not stop: 
        gradiente=np.array(gradiente_calculation(xk,f))
        if np.linalg.norm(gradiente)< e1 or k >=M:
            stop=True 
        else: 
            def alpha_funcion(alpha):
                return xk - alpha*gradiente
            alfa=optimizador(f,e2)
            x_k1= xk - alfa * gradiente#Punto siguiente a xk
            if np.linalg.norm((x_k1-xk))/ (np.linalg.norm(xk) + 0.0000001 )<= e2:
                stop=True 
            else:
                k+=1
                xk=x_k1
    return xk



