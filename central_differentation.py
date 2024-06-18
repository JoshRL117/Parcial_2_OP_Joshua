import numpy as np 

def gradiente_calculation(x,delta,f):
    if delta == None: 
        delta=0.00001
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
def segundaderivadaop(x,i,delta,f):
    mof=x[i]
    p=np.array(x,copy=True)
    p2=np.array(x,copy=True)
    nump1=mof + delta
    nump2 =mof - delta
    p[i]= nump1
    p2[i]=nump2
    numerador=f(p) - (2 * f(x)) +  f(p2)
    return numerador / (delta**2) 

def derivadadodadoop(x,index_principal,index_secundario,delta,f):
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
        numerador=((f(p)) - f(p2) - f(p3) + f(p4))
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
        numerador=((f(p)) - f(p2) - f(p3) + f(p4))
        return numerador / (4*delta)

    
def hessian_matrix(x,delt,f):# x es el vector de variables
    matrix_f2_prim=[([0]*len(x)) for i in range(len(x))]
    x_work=np.array(x)
    x_work_f=x_work.astype(np.float64)
    for i in range(len(x)):
        point=np.array(x_work_f,copy=True)
        for j in range(len(x)):
            if i == j:
                matrix_f2_prim[i][j]=segundaderivadaop(point,i,delt,f)
            else:
                matrix_f2_prim[i][j]=derivadadodadoop(point,i,j,delt,f)
    return matrix_f2_prim

def minormax(hessian):
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
def himmelblau(p):
    return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2

if __name__ == "__main__":
    delt=0.01
    entrada=[1,1]
    g=(gradiente_calculation(entrada,delt,himmelblau))
    h=(hessian_matrix(entrada,delt,himmelblau))
    print(g)


