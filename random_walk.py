import numpy as np
#random_value = np.random.normal(mu, sigma)
def step_calculation(x):
    mu=0
    stddev=1
    random_value = np.random.normal(mu, stddev)
    return x + random_value

def testfunction(x):
    return x[0]**2 + x[1]**2
def randomwalk(x,f,stop):
    x=np.array(x)
    xmejor=x[0]
    cont=0
    while(stop > cont) or cont > len(x):
        x_nuevo=randomwalk(x[cont])
        if f(x_nuevo)< f(xmejor):
            xmejor=x_nuevo
    return xmejor

def hillclimbing(x,f,stop):
    x=np.array(x)
    xactual=x[0]
    climbing=[]
    climbing.append(xactual)
    cont=0
    while(stop > cont) or cont > len(x):
        x_nuevo=randomwalk(x[cont])
        if f(x_nuevo)< f(xactual):
            xactual=x_nuevo
            climbing.append(xactual)
    return xactual

