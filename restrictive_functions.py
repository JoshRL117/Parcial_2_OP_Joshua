import numpy as np 
#(((x[0] - 1  )**3 - x[1] + 1))>= 0 and (x[0] + x[1] - 2) <= 0
def rosenbrooklinecubic(x): 
    if (((x[0] - 1  )**3 - x[1] + 1))>= 0 and (x[0] + x[1] - 2) <= 0: 
        return np.array([(1 - x[0])**2 + 100 * (x[1] - (x[0]**2))**2, (((x[0] - 1  )**3 - x[1] + 1))>= 0 and (x[0] + x[1] - 2)])
    else: 
        print("VALUE ERROR  the function is out of restriction")


def rosenbrookdisk(x): 
    if (x[0]**2 + x[1] ** 2 ) <= 2:
        return np.array([(1 - x[0])**2 + 100 * (x[1] - (x[0]**2))**2,(x[0]**2 + x[1] ** 2 )])
    else: 
        print("VALUE ERROR  the function is out of restriction")



class restrictivefunctios: 
    def __init__(self) -> None:
        pass