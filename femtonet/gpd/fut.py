import numpy as np

def fut(xval,erbl,x):

    emme = np.zeros(490)
    enne = np.zeros(490)

    for i in range(490):
        y1 = erbl[i]
        y2 = erbl[i+1]
        x1 = xval[i]
        x2 = xval[i+1]

        emme[i] = (y2 - y1)/(x2 - x1)
        enne[i] = y1 - emme[i]*x1

    ret = 0
    
    for i in range(490):

        if x >= xval[i] and x <= xval[i+1]:

            ret = emme[i]*x + enne[i]
            
    return ret
