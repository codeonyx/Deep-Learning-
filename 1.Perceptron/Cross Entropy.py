import numpy as np

#The function takes in two lists. One Y and another P
#The list Y specifies the output and the P means the probability


def cross_entropy(Y,P):
    Y=np.float_(Y)
    P=np.float_(P)

    return -np.sum(Y*np.log(P)+(1-Y)*np.log((1-P)))