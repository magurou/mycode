import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from torchviz import make_dot
from IPython import display
from sklearn.datasets import load_boston
from torch import nn

def func(x):
    #print(np.log(x))
    return 1 / (1 + x ** 2)
def quad(y, cut):
    n = y.size - 1
    #print(n)
    h = (x[-1] - x[0]) / cut
    #print(h)
    i = y[0] / 2 + y[-1] / 2
    #print(i)
    for j in range(1, n):
        i += y[j]
        #print(y[j])
    #print(i)
    return i * h
def simp(y, cut):
    i = y[0] + y[-1]
    n = y.size - 1
    #print(y.size)
    h = (x[-1] - x[0]) / cut
    #print(h)
    #print(n)
    for num in range(1, n):
        if(num % 2 == 0):
            i += 2 * y[num]
            #print('偶数')
            #print(y[num])
        else :
            i += 4 * y[num]
            #print('奇数')
            #print(y[num])
    #print(i)
    return i * h / 3 
def romb(j, k, r):
    if(k == 1):
        return (4 * r[j] - r[j - 1]) / (4 - 1)
    
    return ((4 ** k) * romb(j, k - 1, r) - romb(j - 1, k - 1, r)) / ((4 ** k) - 1)
r = list()
for l in range(0, 3):
    n  = 2 ** l
    x = np.arange(0, 1 + 0.001, 1 / n)
    y = func(x)
    print(y)
    print()
    print(quad(y, n))
    r.append(quad(y, n))

#print(romb(0, 1, r))
    
#print(x)
#print(y)
#print(simp(y, n))

#print(quad(y, n))

