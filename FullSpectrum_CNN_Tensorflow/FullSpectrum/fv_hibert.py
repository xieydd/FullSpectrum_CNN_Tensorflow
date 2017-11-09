# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:32:08 2017

@author: xieydd
"""
import numpy as np
import pandas as pd
from scipy import *
import matplotlib.pyplot as plt

N = 20000
fs = 20000
t = np.linspace(0/fs,N/fs,N)
chioce_num = 1

#时域信号
def time(x,y):
    x = x[chioce_num,0:20000]
    y = y[chioce_num,0:20000]
    plt.subplot(2,1,1)
    plt.plot(t,np.array(x).flatten(),'r')
    plt.ylabel('加速度m/s^2')
    plt.xlabel('时间/s')#绘出Nyquist频率之前随频率变化的振幅
    plt.title('X通道时域信号')
    
    plt.subplot(2,1,2)
    plt.plot(t,np.array(y).flatten(),'r')
    plt.ylabel('加速度m/s^2')
    plt.xlabel('时间/s')#绘出Nyquist频率之前随频率变化的振幅
    plt.title('Y通道时域信号')
    plt.show()
    
    
def fv_hibert(x,y):
    x = x[chioce_num,0:20000]
    y = y[chioce_num,0:20000]
    #X_Hilbert包络谱
    