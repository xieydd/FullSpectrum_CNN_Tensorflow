# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:32:08 2017

@author: xieydd
"""
import numpy as np
import pandas as pd
from scipy import *
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as signals
from scipy import fftpack

#保证图像中文问题和符号问题
matplotlib.rcParams['axes.unicode_minus']=False
plt.rc('font', family='SimHei', size=13)

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
    
 #希尔伯特   
def hibert(x,y):
    x = x[chioce_num,0:20000]
    y = y[chioce_num,0:20000]
    
    #X_Hilbert包络谱
    x_signal = np.array(x).flatten()#展成一维
    x_analytic_signal = signals.hilbert(x_signal)#希尔伯特变换
    x_amplitude_envelope = np.abs(x_analytic_signal)
    x_instantaneous_phase = np.unwrap(np.angle(x_analytic_signal))#瞬时相位
    x_instantaneous_frequency = (np.diff(x_instantaneous_phase)/(2.0*np.pi) * fs)#瞬时频率
    
    x_signal_fft = np.abs(fftpack.fft(x_analytic_signal)/10000)
    f = [i*fs/N for i in range(10000)]
    
    fig1 = plt.figure(figsize=(12,12))
    ax0 = fig1.add_subplot(211)
    ax0.plot(t, x_signal, label='signal')
    ax0.plot(t, x_amplitude_envelope, label='envelope')
    ax0.set_xlabel("时间/s")
    ax0.set_ylabel('加速度m/s^2')
    ax0.set_title('X通道希尔伯特包络')
    ax0.legend()
    #Y_Hilbert包络谱
    y_signal = np.array(y).flatten()#展成一维
    y_analytic_signal = signals.hilbert(y_signal)#希尔伯特变换
    y_amplitude_envelope = np.abs(y_analytic_signal)
    y_instantaneous_phase = np.unwrap(np.angle(y_analytic_signal))#瞬时相位
    y_instantaneous_frequency = (np.diff(y_instantaneous_phase)/(2.0*np.pi) * fs)#瞬时频率
    
    y_signal_fft = np.abs(fftpack.fft(y_analytic_signal)/10000)
    f = [i*fs/N for i in range(10000)]
    
    ax1 = fig1.add_subplot(212)
    ax1.plot(t, y_signal, label='signal')
    ax1.plot(t, y_amplitude_envelope, label='envelope')
    ax1.set_xlabel("时间/s")
    ax1.set_ylabel('加速度m/s^2')
    ax1.set_title('Y通道希尔伯特包络')
    ax1.legend()
    
    fig2 = plt.figure(figsize=(12,12))
    ax0 = fig2.add_subplot(211)
    ax0.plot(t[1:], x_instantaneous_frequency)
    ax0.set_xlabel("时间/s")
    ax0.set_ylabel("瞬时频率/Hz")
    ax0.set_title('X通道瞬时频率')
    
    ax1 = fig2.add_subplot(212)
    ax1.plot(t[1:], y_instantaneous_frequency)
    ax1.set_xlabel("时间/s")
    ax1.set_ylabel("瞬时频率/Hz")
    ax1.set_title('Y通道瞬时频率')
    
    fig3 = plt.figure(figsize=(12,12))
    ax0 = fig3.add_subplot(211)
    ax0.plot(f,x_signal_fft[0:10000])
    ax0.set_ylim(0.0,0.1)
    ax0.set_xlabel("频率/Hz")
    ax0.set_ylabel("加速度m/s^2")
    ax0.set_title('X通道Hiblert频谱')
    
    ax1 = fig3.add_subplot(212)
    ax1.plot(f,y_signal_fft[0:10000])
    ax1.set_ylim(0.0,0.1)
    ax1.set_xlabel("频率/Hz")
    ax1.set_ylabel("加速度m/s^2")
    ax1.set_title('Y通道Hiblert频谱')
    
    return x_amplitude_envelope,y_amplitude_envelope
   
'''
全矢希尔伯特  输入x_amplitude_envelope,y_amplitude_envelope
y为0时即单通道数据
vm,vs,vr,alpha,phase,wave_time:分别为主振矢，副振矢，振矢比，振矢角，矢相位和时域融合结果
'''
def fv_hibert(x,y):
    
    