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
import inspect
import math
from sympy import Symbol, exp, I

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
    x = x[chioce_num,:]
    y = y[chioce_num,:]
    
    #X_Hilbert包络谱
    x_signal = np.array(x).flatten()#展成一维
    x_analytic_signal = signals.hilbert(x_signal)#希尔伯特变换
    x_amplitude_envelope = np.abs(x_analytic_signal)
    x_instantaneous_phase = np.unwrap(np.angle(x_analytic_signal))#瞬时相位
    x_instantaneous_frequency = (np.diff(x_instantaneous_phase)/(2.0*np.pi) * fs)#瞬时频率
    
    x_signal_fft = np.abs(fftpack.fft(x_analytic_signal)/2048)
    f = [i*fs/N for i in range(2048)]
        
    
    fig1 = plt.figure(figsize=(12,12))
    ax0 = fig1.add_subplot(211)
    ax0.plot(t[0:2048], x_signal, label='signal')
    ax0.plot(t[0:2048], x_amplitude_envelope, label='envelope')
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
    
    y_signal_fft = np.abs(fftpack.fft(y_analytic_signal)/2048)
    f = [i*fs/N for i in range(2048)]
    
    ax1 = fig1.add_subplot(212)
    ax1.plot(t[0:2048], y_signal, label='signal')
    ax1.plot(t[0:2048], y_amplitude_envelope, label='envelope')
    ax1.set_xlabel("时间/s")
    ax1.set_ylabel('加速度m/s^2')
    ax1.set_title('Y通道希尔伯特包络')
    ax1.legend()
    
    fig2 = plt.figure(figsize=(12,12))
    ax0 = fig2.add_subplot(211)
    ax0.plot(t[0:2047], x_instantaneous_frequency)
    ax0.set_xlabel("时间/s")
    ax0.set_ylabel("瞬时频率/Hz")
    ax0.set_title('X通道瞬时频率')
    
    ax1 = fig2.add_subplot(212)
    ax1.plot(t[0:2047], y_instantaneous_frequency)
    ax1.set_xlabel("时间/s")
    ax1.set_ylabel("瞬时频率/Hz")
    ax1.set_title('Y通道瞬时频率')
    
    fig3 = plt.figure(figsize=(12,12))
    ax0 = fig3.add_subplot(211)
    ax0.plot(f[0:2048],x_signal_fft[0:2048])
    ax0.set_ylim(0.0,0.1)
    ax0.set_xlabel("频率/Hz")
    ax0.set_ylabel("加速度m/s^2")
    ax0.set_title('X通道Hiblert频谱')
    
    ax1 = fig3.add_subplot(212)
    ax1.plot(f[0:2048],y_signal_fft[0:2048])
    ax1.set_ylim(0.0,0.1)
    ax1.set_xlabel("频率/Hz")
    ax1.set_ylabel("加速度m/s^2")
    ax1.set_title('Y通道Hiblert频谱')
    
    return x_amplitude_envelope,y_amplitude_envelope
   
'''
全矢希尔伯特  输入x_amplitude_envelope,y_amplitude_envelope
xdata,ydata是两通道数据，当ydata数据为空或全为0时认为是单通道数据
dir_sensor:水平方向传感器到垂直方向传感器的转向与旋转方向一致为1，相反为－1；
angle_x:水平方向传感器与水平方向的夹角
eps：计算误差要求
vm,vs,vr,alpha,phase,wave_time:分别为主振矢，副振矢，振矢比，振矢角，矢相位和时域融合结果
'''
from sympy import Symbol, exp, I
def fv_hibert(xdata,ydata,dir_sensor,angle_x,eps):
    
    #输入变量检测
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    nargin = len(args)
    if nargin<1:
        print('输入变量数不能小于1')
    elif nargin==1:
        n=len(xdata)
        ydata=np.zeros((n,1))
        dir_sensor=1
        angle_x=0
        eps=0.05
    elif nargin==2:
        dir_sensor=1
        angle_x=0
        eps=0.05
    elif nargin==3:
        angle_x=0
        eps=0.05
    elif nargin==4:
        eps=0.05
     
    #判断输入单通道还是双通道
    n =len(xdata)
    n_half = int(n/2)
    flag_channel = 2
    if ydata.all() == None:
        ydata = zeros((n,1))
        flag_channel = 1
    
    #flag位为1时为正变换，为-1时为反变换
    flag=1
    
    vm = np.zeros((n_half,1))#定义主振矢
    vs = np.zeros((n_half,1))#定义副振矢
    alpha = np.zeros((n_half,1))#定义振矢角
    rvN_k = np.zeros((n_half,1)) 
    ivN_k = np.zeros((n_half,1)) 
    phase = np.zeros((n_half,1)) #定义矢相位，正进动相位角，反进动相位角
    faia = np.zeros((n_half,1))
    faib = np.zeros((n_half,1))
    xdata = xdata-np.mean(xdata)
    ydata = ydata-np.mean(ydata)

    xdata.reshape(2048,1)
    z = np.zeros((len(xdata),1),dtype=complex)
    
    for i in range(len(xdata)):
        z[i] =complex(xdata[i],round(float(ydata[i]),4))
    
    
    Z = 2*fftpack.fft(z)/n
    rv=real(Z)
    iv=imag(Z)
    rvk = rv[0:n_half]
    ivk = iv[0:n_half]
    
    
    for i in range(0,n_half):
        rvN_k[i]=rv[n-i-1]
        ivN_k[i]=iv[n-i-1]
    vm[1]=0 #主振矢
    vs[1]=0 #副振矢
    alpha[1]=0 #振矢角
    Xck=(rvN_k+rvk)/2
    Xsk=(ivk-ivN_k)/2
    Yck=(ivk+ivN_k)/2
    Ysk=(rvN_k-rvk)/2
    
    zv = np.zeros(shape=(len(iv),1),dtype=complex)
    for j in range(len(iv)):
        zv[j] = complex(rv[j],round(float(iv[j]),4))   
    
    xp=0.5*np.abs(zv[2:n_half])      #正进动幅值序列
    mxr=0.5*np.abs(zv[n_half+2:n])   #反进动幅值序列所需中间变量
    nn=len(mxr)            #反进动幅值序列长度
    xr=np.zeros((nn,1))             #反进动幅值序列
    tr=np.zeros((nn,1))
    mmivN_k=np.zeros((nn,1))
    mmrvN_k=np.zeros((nn,1))
    
    tanpk=iv[2:n_half]/rv[2:n_half] #正进动相位角
    mtr=iv[n_half+2:n]/rv[n_half+2:n] #反进动相位角
    for i in range(1,nn):
        xr[i]=mxr[nn-i-1]      #反进动幅值序列
        tr[i]=mtr[nn-i-1]      #反进动相位角反向排序
        mmivN_k[i]=iv[n-i-1]
        mmrvN_k[i]=rv[n-i-1]
        
    #求主振矢、副振矢、振矢比
    vm[2:n_half]=xp+xr           #求主振矢
    vs[2:n_half]=dir_sensor*(xp-xr) #求副振矢，考虑传感器安装方向与转速方向
    vr=vs/vm   #振矢比的值域为[-1，1]
    bb=(iv[2:n_half]*mmrvN_k+mmivN_k*rv[2:n_half])#./(rv(2:n/2).*mmrvN_k);
    aa=(rv[2:n_half]*mmrvN_k-iv[2:n_half]*mmivN_k)#./(rv(2:n/2).*mmrvN_k);
    
    atan2a = np.zeros((len(bb),1))
    for i in range(len(bb)):
        atan2a[i]=math.atan((bb/aa)[i])
    
    #根据2a所在象限调整2a的值
    for i in range(0,n_half-2):
        if (aa[i]<0 and bb[i]<0):                #2a位于第三象限时
            atan2a[i]=atan2a[i]+pi
        elif (aa[i]<0 and bb[i])>0:            #2a位于第二象限时
            atan2a[i]=atan2a[i]+pi
        elif (aa[i]>0 and bb[i]<0):            #2a位于第四象限时
            atan2a[i]=atan2a[i]+2*pi
    
    #计算振矢角
    alpha[2:n_half]=0.5*atan2a*180/pi            #通过2a算振矢角a，单位：角度,值域为[0，180]
    alpha=alpha+angle_x                       #把相角从与X方向夹角变换到与水平方向夹角
    for i in range(0,n_half-1):
        if (alpha[i]>180):
            alpha[i]=alpha[i]-180
    #计算矢相位
    for i in range(0,n_half-1):
        faia[i]=math.atan2(Xsk[i]*cos(alpha[i])+Ysk[i]*sin(alpha[i]),Xck[i]*cos(alpha[i])+Yck[i]*sin(alpha[i])) # 设xr1=vm*cos(omega*t+faia1)
        faib[i]=math.atan2(-Xsk[i]*sin(alpha[i])+Ysk[i]*cos(alpha[i]),-Xck[i]*sin(alpha[i])+Yck[i]*cos(alpha[i])) #设yr1=vs*cos(omega*t+faib1)  把椭圆方程化成标准形式
    fai=faia-faib
    phase = np.zeros((len(ivk),1))
    for i in range(len(ivk)):
        phase[i]=math.atan((ivk/rvk)[i])*180/pi #矢谱分析技术中的相位角
    
    #根据相角所在象限调整矢相角值
    for i in range(0,n_half-1):                     #根据相角所在象限调整矢相角值，目的是使相位角始终位于[0，2*pi]         
        if (ivk[i]>0 and rvk[i]<0):      #当相角位于第二象限时
            phase[i]=phase[i]+180
        elif (ivk[i]<0 and rvk[i]<0): #当相角位于第三象限时  
            phase[i]=phase[i]+180
        elif (ivk(i)<0 and rvk(i)>0):  #当相角位于第四象限
            phase[i]=phase[i]+2*180
        else:                       #当相角位于第一象限时
            phase[i]=phase[i]
        phase=phase+angle_x        #把相角从与X方向夹角变换到与水平方向夹角   
        for i in range(0,n_half-1):
            if(phase[i]>360):
                phase[i]=phase[i]-360
        
        #计算融合后的时域波形图
        wave_time=np.zeros((n,2))
        
        vs_a = np.zeros(shape=(len(vs),1),dtype=complex)
        for j in range(len(vs)):
            vs_a[j] = complex(0,round(float(vs[j]),4))    
        
        Xvr=vm+vs_a
        Xv=np.zeros((n,1))
        Xv[2:n_half]=Xvr[2:len(Xvr)]
        for i in range(2,n_half-1):
            Xv[n-i+1]=vm[i]-complex(0,round(float(vs[i]),4))
        wave_time=fftpack.ifft(Xv)*n_half
        
        #根据误差限调整各参数的值
        maxvm=np.max(vm[2:n_half])
        for ii in range(0,n_half):
            if (vm[ii]<eps*maxvm):
                alpha[ii]=0
                phase[ii]=0
                vs[ii]=0
                vr[ii]=0
            if (np.abs(alpha[ii])<eps):
                alpha[ii]=0
                phase[ii]=0
                
                
        #有五个输出参数vm，vs，vr,alpha,phase，vm,vs,vr，alpha分别为主、副振矢、振矢比与振矢角，alpha为2个变量的细胞数组，其中vs（1，1）保存矢相位，vm（2，1）保存的是时域融合结果！！！     
        aa = phase
        phase = {}
        phase["aa"] = aa
        phase["wave_time"] = wave_time
        
        #全矢希尔伯特图
        fig1 = plt.figure(figsize=(12,12))
        ax0 = fig1.add_subplot(211)
        f = np.arange(0,n_half,1)
        ax0.plot(f[0:],vm)
        ax0.set_xlabel("频率/Hz")
        ax0.set_ylabel("'加速度m/s^2")
        ax0.set_title('全矢Hilbert解调信号')
        return vm,vs,vr,alpha,phase