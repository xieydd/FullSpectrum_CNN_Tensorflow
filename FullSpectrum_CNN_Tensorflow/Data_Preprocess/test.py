from data import *
from fv_hibert import *
M1,M2 =readFile('H:/bearing data/bearing_IMS/1','E:/FullSpectrum_CNN_Tensorflow/FullSpectrum_CNN_Tensorflow/Data_Preprocess')
x_amplitude_envelope,y_amplitude_envelope = hibert(M1,M2)
vm,vs,vr,alpha,phase = fv_hibert(x_amplitude_envelope,y_amplitude_envelope,1,0,0.05)