# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy import *

#读取轴承文件
def readFile(path,txt_path):
    files = os.listdir(path)
    M1 = []
    M2 = []
    for filename in files:
        # 打开文件（注意路径）
        f = open(path+'/'+filename)
        # 逐行进行处理
        first_ele = True
        for data in f.readlines():
            ## 去掉每行的换行符，"\n"
            data = data.strip('\n')
            ## 按照 空格进行分割。
            nums = np.array(data.split("\t"),dtype=float)
            ## 添加到 matrix 中。
            if first_ele:
                ### 将字符串转化为整型数据
                nums = [x for x in nums ]
                ### 加入到 matrix 中 。
                matrix = np.array(nums)
                first_ele = False
            else:
                nums = [x for x in nums]
                matrix = np.c_[matrix,nums]
        matrix = matrix.transpose()
        matrix1 = [x[0] for x in matrix]
        matrix2 = [x[1] for x in matrix]
        M1.append(matrix1)
        M2.append(matrix2)
    M1 = np.matrix(M1)
    M2 = np.matrix(M2)
    #将矩阵转换成csv文件便于以后的读取
    m1 = pd.DataFrame(M1)
    m1.to_csv(txt_path+'/m1.csv')

    m2 = pd.DataFrame(M2)
    m2.to_csv(txt_path+'/m2.csv')

    return M1,M2

#画出2156组数据的均方值和峭度matrix为一个通道的(2156,20480)
def plot_Kurtosis_rms(matrix):
    R = []
    H = []
    for i in range(matrix.shape[0]):
        r = np.sqrt(np.sum(np.square(matrix[i]))/matrix.shape[1])
        h = pd.Series(matrix[i].flatten()).kurt()
        R.append(r)
        H.append(h)
    #print(R)
    #print(H)
    plt.subplot(2,1,1)
    plt.plot(np.array(R).flatten(),'or')
    plt.ylabel('均方值')
    
    
    plt.subplot(2,1,2)
    plt.plot(np.array(H).flatten(),'ob')
    plt.ylabel('峭度值')
    plt.xlabel('数据组/共2156组');
    plt.show()