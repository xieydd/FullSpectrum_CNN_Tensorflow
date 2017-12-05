# -*- coding:utf-8 -*-
#@Description: 基于全矢希尔伯特的CNN模型
#@author xieydd xieydd@gmail.com
#@date 2017-12-02 下午16:50:51
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPool2D,GlobalAveragePooling2D,Dropout
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import plot_model
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow,imsave

import pydot
from IPython.display import SVG
from keras.utils import plot_model

import os  
import numpy as np  
import tensorflow as tf   
import pandas as pd  
from sklearn.model_selection import train_test_split

#基于全矢希尔伯特的CNN模型方法实现
def fv_hibert_CNN(input_shape):
    
    with K.name_scope('CustomLayer'):
        X_input = Input(input_shape)

        X = ZeroPadding2D((1,1))(X_input)
        X = Conv2D(2,(7,7),strides=(1,1),name='conv0')(X)
        X = BatchNormalization(axis=3,name='bn0')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2,2),name='name_pool1')(X)

        X = ZeroPadding2D((3,3))(X_input)
        X = Conv2D(2,(7,7),strides=(1,1),name='conv1')(X)
        X = BatchNormalization(axis=3,name='bn1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2,2),name='name_pool1')(X)

        X = Flatten()(X)
        X = Dense(1024,activation='relu',name='fc0')(X)
        X = Dropout(0.5)(X)
        X = Dense(9,activation='softmax',name='fc1')(X)
        model = Model(input=X_input,outputs=X,name='fv_hibert_CNN')

    return model


######################################  
# TODO: set the gpu memory using fraction #  
#####################################  
def get_session(gpu_fraction=0.6):  
    """ 
    This function is to allocate GPU memory a specific fraction 
    Assume that you have 6GB of GPU memory and want to allocate ~2GB 
    """  
  
    num_threads = os.environ.get('OMP_NUM_THREADS')  
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)  
  
    if num_threads:  
        return tf.Session(config=tf.ConfigProto(  
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))  
    else:  
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
KTF.set_session(get_session(0.6))  # using 60% of total GPU Memory  
os.system("nvidia-smi")  # Execute th2e command (a string) in a subshell
#input("Press Enter to continue...")




# 第一次遍历图片目录是为了获取图片总数  
input_count = 0  
for i in range(0,9):  
    dir = 'J:/1/啦啦啦/sample/%s/' % i               # 这里可以改成你自己的图片目录，i为分类标签  
    for rt, dirs, files in os.walk(dir):  
        for filename in files:  
            input_count += 1  
#定义对应维度与各维度长度的数组
#定义输入数据
input_images=np.array([[0]*1024 for i in range(input_count)],dtype=float)
#定义标签1
input_lables=np.array([[0]*9 for i in range(input_count)],dtype=int)
#第二次遍文件目录是为了生成文件数据和标签
index=0
for i in range(0,9):
    dir= 'J:/1/啦啦啦/sample/%s/' % i     
    for rt,dirs,files in os.walk(dir):
        for filename in files:
#            找到第i个类别下的样本名
            filename=dir+filename
            #通过读取样本名进而读取矩阵                            
            input_images_single=pd.read_table(filename,header=None)
            input_images_single_array=np.array(input_images_single).flatten()
            size=input_images_single_array.shape[0]
            #将单个样本传入矩阵
            for j in range (0,size):
                input_images[index][j]=input_images_single_array[j]
#            input_images[index]=input_images_single_array                           
            input_lables[index][i]=1
#            print(i)
            index+=1

#对每一个batch进行记录loss
'''
def write_log(callback, names, logs, batch_no):  
    for name, value in zip(names, logs):  
        summary = tf.Summary()  
        summary_value = summary.value.add()  
        summary_value.simple_value = value  
        summary_value.tag = name  
        callback.writer.add_summary(summary, batch_no)  
        callback.writer.flush()


train_names = ['train_loss', 'train_mae']  
val_names = ['val_loss', 'val_mae']  
for batch_no in range(100):  
    logs = model.train_on_batch(X_train, Y_train)  
    write_log(callback, train_names, logs, batch_no)  
      
    if batch_no % 10 == 0:  
        X_val, Y_val = np.random.rand(32, 3), np.random.rand(32, 1)  
        logs = model.train_on_batch(X_val, Y_val)  
        write_log(callback, val_names, logs, batch_no)   
        #  batch_no//10
'''


x_image=tf.reshape(input_images,[-1,32,32,1])
#print(x_image)
#print(x_image[1].shape)
embedding_layer_names = set(layer.name
                            for layer in fv_hibert_CNN_Model.layers)
                            #if layer.name.startswith('dense_')
tb_cb = TensorBoard(log_path, histogram_freq=10, batch_size=32,
                           write_graph=True, write_grads=True, write_images=True,
                           embeddings_freq=10, embeddings_metadata=None,
                           embeddings_layer_names=embedding_layer_names)
cbks = [tb_cb]

fv_hibert_CNN_Model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics= ['accuracy'])
#fv_hibert_CNN_Model.fit(x=x_image,y=input_lables,epochs=40,verbose=1,validation_split=0.2,batch_size=16)
history = fv_hibert_CNN_Model.fit(x=x_image,y=input_lables,epochs=40,verbose=1,validation_split=0.2,batch_size=16)
fv_hibert_CNN_Model.summary()
tb_cb.set_model(fv_hibert_CNN_Model)

####画出train和val的loss图(横轴是 epoch)###
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
