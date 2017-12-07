# -*- coding:utf-8 -*-
#@Description: 基于全矢希尔伯特的CNN模型
#@author xieydd xieydd@gmail.com
#@date 2017-12-02 下午16:50:50
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPool2D,GlobalAveragePooling2D,Dropout
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import plot_model
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint,EarlyStopping


import pydot
from IPython.display import SVG
import time, pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow,imsave

import os  
import numpy as np  
import tensorflow as tf   
import pandas as pd  
from sklearn.model_selection import train_test_split



######################################  
# TODO: 基于全矢希尔伯特的CNN模型方法实现 输入的是[-1,32,32,1]  
#####################################
def fv_hibert_CNN(input_shape):
    
    with K.name_scope('CustomLayer'):
        X_input = Input(input_shape)
        '''
        TODO 这里发现加了BatchNormalization会使val_acc和val_loss莫名的很大的波动
        使用tanh>relu>sigmoid
        应该不用在加层数，学习效果已经不错了
        '''
        #X = ZeroPadding2D((1,1))(X_input)
        X = Conv2D(6,(5,5),strides=(1,1),name='conv0',padding='same')(X_input)
        X = BatchNormalization(axis=2,name='bn0')(X)
        X = Activation('tanh')(X)
        X = MaxPooling2D((2,2),strides=(2,2),name='pool0',padding='same')(X)

        #X = ZeroPadding2D((3,3))(X_input)
        X = Conv2D(12,(3,3),strides=(1,1),name='conv1',padding='same')(X)
        X = BatchNormalization(axis=2,name='bn1')(X)
        X = Activation('tanh')(X)
        X = MaxPooling2D((2,2),strides=(2,2),name='pool1',padding='same')(X)

        X = Flatten()(X)
        #加入,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)发现val_loss和loss下降，但是val_acc和acc都先上升后下降
        X = Dense(100,activation='tanh',kernel_initializer='he_normal',name='dense_fc0')(X)
        X = Dropout(0.5)(X)
        X = Dense(9,activation='softmax',kernel_initializer='he_normal',name='dense_fc1')(X)
        model = Model(input=X_input,outputs=X,name='fv_hibert_CNN')

    return model

######################################  
# TODO: 获取全部数据input和labels filename暂定J:/1/啦啦啦/sample  
#####################################
def loadData(filename):
    # 第一次遍历文件目录是为了获取文件总数  
    input_count = 0  
    for i in range(0,9):  
        dir = 'filename/%s/' % i               # 这里可以改成你自己的图片目录，i为分类标签  
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
        dir= 'filename/%s/' % i     
        for rt,dirs,files in os.walk(dir):
            for filename in files:
                #找到第i个类别下的样本名
                filename=dir+filename
                #通过读取样本名进而读取矩阵                            
                input_images_single=pd.read_table(filename,header=None)
                input_images_single_array=np.array(input_images_single).flatten()
                size=input_images_single_array.shape[0]
                #将单个样本传入矩阵
                for j in range (0,size):
                    input_images[index][j]=input_images_single_array[j]                          
                input_lables[index][i]=1
                index+=1
    return input_images,input_lables


######################################  
# TODO: 设置GPU使用量
'''
在Tensorflow中可以直接通过 
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True #设置最小GPU使用量
session = tf.Session(config=config)
'''
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



######################################  
# TODO: 多GPU并行计算
#####################################
def make_parallel(model, gpu_count):

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    # 也可以设为GPU，如果CPU负载已经很大的话
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(Concatenate(axis=0)(outputs))
        return Model(model.inputs, merged)



######################################  
# TODO: 对每一个batch进行记录loss
'''
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
#####################################
def write_log(callback, names, logs, batch_no):  
    for name, value in zip(names, logs):  
        summary = tf.Summary()  
        summary_value = summary.value.add()  
        summary_value.simple_value = value  
        summary_value.tag = name  
        callback.writer.add_summary(summary, batch_no)  
        callback.writer.flush()


######################################  
# TODO: 画出train和val的loss图(横轴是 epoch)
#####################################
def accuracy_curve(h):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()






KTF.set_session(get_session(0.8))  # using 80% of total GPU Memory  
os.system("nvidia-smi")  # Execute th2e command (a string) in a subshell
input("Press Enter to continue...")

log_path = 'D:/Graph'
filename = "J:/1/啦啦啦/sample"
input_images,input_lables = loadData(filename)
x_train,x_test,y_train,y_test=train_test_split(input_images,input_lables,test_size=0.2,random_state=0)
x_train = np.reshape(x_train,[-1,32,32,1])
x_test = np.reshape(x_train,[-1,32,32,1])
fv_hibert_CNN_Model = fv_hibert_CNN(x_train[1].shape)

# checkpoint
#filepath="D:/Graph/checkpoints/weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

#试过adam>RMSProp约等于Nadam>Adaprop>adadelta约等于adamax
fv_hibert_CNN_Model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics= ['accuracy'])
#fv_hibert_CNN_Model.fit(x=x_image,y=input_lables,epochs=40,verbose=1,validation_split=0.2,batch_size=16)
fv_hibert_CNN_Model.summary()
start = time.time()
#fv_hibert_CNN_Model.fit(x=x_image,y=input_lables,epochs=40,verbose=1,validation_split=0.2,batch_size=16)
#在callbacks中加入EarlyStopping(monitor='val_loss', patience=2, verbose=0),
history = fv_hibert_CNN_Model.fit(x=x_train,y=y_train,epochs=40,verbose=1,callbacks=[ModelCheckpoint('D:/Graph/checkpoints/weights.best-{epoch}.hdf5', monitor='acc', verbose=0, save_best_only=True, mode='max')],validation_split=0.2,validation_data=(x_test,y_test),batch_size=64,shuffle=True)
fv_hibert_CNN_Model.save('fv_hibert_CNN_Model_Kreas_1.h5')
print('@ Total Time Spent: %.2f seconds' % (time.time() - start))

loss, accuracy = fv_hibert_CNN_Model.evaluate(x_image, input_lables, verbose=0)
print("Training Accuracy = %.2f %%     loss = %f" % (accuracy * 100, loss))
accuracy_curve(history)

#保存模型
layers_to_save = ['fc0','fc1']#,'pool0','conv1','pool1','fc0','fc1'试过不能议案家太多 或者也可以set(layer.name for layer in fv_hibert_CNN_Model.layers  if layer.name.startswith('dense_'))
embedding_layer_names = set(layer.name for layer in fv_hibert_CNN_Model.layers  if layer.name in layers_to_save）
tb_cb = TensorBoard(log_path, histogram_freq=10, batch_size=32,
                           write_graph=True, write_grads=True, write_images=True,
                           embeddings_freq=10, embeddings_metadata=None,
                           embeddings_layer_names=embedding_layer_names)

tb_cb.set_model(fv_hibert_CNN_Model)

