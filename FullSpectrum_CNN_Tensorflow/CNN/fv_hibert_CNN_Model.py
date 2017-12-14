# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:08:09 2017

@author: xieydd
"""

import os  
import numpy as np  
import tensorflow as tf   
import pandas as pd  
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
import keras.backend.tensorflow_backend as KTF

class FV_Hilbert_CNNConfig(object):
    """CNN配置参数"""
    kernel_size1 = 3    #卷积层1核尺寸
    kernel_size2 = 5    #卷积层2核尺寸
    channels1 = 1       #卷积1层数
    channels2 = 6       #卷积2层数
    kernel_num1 = 6    #卷积核个数
    kernel_num2 = 12   #卷积核个数
    num_filter1 = 6     #卷积层1核数目
    num_filter2 = 12    #卷积层2核数目
    num_pool1 = 2       #池化层
    num_pool2 = 2       #池化层
    num_fc1 = 100       #全连接层1神经元
    num_fc2 = 9         #全连接层2神经元

    signal_length = 768    #信号长度
    signal_classes = 9          #信号种类
    signal_reshaped = 32         #输

    batch_size = 64         #每批训练大小
    num_epochs = 4000         #总迭代轮数
    learning_rate = 1e-4    #学习率
    drop_out = 0.5          #drop参数

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
    train = True             # 是否使用DropOut
    Gpu_used = 0.8           # Gpu使用量
    moving_average_decay = 0.99 #滑动平均衰减率
    learning_rate_base = 0.0001 #学习率
    learning_rate_decay = 0.95  #学习率衰减率
    
    
class FV_Hilbert_CNN(object):
    '''全矢希尔伯特模型'''
    def __init__(self,config):
        self.config = config

        #三个待输入参数
        self.input_x = tf.placeholder(tf.float32,shape=[None,self.config.signal_reshaped,self.config.signal_reshaped,1],name='input_x')
        self.input_y = tf.placeholder(tf.float32,shape=[None,self.config.signal_classes],name='input_y')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.cnn()
        
    def cnn(self):
        KTF.set_session(FV_Hilbert_CNN.get_session(0.8))  # using 80% of total GPU Memory  
        #TODOwith tf.device('/cpu:0'):
        
        
        regularizer = tf.contrib.layers.l2_regularizer(self.config.learning_rate_base)    
        
        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable("weights",[self.config.kernel_size1,self.config.kernel_size1,self.config.channels1,self.config.kernel_num1],initializer = tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("biases",[self.config.kernel_num1],initializer=tf.constant_initializer(0.0))
            #TODO
            conv1 = tf.nn.conv2d(self.input_x,conv1_weights,strides=[1,1,1,1],padding="SAME")
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        with tf.variable_scope('layer2-pool1'):
            pool1 = tf.nn.max_pool(relu1,ksize=[1,self.config.num_pool1,self.config.num_pool1,1],strides=[1,2,2,1],padding='SAME')
            
        with tf.variable_scope('layer3-conv2'):
            conv2_weights = tf.get_variable("weights",[self.config.kernel_size2,self.config.kernel_size2,self.config.channels2,self.config.kernel_num2],initializer = tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("biases",[self.config.kernel_num2],initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding="SAME")
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

        with tf.variable_scope("layer4-pool2"):
            self.pool2 = tf.nn.max_pool(relu2,ksize=[1,self.config.num_pool2,self.config.num_pool2,1],strides=[1,2,2,1],padding="SAME")
            
            h_pool2_flat=tf.reshape(self.pool2,[-1,self.config.signal_length])
        
        with tf.variable_scope("layer5-fc1"):
            fc1_weights = tf.get_variable("weights",[self.config.signal_length,self.config.num_fc1],initializer = tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection("losses",regularizer(fc1_weights))
            fc1_biases = tf.get_variable("biases",[self.config.num_fc1],initializer=tf.constant_initializer(0.1))
            fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat,fc1_weights)+fc1_biases)
            if self.config.train: fc1 = tf.nn.dropout(fc1,self.config.drop_out)
            
        with tf.variable_scope("layer6-fc2"):
            fc2_weights = tf.get_variable("weights",[self.config.num_fc1,self.config.num_fc2],initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection("losses",regularizer(fc2_weights))
            fc2_biases = tf.get_variable("biases",[self.config.num_fc2],initializer=tf.constant_initializer(0.1))
            self.logit = tf.matmul(fc1,fc2_weights)+fc2_biases
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logit),1)
        
        
        #使用滑动平均输出
        global_step =  tf.Variable(0.0,trainable=True)
        with tf.name_scope('moving_average'):
            variable_average = tf.train.ExponentialMovingAverage(self.config.moving_average_decay,global_step)
            variable_average_op = variable_average.apply(tf.trainable_variables())
            
        with tf.name_scope('loss_function'):
            cross_entrypy_mean  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logit))
            self.loss = cross_entrypy_mean + tf.add_n(tf.get_collection('losses'))
             
            
        with tf.name_scope('train_step_optimize'):
           #TODO这里还有一个参数没加
            self.learning_rate = tf.train.exponential_decay(self.config.learning_rate_base,global_step,1800/self.config.batch_size,self.config.learning_rate_decay,staircase=True)
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            with tf.control_dependencies([self.train_step,variable_average_op]):
                self.config.train_op = tf.no_op(name='train')
                
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    
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