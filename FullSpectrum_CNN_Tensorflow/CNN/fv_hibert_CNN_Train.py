# -*- coding:utf-8 -*-
#@Description: 运行全矢希尔伯特的CNN模型
#@author xieydd xieydd@gmail.com
#@date 2017-12-07 下午20:50:50
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from fv_hibert_CNN_Model import FV_Hilbert_CNNConfig,FV_Hilbert_CNN
from sklearn import metrics


filename = 'J:/1/啦啦啦/sample'
save_dir = 'D:/Graph'
save_path = os.path.join(save_dir,'best_validation')#最佳验证 结果保存地址

######################################  
# TODO: 获取全部数据input和labels filename暂定J:/1/啦啦啦/sample  
#####################################
def load_data(filename):
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
# TODO: 获取程序运行时间  
#####################################
def get_time(start_time):
   end_time = time.time()
   return timedelta(seconds=int(round(end_time-start_time)))


def feed_data(x_batch,y_batch,keep_prob):
   feed_dict = {
         model.input_x:x_batch,
         model.input_y:y_batch,
         model.keep_prob:keep_prob
         }
   return feed_dict


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def evaluate(sess,x_,y_):
   """评估在某一数据上报的准确率和损失"""
   data_len = len(x_)
   batch_eval = batch_iter(x_,y_,64)
   total_loss = 0.0
   total_acc = 0.0
   for x_batch,y_batch in batch_eval:
      batch_len = len(x_batch)
      feed_dict = feed_data(x_batch,y_batch,1.0)
      loss,acc = sess.run([model.loss,model.acc],feed_dict=feed_dict)
      total_loss += loss *  batch_len
      total_acc += acc * batch_len
      
      return total_loss / data_len,total_acc / data_len




def train(signal,labels):
   print('Configuting TensorBoard and Saver...')
   # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
   tensorboard_dir = 'tensorboard/fv_hibert_CNN'
   if not os.path.exists(tensorboard_dir):
      os.makedirs(tensorboard_dir)
   
   tf.summary.scalar("loss",model.loss)
   tf.summary.scalar("accuracy",model.acc)
   
   merged_summary = tf.summary.merge_all()
   writer = tf.summary.FileWriter(tensorboard_dir)
   
   #配置Saver
   saver = tf.train.Saver()
   if not os.path.exists(save_dir):
      os.makedirs(save_dir)
   print('Loading training and validation data...')      
   #载入训练集和验证集
   start_time = time.time()
   x_train,x_test,y_train,y_test=train_test_split(signal,labels,test_size=0.2,random_state=0)
   x_train = np.reshape(x_train,[-1,32,32,1])
   x_test = np.reshape(x_train,[-1,32,32,1])
   time_dif = get_time(start_time)
   print('The Usage :',time_dif)
   
   #创建Session
   session= tf.Session()
   session.run(tf.global_variables_initializer())
   writer.add_graph(session.graph)
   print("Training and evaluaating.......")   
   start_time = time.time()
   total_batch = 0               #总批次
   best_acc_val  = 0.0           #最佳验证集概率
   last_improved = 0          #记录上一次提升批次
   required_improved = 1000   #如果超过1000lunix不提升就提前结束
   
   flag = False
   for epoch in range(config.num_epochs):
      print("Epoch:", epoch+1 )
      train_batch = batch_iter(x_train,y_train,config.batch_size)
      for x_batch,y_batch in train_batch:
         feed_dict = feed_data(x_batch,y_batch,config.drop_out)
         
         if total_batch % config.save_per_batch == 0:
            # 每多少轮次将训练结果写入tensorboard scalar
            s = session.run(merged_summary,feed_dict=feed_dict)
            writer.add_summary(s,total_batch)
            
         if total_batch % config.print_per_batch == 0:
            # 每多少轮次输出在训练集和验证集上的性能
            feed_dict[model.keep_prob] = 1.0
            loss_train,acc_train = session.run([model.loss,model.acc],feed_dict=feed_dict)
            loss_val,acc_val = evaluate(session,x_test,y_test)
            
            if acc_val > best_acc_val:
               best_acc_val = acc_val
               last_improved = total_batch
               saver.save(sess=session,save_path=save_path)
               improved_str = '*'
            else:
               improved_str = ''
            time_dif = get_time(start_time)
            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                    + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
            print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
         session.run(model.train_step,feed_dict = feed_dict)
         total_batch += 1
         
         if total_batch - last_improved > required_improved:
            print("No Optimization for a long time,auto-stopping.....")
            flag = True
            break
      if flag:
         break
      
      
def test(signal,labels):
     print("Loading test data...")
     x_train,x_test,y_train,y_test=train_test_split(signal,labels,test_size=0.2,random_state=0)
     x_train = np.reshape(x_train,[-1,32,32,1])
     x_test = np.reshape(x_train,[-1,32,32,1])
     start_time = time.time()
     session = tf.Session()
     session.run(tf.global_variables_initializer())
     saver = tf.train.Saver()
     saver.restore(sess= session,save_path=save_path)#
     
     print('Testing')
     loss_test,acc_test = evaluate(session,x_test,y_test)
     msg = 'Test Loss: {0:>6.2},Test Acc:{1:>7.2%}'
     print(msg.format(loss_test,acc_test))
     
     batch_size = 64
     data_len = len(x_test)
     num_batch = int((data_len - 1)/batch_size) + 1
     
     y_test_cls = np.argmax(y_test,1)
     y_pred_cls = np.zeros(shape=len(x_test),dtype=np.int32)
     for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size,data_len)
        feed_dict = {
              model.input_x:x_test[start_id:end_id],
              model.keep_prob:1.0
              }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls,feed_dict=feed_dict)
        
       #评
       print("Precision,Recall and F1-Score...")
       print(metrics.classification_report(y_test_cls,y_pred_cls,target_names=categories))
       
       #混
       print("Confusion Matrix...")
       cm = metrics.confusion_matrix(y_test_cls,y_pred_cls)
       print(cm)
       
       time_dif = get_time(start_time)
       print("Time Usage:",time_dif)
     





def main(argv=None):
   signal,labels = load_data(filename)
   train(signal,labels)

if __name__ == 'main':
   if len(sys.argv) != 2 or sys.argv[1] not in ['train','test']:
      raise ValueError("""usage: python fv_hibert_CNN_Train.py [train/test]""")
      
   print('Configguring CNN Model')
   config = FV_Hilbert_CNNConfig()
   model = FV_Hilbert_CNN(config)
   if sys.argv[1] == 'train':
      train()
   else:
      test()
