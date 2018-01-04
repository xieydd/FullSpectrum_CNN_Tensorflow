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
import scipy
from datetime import timedelta
from sklearn.model_selection import train_test_split
from fv_hibert_CNN_Model import FV_Hilbert_CNNConfig,FV_Hilbert_CNN
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from mpl_toolkits.mplot3d import Axes3D

filenames = 'J:/1/啦啦啦/sample'
save_dir = 'D:\Graph'
save_path = os.path.join(save_dir,'best_validation')#最佳验证 结果保存地址

######################################  
# TODO: 获取全部数据input和labels filename暂定J:/1/啦啦啦/sample  
#####################################
def load_data(filename):
    # 第一次遍历文件目录是为了获取文件总数  
    input_count = 0  
    for i in range(0,9):  
        dir = '%s/%s/' % (filenames,i)               # 这里可以改成你自己的图片目录，i为分类标签  
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
        dir = '%s/%s/' % (filenames,i)      
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
    #有玄学，必须使用\不然报错PermissionDeniedError: Failed to create a directory，而且C盘好像还不行，服气
    tensorboard_dir = 'D:\Graph'
    t_sne = None
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
   
    tf.summary.scalar("loss",model.loss)
    tf.summary.scalar("accuracy",model.acc)
    tf.summary.scalar("learning_rate",model.learning_rate)
    tf.summary.histogram("logit",model.logit)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    
    #配置Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Loading training and validation data...')      
    #载入训练集和验证集
    start_time = time.time()
    time_dif = get_time(start_time)
    print('The Usage :',time_dif)
   
    #创建Session
    session= tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)
    print("Training and evaluating.......")   
    start_time = time.time()
    total_batch = 0               #总批次
    best_acc_val  = 0.0           #最佳验证集概率
    last_improved = 0          #记录上一次提升批次
    required_improved = 1000   #如果超过1000lunix不提升就提前结束
   
    
    flag = False
    for epoch in range(config.num_epochs):
        x_train,x_test,y_train,y_test=train_test_split(signal,labels,test_size=0.2,random_state=0)
        x_test = np.reshape(x_test,[-1,32,32,1])
        print("Epoch:", epoch+1 )
        train_batch = batch_iter(x_train,y_train,config.batch_size)
        for x_batch,y_batch in train_batch:
            x_batch = np.reshape(x_batch,[-1,32,32,1])
            y_batch = np.reshape(y_batch,[-1,9])
            feed_dict = feed_data(x_batch,y_batch,config.drop_out)
            #print(session.run(model.logit,feed_dict=feed_dict))
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary,feed_dict=feed_dict)
                writer.add_summary(s,total_batch)
            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 0.5
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
            
            
            x_train_reshaped = np.reshape(x_train,[-1,32,32,1])
            feed_tsne = feed_data(x_train_reshaped,y_train,config.drop_out)
            t_sne,cValus = session.run([model.logit_1,model.input_y],feed_dict=feed_tsne)
            
            if total_batch == 3000:
                #scipy.io.savemat('J:/1/啦啦啦/t-sne/arg_label.mat',{'arg_label':y_train})
                #scipy.io.savemat('J:/1/啦啦啦/t-sne/arg_logit.mat',{'arg_logit':t_sne})
                #scipy.io.savemat('J:/1/啦啦啦/t-sne/y_pred_cls.mat',{'y_pred_cls':t_sne})
                #x_train_reshaped = np.reshape(x_train,[-1,32,32,1])
                #feed_tsne = feed_data(x_train_reshaped,y_train,config.drop_out)
                '''
                t_sne = np.zeros((t_sne_1.shape))
                for i in range(t_sne_1.shape[0]):
                   a = t_sne_1[i,:].max()
                   for j in range(t_sne_1.shape[1]):
                      if (t_sne_1[i][j] == a):
                         t_sne[i][j] = 1.0
                      else:
                         t_sne[i][j] = 0.0
                print(t_sne)
                '''
                X_tsne = TSNE(n_components = 3,learning_rate=50).fit_transform(t_sne)
                X_pca = PCA(n_components=3).fit_transform(t_sne)
                
                '''
                import seaborn as sns
                palette = np.array(sns.color_palette("hls", 10))
                c=palette[colors.astype(np.int)]
                
                
                X_tsne_min, X_tsne_max = np.min(X_tsne,axis=0), np.max(X_tsne,axis=0)
                X_tsne = (X_tsne - X_tsne_min) / (X_tsne_max - X_tsne_min)
                
                
                X_pca_min, X_pca_max = np.min(X_pca,axis=0), np.max(X_pca,axis=0)
                X_pca = (X_pca - X_pca_min) / (X_pca_max - X_pca_min)
                '''
                
                cValue = ['#7e1e9c','#15b01a','#0343df','#ff81c0','#653700','#e50000','#95d0fc','#029386','#f97306']
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121, projection='3d')
                print(cValus[:,0])
                ax1.scatter(X_tsne[:, 0], X_tsne[:, 1],X_tsne[:, 2],c=cValus[:,0])
                ax1.set_title('T-SNE')
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(X_pca[:, 0], X_pca[:, 1],X_pca[:, 2],c=cValus[:,0])
                ax2.set_title('PCA')
                
                plt.show()
                plt.savefig("examples.jpg") 
                
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
    x_test = np.reshape(x_test,[-1,32,32,1])
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
        
       #评价指标
        print("Precision,Recall and F1-Score...")
        categories = ['0','1','2','3','4','5','6','7','8']
        print(metrics.classification_report(y_test_cls,y_pred_cls,target_names=categories))

        #混淆函数
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls,y_pred_cls)
        print(cm)

        time_dif = get_time(start_time)
        print("Time Usage:",time_dif)
     




'''
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
'''

def main(argv=None):
    signal,labels = load_data(filenames)
    train(signal,labels)
    #test(signal,labels)
    

config = FV_Hilbert_CNNConfig()
model = FV_Hilbert_CNN(config)
main()