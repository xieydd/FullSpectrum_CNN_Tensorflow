# -*- coding:utf-8 -*-
#@Description: 将Keras的Model转换成Tensorflow的Graph、用Tensorflow的Graph做模型预测、计算雅克比行列式
#@author xieydd xieydd@gmail.com
#@date 2017-12-06 上午10:50:51

from keras.models import load_model
import tensorflow as tf
import numpy as np


# Create function to convert saved keras model to tensorflow graph
def convert_to_pb(weight_file,input_fld='',output_fld=''):

    import os
    import os.path as osp
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    from keras.models import load_model
    from keras import backend as K


    # weight_file is a .h5 keras model file
    output_node_names_of_input_network = ["pred0"] 
    output_node_names_of_final_network = 'output_node'

    # change filename to a .pb tensorflow file
    output_graph_name = weight_file[:-2]+'pb'
    weight_file_path = osp.join(input_fld, weight_file)

    net_model = load_model(weight_file_path)

    num_output = len(output_node_names_of_input_network)
    pred = [None]*num_output
    pred_node_names = [None]*num_output

    for i in range(num_output):
        pred_node_names[i] = output_node_names_of_final_network+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

    return output_fld+output_graph_name

#Create function to load the tf model as a graph
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )

    input_name = graph.get_operations()[0].name+':0'
    output_name = graph.get_operations()[-1].name+':0'

    return graph, input_name, output_name

#Create a function to make model predictions using the tf graph
def predict(model_path, input_data):
    # load tf graph
    tf_model,tf_input,tf_output = load_graph(model_path)

    # Create tensors for model input and output
    x = tf_model.get_tensor_by_name(tf_input)
    y = tf_model.get_tensor_by_name(tf_output) 

    # Number of model outputs
    num_outputs = y.shape.as_list()[0]
    predictions = np.zeros((input_data.shape[0],num_outputs))
    for i in range(input_data.shape[0]):        
        with tf.Session(graph=tf_model) as sess:
            y_out = sess.run(y, feed_dict={x: input_data[i:i+1]})
            predictions[i] = y_out

    return predictions

#计算雅克比行列式
def compute_jacobian(model_path,input_data):

    tf_model,tf_input,tf_output = load_graph(model_path)    

    x = tf_model.get_tensor_by_name(tf_input)
    y = tf_model.get_tensor_by_name(tf_output)
    y_list = tf.unstack(y)
    num_outputs = y.shape.as_list()[0]
    jacobian = np.zeros((num_outputs,input_data.shape[0],input_data.shape[1]))
    for i in range(input_data.shape[0]):
        with tf.Session(graph=tf_model) as sess:
            y_out = sess.run([tf.gradients(y_, x)[0] for y_ in y_list], feed_dict={x: input_data[i:i+1]})
            jac_temp = np.asarray(y_out)
        jacobian[:,i:i+1,:]=jac_temp[:,:,:,0]
    return jacobian



#xxx.h5
model_filename = ''
tf_model_path = convert_to_pb('model_file.h5','/model_dir/','/model_dir/')
tf_predictions = predict(tf_model_path,test_data)
jacobians = compute_jacobian(tf_model_path,test_data)