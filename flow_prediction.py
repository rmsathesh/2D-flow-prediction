# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:44:12 2020

@author: hp
"""


import numpy as np
import os
import sys
import tqdm
#%%

import tensorflow as tf
import matplotlib.pyplot as plt
#%%
fpath = r'D:\study\ANN\CNN\projects\steady state flow'
fname = os.path.join(fpath, 'train.tfrecords')

#%%


def parse(serialized):
    
    features = {
        'boundary': tf.FixedLenFeature([], tf.string),
        'sflow': tf.FixedLenFeature([], tf.string)
        }
    parsed_value = tf.parse_single_example(serialized, features=features)
    
    raw_boundary = parsed_value['boundary']
    raw_velocity = parsed_value['sflow']
    boundary = tf.decode_raw(raw_boundary, tf.uint8)
    boundary = tf.reshape(boundary, [128, 256, 1])
    boundary = tf.cast(boundary, tf.float32)
    velocity = tf.decode_raw(raw_velocity, tf.float32)
    velocity = tf.reshape(velocity, [128, 256, 2])
    velocity = tf.cast(velocity, tf.float32)
    
    return boundary, velocity

#%%

def data_gen(filename, train=True, batch_size=8, buffer_size=1024):
    
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse)
    
    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    
    return iterator


#%%
def model(input_layer):
    a = []
    
    with tf.variable_scope('model'):
    
        value = tf.layers.conv2d(input_layer, filters=16, kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        value = tf.layers.conv2d(value, filters=16,  kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        a.append(value)
        value = tf.layers.max_pooling2d(value, pool_size=2, strides=2)
    
        
        value = tf.layers.conv2d(value, filters=32, kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        value = tf.layers.conv2d(value, filters=32,  kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)   
        a.append(value)
        value = tf.layers.max_pooling2d(value, pool_size=2, strides=2)
        
        value = tf.layers.conv2d(value, filters=64, kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        value = tf.layers.conv2d(value, filters=64,  kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        a.append(value)
        value = tf.layers.max_pooling2d(value, pool_size=2, strides=2) 
        
        value = tf.layers.conv2d(value, filters=128, kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        value = tf.layers.conv2d(value, filters=128,  kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        a.append(value)
        value = tf.layers.max_pooling2d(value, pool_size=2, strides=2)  
        
        value = tf.layers.conv2d(value, filters=256, kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        value = tf.layers.conv2d(value, filters=256,  kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        
        value = tf.layers.conv2d_transpose(value, filters=128, kernel_size=(2, 2), strides=2, padding='same')
        value = tf.concat([value, a[-1]], axis=3)
        value = tf.layers.conv2d(value, filters=128, kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        value = tf.layers.conv2d(value, filters=128,  kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        
        value = tf.layers.conv2d_transpose(value, filters=64, kernel_size=(2, 2), strides=2, padding='same')
        value = tf.concat([value, a[-2]], axis=3)
        value = tf.layers.conv2d(value, filters=64, kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        value = tf.layers.conv2d(value, filters=64,  kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        
        value = tf.layers.conv2d_transpose(value, filters=32, kernel_size=(2, 2), strides=2, padding='same')
        value = tf.concat([value, a[-3]], axis=3)
        value = tf.layers.conv2d(value, filters=32, kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        value = tf.layers.conv2d(value, filters=32,  kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
    
        value = tf.layers.conv2d_transpose(value, filters=16, kernel_size=(2, 2), strides=2, padding='same')
        value = tf.concat([value, a[-4]], axis=3)
        value = tf.layers.conv2d(value, filters=16, kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)
        value = tf.layers.conv2d(value, filters=16,  kernel_size=(3, 3), padding='same')
        value = tf.nn.relu(value)    
        
        out = tf.layers.conv2d(value, filters=2, kernel_size=(1, 1), strides=1, padding='same')
        out = tf.nn.tanh(out)
    
    return out

#%%
def loss(sflow, sflow_pred):
    
    with tf.variable_scope('loss'):
        total_loss = tf.nn.l2_loss(sflow, sflow_pred)
        tf.summary.scalar('loss', total_loss)
    return total_loss

def optimizer(total_loss, global_step, learning_rate=0.01):
    
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = optimizer.minimize(total_loss, global_step)
    return optimizer




#%%
def train(fname, batch_size=8, epochs=1):
    
    fpath = r'D:\study\ANN\CNN\projects\steady state flow\graph'
    mpath = r'D:\study\ANN\CNN\projects\steady state flow\model'
    
    
    with tf.variable_scope('train'):
    
        global_step = tf.Variable(initial_value=1, trainable=False)
        
        iterator = data_gen(fname, batch_size=batch_size)
        boundary, velocity = iterator.get_next() 
        tf.summary.image('boundary_input', boundary)
        tf.summary.image('velocity_x', velocity[:,:,:,0:1])
        tf.summary.image('velocity_y', velocity[:,:,:,1:2])
        velocity_pred = model(boundary)
        tf.summary.image('velocity_x_pred', velocity_pred[:,:,:,0:1])
        tf.summary.image('velocity_y_pred', velocity_pred[:,:,:,1:2])
        loss_val = loss(velocity, velocity_pred)
        opti = optimizer(loss_val, global_step, learning_rate=0.001)
        init = tf.global_variables_initializer()
        
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        
        epochs = 5
        stream_loss = 0
        log_iter=1
        
        with tf.Session() as sess:
            sess.run(init)
            
            writer = tf.summary.FileWriter(fpath, sess.graph)
            
            ckpt = tf.train.get_checkpoint_state(mpath)
            if ckpt and ckpt.model_checkpoint_path:
                print(f'Restoring model from {mpath}')
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            for i in range(1, epochs+1):  
                
                k=0
                sess.run(iterator.initializer)
                while True:
                    try:
                        _, total_loss = sess.run([opti, loss_val])
                        if k % 25 == 0:
                            summary = sess.run(summary_op)
                            writer.add_summary(summary, i*k)
                        stream_loss += total_loss
                        k += 1
                        print(f'completed{k} iterations with loss: {total_loss}')
                        
                    except:
                        print(f'finished full across')
                        saver.save(sess, save_path=mpath + '/model', global_step=i)
                        break
                        
                        
                if i%log_iter == 0:
                    print(f' Completed {i} epochs with loss: {stream_loss / (log_iter * k) :.2f}')
                    stream_loss = 0
        
        
#%%
tf.reset_default_graph()
train(fname, batch_size=16)

#%%
import h5py
import numpy as np
import glob 



#%%
test_files = glob.glob('D:\study\ANN\CNN\projects\steady state flow\computed_car_flow/*/')
#%%
def test_boundary(fname):
    file = h5py.File(fname, 'r')
    boundary = np.array(file['Gamma'][:]).reshape(128, 384, 1)
    boundary = boundary[:,0:256,:]
    file.close()
    return boundary

def test_velocity(fname):
    file = h5py.File(fname, 'r')
    velocity = np.array(file['Velocity_0'][:]).reshape(128, 384, 3)
    velocity = velocity[:,0:256,0:2]
    file.close()
    return velocity


#%%
mpath = r'D:\study\ANN\CNN\projects\steady state flow\model'
with tf.Graph().as_default():
    
    with tf.variable_scope('train'):
        bound = tf.placeholder(dtype=tf.float32, shape=[1, 128, 256, 1])
        vel_test_pred = model(bound)
        var_list = tf.trainable_variables()
        
        saver = tf.train.Saver(var_list)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mpath)
            if ckpt and ckpt.model_checkpoint_path:
                print('Restoring old saved model from {ckpt.model_checkpoint_path}')
                saver.restore(sess, ckpt.model_checkpoint_path)
            graph = tf.get_default_graph().as_graph_def()
            
            for f in test_files:
                test_name = f + 'fluid_flow_0002.h5'
                bound_test = test_boundary(test_name).reshape(1, 128, 256, 1)
                vel_test = test_velocity(test_name)
                
                vel_pred = sess.run(vel_test_pred, feed_dict={bound: bound_test})[0]
                
                vel_plt = np.concatenate([vel_test, vel_pred, vel_test - vel_pred], axis=1) 
                bound_con = np.concatenate(3*[bound_test], axis=2)
                vel_plt = np.sqrt(np.square(vel_plt[:,:,0]) + np.square(vel_plt[:,:,1])) - 0.05*bound_con[0,:,:,0]
                plt.imshow(vel_plt)
                plt.colorbar()
                plt.savefig(f + 'test_image.png')
                plt.show()
              

#%%




    
    


