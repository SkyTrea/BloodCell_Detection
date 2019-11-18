# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:53:53 2019

@author: pc
"""

from keras.preprocessing import image
from keras.utils import OrderedEnqueuer
import tensorflow as tf
import math
import numpy as np
from time import time,ctime
import os
from layers import (Variable_with_weight_loss,
                    Variable_with_weight_loss_xavier,
                    Variable_with_weight_loss_msra,
                    Conv,LRN,Maxpool,Flatten,Fc)

tf.reset_default_graph()

def Get_data(datapath,hight, width, batch_size):
    generator = image.ImageDataGenerator(
            rescale = 1./255,
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)
    dataset = generator.flow_from_directory(
        shuffle = True,
        batch_size = batch_size,
        target_size = (hight, width),
        directory = datapath)
    enqueuer = OrderedEnqueuer(dataset,use_multiprocessing=False,shuffle=True)
    enqueuer.start(workers=1, max_queue_size=10)
    output_generator = enqueuer.get()

    return output_generator
'''
Model
'''
def Inference(in_op,num_labels):
    conv1 = Conv(in_op,'conv1',3,1,80)
    conv2 = Conv(conv1,'conv2',3,1,64)
    pool1 = Maxpool(conv2,'pool1',2,2)

    conv3 = Conv(pool1,'conv3',3,1,64)
#    conv4 = Conv(conv3,'conv4',3,1,64)
    pool2 = Maxpool(conv3,'pool2',2,2)

#    conv5 = Conv(pool2,'conv5',3,1,32)
#    conv6 = Conv(conv5,'conv6',3,1,32)
#    pool3 = Maxpool(conv6,'pool3',2,2)
##
#    conv7 = Conv(pool3,'conv7',3,1,246)
#    conv8 = Conv(conv7,'conv8',3,1,256)
#    pool4 = Maxpool(conv8,'pool4',2,2)
#
#    conv9 = Conv(pool4,'conv9',3,1,128)
#    conv10= Conv(conv9,'conv10',3,1,128)
#    pool5 = Maxpool(conv10,'pool5',2,2)

    flat  = Flatten(pool2)
    fc1   = Fc(flat, 'fc1', 128, activation = 'relu')
#    drop1 = tf.nn.dropout(fc1,0.75)
#    fc2   = Fc(drop1,'fc2',32)
#    drop2 = tf.nn.dropout(fc2,0.5)
    logit = Fc(fc1, 'logit', num_labels)
    return logit
'''
Trainning
'''
def Train(datapath):
    log             = open('log.txt','w+')
    loss_value_file = open('loss.txt','w+')
    accuracy_file   = open('accuracy.txt','w+')

    log.write('Train\nStart: %s\n'%ctime())
    loss_value_file.write(' epochs      loss\n')
    accuracy_file.write(' epochs      accuracy\n')

    samples              = 9957
    hight                = 80
    width                = 80
    num_channels         = 3
    num_labels           = 4

    learning_base_rate   = 0.001
    learning_rate_decay  = 0.99
    epochs               = 30
    batch_size           = 100
    steps                = math.ceil(samples/batch_size)
    print('Train: %s'%ctime())

    x = tf.placeholder(tf.float32, [None ,hight, width, num_channels], name='x-input')
    y_= tf.placeholder(tf.int64,shape=[None,num_labels],name='y-input')

    y = Inference(x,num_labels)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1), logits=y)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    global_step = tf.Variable(0,trainable=False)

    lr = tf.train.exponential_decay(learning_base_rate,global_step,steps,learning_rate_decay)

    train_step = tf.train.AdamOptimizer(lr).minimize(loss,global_step=global_step)

    saver = tf.train.Saver()
    model_save_path = './model1/'
    model_name      = 'model.ckpt'

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    summary_save_path = './struct1'
    if not os.path.exists(summary_save_path):
        os.mkdir(summary_save_path)
    writer = tf.summary.FileWriter(summary_save_path,tf.get_default_graph())
    writer.close()
    '''  '''
    imagedata = Get_data(datapath,hight, width, batch_size)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        image_test = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        test_accur = tf.reduce_mean(tf.cast(image_test, tf.float32))

        for epoch in range(1,epochs+1):
            print('\n\nEpoch %2d/%2d\n===================='%(epoch,epochs))
            acc = 0
            los = 0
            t1  = time()
            for step in range(1,steps+1):
                t3 = time()
                xs, ys = next(imagedata)
                _,loss_value = sess.run([train_step,loss],feed_dict={x:xs, y_:ys})
                los += loss_value
                mloss= los/step
                accu = test_accur.eval({x:xs, y_:ys})
                acc += accu
                accu = acc/step
                t4 = time()
                print('\r%3d/%3d >> loss: %.4f   accu: %.4f   %.2f    '%(step,steps,mloss,accu,t4-t3),end='')
            t2 = time()
            log.write('Epoch %2d/%2d >> loss: %.4f   accu: %.4f   %.2f\n'%(epoch,epochs,mloss,accu,t2-t1))
            loss_value_file.write('%5d   %g\n'%(epoch,mloss))
            accuracy_file.write('%5d      %f\n'%(epoch,accu))
            loss_value_file.flush()
            accuracy_file.flush()

            saver.save(sess,os.path.join(model_save_path,model_name), global_step = epoch)

        log.write('End: %s\n'%ctime())
        log.close()
        loss_value_file.close()
        accuracy_file.close()
'''
Running
'''
def main(argv=None):

    t1 = time()
    datapath = './cells/'
    Train(datapath)
    t2 = time()
    print('Time used: %.4f'%(t2-t1))
    print(ctime())

if __name__ == '__main__':
    tf.app.run()