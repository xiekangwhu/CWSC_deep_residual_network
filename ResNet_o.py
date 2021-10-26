# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 18:42:01 2018

@author: shshen xiekang
"""

import keras
import argparse
import numpy as np
import random
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D,Lambda
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K


# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=32, metavar='NUMBER',
                 help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=150, metavar='NUMBER',
                help='epochs(default: 200)')
parser.add_argument('-d','--dataset', type=str, default="1111", metavar='STRING', help='dataset. (default: 1111)')
parser.add_argument('-n','--stack_n', type=int, default=5, metavar='NUMBER',
                help='stack number n, total layers = 6 * n + 2 (default: 5)')

args = parser.parse_args()

stack_n            = args.stack_n
layers             = 6 * stack_n + 2
num_classes        = 1
img_rows, img_cols = 10, 10
img_channels       = 13
batch_size         = args.batch_size
epochs             = args.epochs
iterations         = 37420 // batch_size + 1
weight_decay       = 0

def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f

def scheduler(epoch):
    if epoch < 30:
        return 0.01
    if epoch < 60:
        return 0.003
    if epoch < 90:
        return 0.001
    if epoch < 120:
        return 0.0003
    return 0.0001

def residual_network(img_input,classes_num=10,stack_n=5):
    
    def residual_block(x,o_filters,increase=False):
        stride = (1,1)
        if increase:
            stride = (2,2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 13x5x5 output: 13x5x32
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 13x5x32 output: 13x5x32
    for _ in range(stack_n):
        x = residual_block(x,16,False)

    # input: 13x5x32 output: 7x3x64
    x = residual_block(x,32,True)
    for _ in range(1,stack_n):
        x = residual_block(x,32,False)
    
   #  input: 7x3x64 output: 4x2x128
    x = residual_block(x,128,True)
    for _ in range(1,stack_n):
        x = residual_block(x,128,False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
   # x = Lambda(lambda y: K.sum(y, axis=1))(x)
    # input: 64 output: 10
    x = Dense(1,activation='sigmoid',kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    
    return x


if __name__ == '__main__':


    print("========================================") 
    print("MODEL: Residual Network ({:2d} layers)".format(6*stack_n+2)) 
    print("BATCH SIZE: {:3d}".format(batch_size)) 
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))


    print("== LOADING DATA... ==")
    # load data
    all_data= np.load('data/array_ncdata_Allmodel_mask.npy', allow_pickle=True)
    all_label= np.load('data/array_nclabel_Allmodel_mask.npy', allow_pickle=True)
   # random.shuffle(all_label)
    print(np.shape(all_data), np.shape(all_label))
    scaler = MinMaxScaler( )
    scaler.fit(all_label)
    scaler.data_max_
    all_label=scaler.transform(all_label)
    #all_data = np.transpose(all_data, [0,2,3,1])
    all_data2 = []
    for item in all_data:
        item = np.reshape(item, [-1,1])
        scaler.fit(item)
        scaler.data_max_
        item=scaler.transform(item)
        item = np.reshape(item, [10,10,13])
        all_data2.append(item)
    all_data = np.array(all_data2)

    x_train,x_test, y_train, y_test  = train_test_split(all_data, all_label, test_size=0.2, random_state=5)
    print(np.shape(x_train), np.shape(x_test), np.shape(y_train), np.shape(y_test))
  #  for i in range(5):
   #     x_train[:,:,:,i] = (x_train[:,:,:,i] - np.mean(x_train[:,:,:,i])) / np.std(x_train[:,:,:,i])
    #    x_test[:,:,:,i] = (x_test[:,:,:,i] - np.mean(x_test[:,:,:,i])) / np.std(x_test[:,:,:,i])

    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = residual_network(img_input,num_classes,stack_n)
    resnet    = Model(img_input, output)
    
    # print model architecture if you need.
    print(resnet.summary())


    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae',r2])

    # set callback
    cbks = [TensorBoard(log_dir='./resnet_{:d}_{}/'.format(layers,args.dataset), histogram_freq=0),
            LearningRateScheduler(scheduler)]
    
    # dump checkpoint if you need.(add it to cbks)
    # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)


    resnet.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=cbks,
              validation_data=(x_test, y_test),
              shuffle=True)


    print("== Construct DATA... ==")
    # load data
    all_construct_data= np.load('data/array_ncdata_Allmodel_all.npy', allow_pickle=True)
    print(np.shape(all_construct_data))
    #all_data = np.transpose(all_data, [0,2,3,1])
    all_construct_data2 = []
    for item in all_construct_data:
        item = np.reshape(item, [-1,1])
        scaler.fit(item)
        scaler.data_max_
        item=scaler.transform(item)
        item = np.reshape(item, [10,10,13])
        all_construct_data2.append(item)
    all_construct_data = np.array(all_construct_data2)

    all_construct_SC = resnet.predict(all_construct_data, batch_size=batch_size)
    np.save(file="all_construct_SC.npy", arr=all_construct_SC)
    # resnet.save('resnet_{:d}_{}.h5'.format(layers,args.dataset))
