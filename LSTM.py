# 利用keras库快速搭建LSTM实现回归实验
from tkinter import SE
from keras.layers import LSTM
from keras.models import Model
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from keras import utils as np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import TimeDistributed, Activation, Dense, Lambda
from keras.optimizers import Adam

batch_start = 0
batch_size = 32
time_steps = 200
input_size = 1
output_size = 1
n_hidden_units = 128
cell_size = 20  # 其实就是RNN1中的n_hidden_units
lr = 0.006      # 学习率
    
model = Sequential()
# 因为model.train_on_batch输入是3D, 而Dense输入要变成2D, 所以先加一层自定义层完成reshape
def ff(x):
    x = tf.reshape(x, [-1, input_size])
    return x
model.add(Lambda(ff, output_shape = (-1, input_size), mask = None, arguments = None))

model.add(Dense(
    n_hidden_units,
    batch_input_shape = (batch_size*time_steps, input_size),   # input_shape的话这里就只能写两个维度，因为batch_size是自动补充的
))  # shape = (50*20, 128)
model.add(Activation('relu'))
def f(x):
    x = tf.reshape(x, [-1, time_steps, n_hidden_units])
    return x
model.add(Lambda(f, output_shape = (-1, time_steps, n_hidden_units), mask = None, arguments = None))

model.add(LSTM(
    cell_size,  
    return_sequences = True,    # True: output at all steps. False: output as last step, 这里是True因为要输出所有的step, 而分类的时候要是False
    stateful = True,    # True: the final state of batch1 is feed into the initial state of batch2. stateful与batch_input_size是一起用的
))

model.add(TimeDistributed(Dense(output_size)))
adam = Adam(lr)
model.compile(
    optimizer = adam, 
    loss = 'mse', )

print('Train')


