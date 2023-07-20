import tensorflow as tf
from tensorflow.keras import datasets, Model
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow import keras

class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = Conv2D(filters = 96, kernel_size = 11, strides = 4, activation = 'relu')
        self.p1 = MaxPooling2D(pool_size = 3, strides = 2)
        self.c2 = Conv2D(filters = 256, kernel_size = 5, padding = 'same', activation = 'relu')
        self.p2 = MaxPooling2D(pool_size = 3, strides = 2)
        self.c3 = Conv2D(filters = 384, kernel_size = 3, padding = 'same', activation = 'relu')
        self.c4 = Conv2D(filters = 384, kernel_size = 3, padding = 'same', activation = 'relu')
        self.c5 = Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu')
        self.p3 = MaxPooling2D(pool_size = 3, strides = 2)
        self.f = Flatten()
        self.d1 = Dense(4096, activation = 'relu')
        self.dr1 = Dropout(0.5)
        self.d2 = Dense(4096, activation = 'relu')
        self.dr2 = Dropout(0.5)
        self.d3 = Dense(10)
    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.p3(x)
        x = self.f(x)
        x = self.d1(x)
        x = self.dr1(x)
        x = self.d2(x)
        x = self.dr2(x)
        return self.d3(x)

model = AlexNet()
adam = tf.keras.optimizers.Adam(learning_rate = 0.01)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)  # from_logits = True则损失函数里面包含了softmax, 那么不用在网络里面加了
model.compile(optimizer = adam,
              loss = loss_func,
              metrics=['accuracy'])