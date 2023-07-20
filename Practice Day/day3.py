# 利用tf.keras.layers搭建VGG16做cifar10分类实验(笔记本跑不动)

import tensorflow as tf
from tensorflow.keras import datasets, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()   # train_images = (50000, 32, 32, 3)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.reshape(-1, 28, 28, 1) / 255.0, test_images.reshape(-1, 28, 28, 1) / 255.0

class MyCNN(Model):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')  # filter数就是该卷积层的输出通道数
        self.conv2 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.pool1 = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')
        self.conv3 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.conv4 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.pool2 = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')
        self.conv5 = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.conv6 = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.conv7 = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.pool3 = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')
        self.conv8 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.conv9 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.conv10 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.pool4 = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')
        self.conv11 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.conv12 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.conv13 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        self.pool5 = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')       
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation = 'relu')
        self.d2 = Dense(1024, activation = 'relu')
        self.d3 = Dense(10)
    def call(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
model = MyCNN()
model()
print(model)
# 前十组为0.001 后面指数减少
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(loss, accuracy)

adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)  # from_logits = True则损失函数里面包含了softmax, 那么不用在网络里面加了
model.compile(optimizer = adam,
              loss = loss_func,
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 5, batch_size = 32, callbacks=[PrintDot()])  # 使用model.fit()训练模型

        