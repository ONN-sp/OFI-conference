# 利用梯度带(tf.GradientTape), 但在笔记本上没运行出来(内存不够)
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import numpy as np

with tf.device('/gpu:0'):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # (60000, 28, 28)
    print(type(x_train))
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    print(x_train.shape)
    # 利用tf.data来将数据集切分为batch和混淆数据集
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).shuffle(10000).batch(32)  # shuffle对准确率几乎无影响

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(input_shape = (28,28,1), filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')  # 32是卷积核个数也就是输出通道数, 3是卷积核大小
            self.pool1 = MaxPooling2D(pool_size = 2, strides = 2)
            self.flatten = Flatten()
            self.d1 = Dense(128, activation = 'relu')
            self.d2 = Dense(10)
        def call(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)
    model = MyModel()
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)  # from_logits = True则损失函数里面包含了softmax, 那么不用在网络里面加了
    # @tf.function
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:  # 使用tf.GradientTape()训练模型
    #         preds = model(images, training = True)
    #         loss = loss_func(labels, preds)
    #     gradients = tape.gradient(loss, model.trainable_variables) #标量梯度
    #     adam.apply_gradients(zip(gradients, model.trainable_variables))
    #     train_loss(loss)
    #     train_accuracy(labels, preds)

    # @tf.function
    # def test_step(images, labels):
    #   # training=False is only needed if there are layers with different
    #   # behavior during training versus inference (e.g. Dropout).
    #   predictions = model(images, training=False)
    #   t_loss = loss_func(labels, predictions)
    #   test_loss(t_loss)
    #   test_accuracy(labels, predictions)

    model.compile(optimizer = adam,
                loss = loss_func,
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs = 5, batch_size = 64)  # 使用model.fit()训练模型
    # model.summary()
    loss, accuracy = model.evaluate(x_test, y_test)

    # EPOCHS = 5

    # for epoch in range(EPOCHS):
    #   # Reset the metrics at the start of the next epoch
    #   train_loss.reset_states()
    #   train_accuracy.reset_states()
    #   test_loss.reset_states()
    #   test_accuracy.reset_states()

    #   for images, labels in train_ds:
    #     train_step(images, labels)

    #   for test_images, test_labels in test_ds:
    #     test_step(test_images, test_labels)

    #   print(
    #     f'Epoch {epoch + 1}, '
    #     f'Loss: {train_loss.result()}, '
    #     f'Accuracy: {train_accuracy.result() * 100}, '
    #     f'Test Loss: {test_loss.result()}, '
    #     f'Test Accuracy: {test_accuracy.result() * 100}'
    #   )