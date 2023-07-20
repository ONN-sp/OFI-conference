# 保存模型, 保存参数
import os
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # type = <class 'numpy.ndarray'>
x_train, x_test = x_train.reshape(-1, 28*28) / 255.0, x_test.reshape(-1, 28*28) / 255.0
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(32)
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # input_shape = (None, 28*28)或者input_shape = (28*28, ), 而不能input_shape = (28*28)
        self.d1 = Dense(input_shape = (28*28, ), units = 512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001))
        self.drop1 = Dropout(0.2)
        self.d2 = Dense(10)
    def call(self, x):
        x = self.d1(x)
        x = self.drop1(x)
        return self.d2(x)
model = MyModel()
model.build(input_shape = (None, 28*28))  # 在这里的input_shape不能写成(28*28, ), 必须写成添加了batch_size之后的shape形式
model.summary()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
test_loss = tf.keras.metrics.Mean()
test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc(labels, predictions)
@tf.function
def test_step(images, labels):
    predictions = model(images, training = False)
    loss = loss_func(labels, predictions)
    test_loss(loss)
    test_acc(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()
    for images, labels in train_ds:
        train_step(images, labels)
    for images, labels in test_ds:
        test_step(images, labels)
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_acc.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_acc.result() * 100}'
    )
