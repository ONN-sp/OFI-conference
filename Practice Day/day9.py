import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
x_test= x_test.reshape(-1, 28, 28, 1).astype("float32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")
class CNN_Model(Model):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.c1 = Conv2D(filters = 32, kernel_size=3, strides=1, padding='same')
        self.m1 = MaxPooling2D(pool_size = 2, strides = 2)
        self.f1 = Flatten()
        self.d1 = Dense(32, activation = 'relu')
        self.d2 = Dense(10, activation = 'softmax')
    def call(self, x):
        x = self.c1(x)
        x = self.m1(x)
        x = self.f1(x)
        x = self.d1(x)
        return self.d2(x)
model = CNN_Model()
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = 32, epochs = 5, validation_split=0.1)
model.summary()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['trainning', 'valivation'])
plt.show()