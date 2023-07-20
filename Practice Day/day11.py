#卷积自编码器
import tensorflow as tf
from tensorflow.keras.layers import Dense, UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)/255.0
x_test = x_test.reshape(-1, 28, 28, 1)/255.0
class ConvEnDe(Model):
    def __init__(self):
        super(ConvEnDe, self).__init__()
        self.c1 = Conv2D(16, 3, activation='relu', padding='same')
        self.m1 = MaxPooling2D(2, padding='same')
        self.c2 = Conv2D(16, 3, activation='relu', padding='same')
        self.u = UpSampling2D(2)
        self.c3 = Conv2D(1, 3, activation='sigmoid', padding='same')
    def call(self, x):
        x = self.c1(x)
        x = self.m1(x)
        x = self.c2(x)
        x = self.u(x)
        return self.c3(x)
model = ConvEnDe()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy())
early_stop = tf.keras.callbacks.EarlyStopping(patience=2, monitor='loss')
model.fit(x_train, x_train, batch_size=64, epochs=2, validation_split=0.1, callbacks=[early_stop])
predicting = model.predict(x_test)
#可视化
n = 5
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(tf.reshape(x_test[i+1], (28,28)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n+i+1)
    plt.imshow(tf.reshape(predicting[i+1], (28,28)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()