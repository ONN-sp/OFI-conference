# 利用tf.keras.layers搭建VGG16做cifar10分类实验(笔记本跑不动)
import tensorflow as tf
from tensorflow.keras import datasets, Model, regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Hyparameters
EPOCHS = 5

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()   # train_images = (50000, 32, 32, 3)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.reshape(-1, 28, 28, 1) / 255.0, test_images.reshape(-1, 28, 28, 1) / 255.0
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(10000).batch(32)
print(train_images.shape)
# 下面用的是子类化和函数式API的方法混合创建网络
class MyCNN(Model):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = Conv2D(input_shape = (28, 28, 1), filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
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
# model.summary()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)  # from_logits = True则损失函数里面包含了softmax, 那么不用在网络里面加了
adam = tf.keras.optimizers.Adam(learning_rate = 0.0001)
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_acc')
test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_acc')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = model(images, training = True)
        loss = loss_func(labels, preds)
    gradients = tape.gradient(loss, model.trainable_variables)
    adam.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc(labels, preds)

@tf.function
def test_step(images, labels):
    preds = model(images, training = False)
    loss = loss_func(labels, preds)
    test_loss(loss)
    test_acc(labels, preds)

for epoch in range(EPOCHS):
    # reset metrics when netx_batch come
    train_acc.reset_states()
    train_loss.reset_states()
    test_acc.reset_states()
    test_loss.reset_states()
    for images, labels in train_ds:
        print(images.shape)
        train_step(images, labels)
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    print('Epoch is %d'%(epoch + 1),
          'train_loss = %f'%train_loss.result(),
          'train_acc = %f'%(train_acc.result()*100),
          'test_loss = %f'%test_loss.result(),
          'test_acc = %f'%(test_acc.result()*100))