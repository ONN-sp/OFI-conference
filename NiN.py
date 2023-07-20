import tensorflow as tf
import numpy as np

with tf.device('/gpu:0'):
    class DataLoader(): # 利用resize对数据进行处理
        def __init__(self):
            fashaion_mnist = tf.keras.datasets.fashion_mnist
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashaion_mnist.load_data()
            self.train_images, self.test_images = self.train_images.reshape(-1, 28, 28, 1)/255, self.test_images.reshape(-1, 28, 28, 1)/255
        def get_train_batch(self, batch_size):
            index = np.random.randint(0, np.shape(self.train_images)[0], batch_size) # 随机在[0,数据总数]之间随机生成batch_size个整数,很可能有相同数,
            self.new_train_images = tf.image.resize_with_pad(self.train_images[index], 224, 224, 'nearest', antialias=True)
            return self.new_train_images.numpy(), self.train_labels[index]
        def get_test_batch(self, batch_size):
            index = np.random.randint(0, np.shape(self.test_images)[0], batch_size) # 随机在[0,数据总数]之间随机生成batch_size个整数,很可能有相同数,
            self.new_test_images = tf.image.resize_with_pad(self.test_images[index], 224, 224, 'nearest', antialias=True)
            return self.new_test_images.numpy(), self.test_labels[index]

    # 定义NiN块,下面是直接用函数定义的block,其实也可以像自定义层那样用class的方法来自定义block(自定义的block可以就理解为一个自定义层)
    def NiN_block(filters, kernel_size, strides, padding):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, activation = 'relu'),
            tf.keras.layers.Conv2D(filters, kernel_size = 1, activation = 'relu'),
            tf.keras.layers.Conv2D(filters, kernel_size = 1, activation = 'relu')
        ])

    def NiN_network():
        return tf.keras.models.Sequential([
            NiN_block(96, kernel_size=11, strides=4, padding='valid'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            NiN_block(256, kernel_size=5, strides=1, padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            NiN_block(384, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            NiN_block(10, kernel_size=3, strides=1, padding='same'), # 分为10类,所以对于NiN来说最后一个卷积层的输出通道=10,因为没有Dense层,所以与VGG、AlexNet这些不同
            tf.keras.layers.GlobalAveragePooling2D(), # output_shape = (1, 10), 第一个1是batch_size
            tf.keras.layers.Reshape((1, 1, 10)), # output_shape = (1, 1, 1, 10)
            tf.keras.layers.Flatten(), # output_shape = (1, 10)
        ])

    # X = tf.random.uniform((1, 224, 224, 1))
    # for layer in NiN_network().layers:
    #     X = layer(X)
    #     print(layer.__class__.__name__,'output shape:\t', X.shape)
    model = NiN_network()
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = 'accuracy'
    )
    batch_size = 32
    epochs = 5
    for _ in range(epochs):
        for time in range(60000//batch_size): # fashaion_mnist是fashion_mnist数据集的总数
            data = DataLoader()
            train_batch_image, train_batch_label = data.get_train_batch(batch_size)
            test_batch_image, test_batch_label = data.get_test_batch(batch_size)
            # method 1
            # cost = model.train_on_batch(train_batch_image, train_batch_label)
            # if time%100 == 0:
            #     print(cost)
            # method 2(一个batch一个batch的fit)
            model.fit(train_batch_image, train_batch_label)
    loss, accuracy = model.evaluate(test_batch_image, test_batch_label)
    print(loss, accuracy)