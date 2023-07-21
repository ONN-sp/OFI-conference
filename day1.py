# 利用全连接层进行mnist图像分类
import tensorflow as tf
import numpy as np

with tf.device('/gpu:0'):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    x_train, x_test = x_train / 255.0, x_test / 255.0  # (60000, 28, 28)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))  # 第一层加input_shape,不能写(None, 28, 28)
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10))
    model.summary()
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)  # from_logits = True则损失函数里面包含了softmax, 那么不用在网络里面加了
    model.compile(optimizer = adam,
                loss = loss_func,
                metrics=['accuracy'])
    x_train = [tf.image.random_flip_left_right(i) for i in x_train] # 对训练集进行数据增广(左右翻转)
    x_train = np.array(x_train)
    model.fit(x_train, y_train, epochs = 5, batch_size = 64)
    loss, accuracy = model.evaluate(x_test, y_test)
