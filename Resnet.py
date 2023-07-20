import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten, GlobalAveragePooling2D, Add, Activation, BatchNormalization, ZeroPadding2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model

with tf.devide('/gpu:1'):
    def basic_block_identity(inp, filters, kernel_size, block, layer):
        conv_name = 'basic_conv_b' + block + '_l' + layer
        batch_name = 'basic_batch_b' + block + '_l' + layer

        z = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', kernel_initializer = 'he_normal', name = conv_name + '_a')(inp)
        z = BatchNormalization(name = batch_name + '_a')(z)
        z = Activation('relu')(z)

        z = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', kernel_initializer = 'he_normal', name = conv_name + '_b')(z)
        z = BatchNormalization(name = batch_name + '_b')(z)
        z = Activation('relu')(z)

        add = Add()([inp, z])
        z = Activation('relu')(add) 
        return z

    def basic_block_convolutional(inp, filters, kernel_size, block, layer, strides = 2):
        conv_name = 'basic_conv_b' + block + '_l' + layer
        batch_name = 'basic_batch_b' + block + '_l' + layer

        w = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', strides = 1, kernel_initializer = 'he_normal', name = conv_name + '_a')(inp)
        w = BatchNormalization(name = batch_name + '_a')(w)
        w = Activation('relu')(w)

        w = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', strides = strides, kernel_initializer = 'he_normal', name = conv_name + '_b')(w)
        w = BatchNormalization(name = batch_name + '_b')(w)
        w = Activation('relu')(w)

        # shortcut需要添加一个1*1的卷积层,为了统计通道,而可以add
        shortcut = Conv2D(filters = filters, kernel_size = 1, strides = strides, kernel_initializer = 'he_normal', name = conv_name + '_shortcut')(inp)  # 若这个卷积块涉及到size变换, 那么要将计算shortcut的conv的strides设置为2
        shortcut = BatchNormalization(name = batch_name + '_shortcut')(shortcut)

        add = Add()([shortcut, w])
        w = Activation('relu')(add) 
        return w

    # identity_block与convolutional_block都是bottle neck
    def identity_block(inp, filters, kernel_size, block, layer):  
    # 此函数针对于尺寸一样时的相加运算, 直接相加  
        f1, f2, f3 = filters  
        conv_name = 'id_conv_b' + block + '_l' + layer
        batch_name = 'id_batch_b' + block + '_l' + layer
        
        x = Conv2D(filters = f1, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal', name = conv_name + '_a')(inp)
        x = BatchNormalization(name = batch_name + '_a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters = f2, kernel_size = kernel_size, padding = 'same', kernel_initializer = 'he_normal', name = conv_name + '_b')(x)
        x = BatchNormalization(name = batch_name + '_b')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters = f3, kernel_size = 1, padding='same', kernel_initializer = 'he_normal', name = conv_name + '_c')(x)
        x = BatchNormalization(name = batch_name + '_c')(x)
        
        add = Add()([inp, x])
        x = Activation('relu')(add) 
        return x

    def convolutional_block(inp, filters, kernel_size, block, layer, strides = 1):  # 
    # 此函数针对于尺寸不一样时的相加运算, 要用1*1的conv卷积来调整通道数, 通道数调整成f3   
        f1, f2, f3 = filters
        
        conv_name = 'res_conv_b' + block + '_l' + layer
        batch_name = 'res_batch_b' + block + '_l' + layer
        
        y = Conv2D(filters = f1, kernel_size = 1, padding = 'same', strides = 1, kernel_initializer = 'he_normal', name = conv_name + '_a')(inp)  # 若这个卷积块涉及到size变换, 那么只需要将卷积块的第一层conv的strides设置为2
        y = BatchNormalization(name = batch_name + '_a')(y)
        y = Activation('relu')(y)
        
        y = Conv2D(filters = f2, kernel_size = kernel_size, padding = 'same', strides = strides, kernel_initializer = 'he_normal', name = conv_name + '_b')(y)
        y = BatchNormalization(name = batch_name + '_b')(y)
        y = Activation('relu')(y)
        
        y = Conv2D(filters = f3, kernel_size = 1, padding = 'same', strides = 1, kernel_initializer = 'he_normal', name=conv_name + '_c')(y)
        y = BatchNormalization(name = batch_name + '_c')(y)
        
        shortcut = Conv2D(filters = f3, kernel_size = 1, strides = strides, kernel_initializer = 'he_normal', name = conv_name + '_shortcut')(inp)  # 若这个卷积块涉及到size变换, 那么要将计算shortcut的conv的strides设置为2
        shortcut = BatchNormalization(name = batch_name + '_shortcut')(shortcut)
        
        add = Add()([shortcut, y])
        y = Activation('relu')(add)  
        return y

    img = Input(shape=(28, 28, 1), name='input')
    padd = ZeroPadding2D(3)(img)

    conv1 = Conv2D(64, 7, strides = 2, padding = 'valid', name = 'conv1')(padd)  # (32, 14, 14, 64)
    conv1 = BatchNormalization(name = 'batch2')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = ZeroPadding2D(1)(conv1)
    conv1 = MaxPool2D(3, 2)(conv1)  # (32, 7, 7, 64)

    #  Resnet50如下
    # conv2 = convolutional_block(conv1, [64, 64, 256], 3, '2', '1')  #conv1的尺寸与经过convolutional_block输出的尺寸不同, 所以要用convolutional_block. 但是因为前面第一层的conv1已经strides=2了, 所以conv2第一层的strides设置为1, 这里不同于后面的convolutional_block设置, 其它Resnet是一样的, 具体可以看Resnet的流程图
    # conv2 = identity_block(conv2, [64, 64, 256], 3, '2', '2')
    # conv2 = identity_block(conv2, [64, 64, 256], 3, '2', '3')

    # conv3 = convolutional_block(conv2, [128, 128, 512], 3, '3', '1', strides = 2)  # 这不能设置strides=1, 因为要使尺寸不同的shortcut经过conv(1*1)能变成与该convolutional_block输出的y通道数相同, 就应该在输入该
    # conv3 = identity_block(conv3, [128, 128, 512], 3, '3', '2')
    # conv3 = identity_block(conv3, [128, 128, 512], 3, '3', '3')
    # conv3 = identity_block(conv3, [128, 128, 512], 3, '3', '4')

    # conv4 = convolutional_block(conv3, [256, 256, 1024], 3, '4', '1', strides = 2)
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '2')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '3')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '4')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '5')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '6')

    # conv5 = convolutional_block(conv4, [512, 512, 2048], 3, '5', '1', strides = 2)
    # conv5 = identity_block(conv5, [512, 512, 2048], 3, '5', '2')
    # conv5 = identity_block(conv5, [512, 512, 2048], 3, '5', '3')

    #Resnet101
    # conv2 = convolutional_block(conv1, [64, 64, 256], 3, '2', '1', strides = 2)  #conv1的尺寸与经过convolutional_block输出的尺寸不同, 所以要用convolutional_block. 但是因为前面第一层的conv1已经strides=2了, 所以conv2第一层的strides设置为1, 这里不同于后面的convolutional_block设置, 其它Resnet是一样的, 具体可以看Resnet的流程图
    # conv2 = identity_block(conv2, [64, 64, 256], 3, '2', '2')HE21e_data
    # conv2 = identity_block(conv2, [64, 64, 256], 3, '2', '3')

    # conv3 = convolutional_block(conv2, [128, 128, 512], 3, '3', '1')  # 这不能设置strides=1, 因为要使尺寸不同的shortcut经过conv(1*1)能变成与该convolutional_block输出的y通道数相同, 就应该在输入该
    # conv3 = identity_block(conv3, [128, 128, 512], 3, '3', '2')
    # conv3 = identity_block(conv3, [128, 128, 512], 3, '3', '3')
    # conv3 = identity_block(conv3, [128, 128, 512], 3, '3', '4')

    # conv4 = convolutional_block(conv3, [256, 256, 1024], 3, '4', '1')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '2')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '3')o
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '5')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '6')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '7')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '8')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '9')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '10')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '11')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '12')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '13')o
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '14')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '15')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '16')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '17')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '18')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '19')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '20')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '21')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '22')
    # conv4 = identity_block(conv4, [256, 256, 1024], 3, '4', '23')

    # conv5 = convolutional_block(conv4, [512, 512, 2048], 3, '5', '1')
    # conv5 = identity_block(conv5, [512, 512, 2048], 3, '5', '2')
    # conv5 = identity_block(conv5, [512, 512, 2048], 3, '5', '3HE21e_data')

    #Resnet34如下
    # conv2 = basic_block_convolutional(conv1, 64, 3, '2', '1', strides = 1)
    # conv2 = basic_block_identity(conv2, 64, 3, '2', '2')
    # conv2 = basic_block_identity(conv2, 64, 3, '2', '3')

    # conv3 = basic_block_convolutional(conv2, 128, 3, '3', '1')
    # conv3 = basic_block_identity(conv3, 128, 3, '3', '2')
    # conv3 = basic_block_identity(conv3, 128, 3, '3', '3')
    # conv3 = basic_block_identity(conv3, 128, 3, '3', '4')

    # conv4 = basic_block_convolutional(conv3, 256, 3, '4', '1')
    # conv4 = basic_block_identity(conv4, 256, 3, '4', '2')
    # conv4 = basic_block_identity(conv4, 256, 3, '4', '3')
    # conv4 = basic_block_identity(conv4, 256, 3, '4', '4')
    # conv4 = basic_block_identity(conv4, 256, 3, '4', '5')
    # conv4 = basic_block_identity(conv4, 256, 3, '4', '6')

    # conv5 = basic_block_convolutional(conv4, 512, 3, '5',  '1')
    # conv5 = basic_block_identity(conv5, 512, 3, '5', '2')
    # conv5 = basic_block_identity(conv5, 512, 3, '5', '3')

    #Resnet18
    conv2 = basic_block_convolutional(conv1, 64, 3, '2', '1', strides = 1)
    conv2 = basic_block_identity(conv2, 64, 3, '2', '2')

    conv3 = basic_block_convolutional(conv2, 128, 3, '3', '1')
    conv3 = basic_block_identity(conv3, 128, 3, '3', '2')

    conv4 = basic_block_convolutional(conv3, 256, 3, '4', '1')
    conv4 = basic_block_identity(conv4, 256, 3, '4', '2')

    conv5 = basic_block_convolutional(conv4, 512, 3, '5',  '1')
    conv5 = basic_block_identity(conv5, 512, 3, '5', '2')

    avg_pool = GlobalAveragePooling2D()(conv5) # 因为ResNet后面又接了一个Dense,所以这个GAP层前一个卷积层的输出通道数不用是分类的类别数,这里的GAP就主要起一个减少参数的作用
    dense = Dense(10, activation='softmax')(avg_pool)  # 最后一个fc层, 10根据图片的类别数而改变

    model = Model(img, dense)
    adam = Adam(0.001)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer = adam, 
        loss = loss_func,
        metrics = ['accuracy']
        )
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape(-1, 28, 28, 1) / 255.0, x_test.reshape(-1, 28, 28, 1) / 255.0  # (60000, 28, 28)
    model.fit(x_train, y_train, batch_size = 64, epochs = 5)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(loss, accuracy)