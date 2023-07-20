import tensorflow as tf
import numpy as np
import os
import zipfile
import wget
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

with tf.device('/gpu:1'):
    # def download_data():
    #     data = os.getcwd() + '/data'
    #     base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/'
    #     wget.download(
    #         base_url + 'gluon/dataset/hotdog.zip',
    #         data
    #     )
    #     with zipfile.ZipFile('data', 'r') as z:
    #         z.extractall(os.getcwd())
    # download_data()
    # 读取下载下来的数据集,并分为训练集和测试集
    with zipfile.ZipFile('hotdog.zip', 'r') as z:
            z.extractall(os.getcwd())
    train_dir = 'hotdog/train'
    test_dir = 'hotdog/test'
    train_dir = pathlib.Path(train_dir)
#     a = np.array([item.name for item in train_dir.glob('*')]) # train_dir对象(就是一个路径)下符合条件的文件,此路径下只有'hotdog'和'not-hotdog'
#     print(a)
    train_count = len(list(train_dir.glob('*/*.png')))
    print(train_count)
    test_dir = pathlib.Path(test_dir)
    test_count = len(list(test_dir.glob('*/*.png')))
    print(test_count)    

    CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != 'LICENSE.txt' and item.name[0] != '.'])
    print(CLASS_NAMES)

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    BATCH_SIZE = 32
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    train_data_gen = image_generator.flow_from_directory(directory=str(train_dir),
                                                        batch_size=BATCH_SIZE,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        shuffle=True,
                                                        classes = list(CLASS_NAMES))

    test_data_gen = image_generator.flow_from_directory(directory=str(test_dir),
                                                        batch_size=BATCH_SIZE,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        shuffle=True,
                                                        classes = list(CLASS_NAMES))

     # 定义和初始化模型
    ResNet50 = tf.keras.applications.resnet_v2.ResNet50V2(include_top = False, weights = 'imagenet', pooling = 'avg', input_shape = (224, 224, 3)) # include_top默认等于true,所以只能是(224, 224, 3)
    
    print(ResNet50.layers[-2])
    print(len(ResNet50.layers))
    # 冻结所有层method 1
#     for layer in ResNet50.layers:
#        layer.trainable = False # (默认会训练的)冻结源数据集的ResNet50层参数,后面目标数据集中也不会更改
    # 冻结所有层method 2
    ResNet50.trainable = False

    model = Sequential()
    model.add(ResNet50) 
    model.add(Dense(2, activation = 'softmax')) # 在源数据集的ResNet50上再加两层,而目标数据集则只训练这一层

    model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    history = model.fit(
                    train_data_gen,
                    epochs=5
                    )