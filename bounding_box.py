import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with tf.device('/gpu:1'):
    img = plt.imread('cat_dog.jpg')
    # 在目标检测中,我们常使用边缘框来描述目标位置,它是一个矩形框
    dog_bbox, cat_bbox = [10, 10, 200, 240], [190, 45, 320, 220]
    # 将上述边缘框转换为matplotlib的边缘框格式
    def bbox_to_rect(bbox, color):
        # 将边缘框(左上x, 左上y, 右下x, 右下y)转换为((左上x, 左上y), 宽, 高)
        return plt.Rectangle(
            xy = (bbox[0], bbox[1]), width = bbox[2] - bbox[0], height = bbox[3] - bbox[1],
            fill = False, edgecolor = color, linewidth = 2)

    fig = plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    plt.show()