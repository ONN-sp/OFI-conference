# FIO会议(Deep Learning Network-based Optical Vector-Eigenmode Decomposition for Mode-Division Multiplexing Links)的深度学习代码
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten, GlobalAveragePooling2D, Add, Activation, BatchNormalization, ZeroPadding2D, Dropout
from tensorflow.keras import utils as np_utils
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.datasets import mnist
from PIL import Image
import math
import scipy
import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import colormap as cm
import matplotlib.gridspec as gridspec
import time

def notjianbing_4(fudu, angle_list, output_result):
    output = []
    output_angle1 = []
    output2 = []
    for i in range(output_result.shape[0]):
        output_temp = np.arcsin((output_result[i][3:5])*2-1)*2
        output.append(np.sqrt(output_result[i][0:3]))
        output2.append(output_result[i][0:3])
        output_angle1.append(output_temp)
        output_data = tf.concat([output, output_angle1], axis = 1)
    output_data = np.array(output_data)
    output_angle1 = np.array(output_angle1)
    output2 = np.array(output2)
    HE11e_fudu_error = np.mean(np.abs(fudu[34000:35000, 0] - output2[:,0]))
    HE31o_fudu_error = np.mean(np.abs(fudu[34000:35000, 1] - output2[:,1]))
    HE41o_fudu_error = np.mean(np.abs(fudu[34000:35000, 2] - output2[:,2]))
    HE31o_angle_error = np.mean(np.abs(angle_list[34000:35000, 0] - output_angle1[:,0])/(2*math.pi))
    HE41o_angle_error = np.mean(np.abs(angle_list[34000:35000, 1] - output_angle1[:,1])/(2*math.pi))
    fudu_error = np.mean(np.abs(fudu[34000:35000] - output2))
    angle_error = np.mean(np.abs(angle_list[34000:35000] - output_angle1)/(2*math.pi))
    print(HE11e_fudu_error)
    print(HE31o_fudu_error)
    print(HE41o_fudu_error)
    print(HE31o_angle_error)
    print(HE41o_angle_error)
    print(fudu_error)
    print(angle_error)
    return output_data

def to_one(x):
    y = (x - np.min(x)) / (np.max(x) - np.min(x))
    return y

def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

def correction_2(m, r, data1, data2):
    orginal_data = []
    max_cor = []
    max_data = []
    best_value = []
    for i in range(r.shape[0]):
        data_value1 = r[i][0]*data1*np.exp(1j*r[i][2])  #TM
        data_value2 = r[i][1]*data2*np.exp(1j*r[i][3])
        measure_data = m[i][0]*data1*np.exp(1j*m[i][2])+m[i][1]*data2*np.exp(1j*m[i][3])
        orginal_data.append(measure_data)
        result_no_zhengfu = data_value1+data_value2
        result = corr2(np.abs(result_no_zhengfu)**2, np.abs(measure_data)**2)      
        max_cor.append(result)
        best_value.append(concat_np4(r[i][0], r[i][1], r[i][2], r[i][3]))
        max_data.append(result_no_zhengfu)
    return np.array(orginal_data), np.mean(max_cor), np.array(max_data), best_value

def correction_3(m, r, data1, data2, data3):
    max_cor = []
    best_value = []
    orignal_data = []
    print(r[0])
    print(m[0])
    max_data = []
    for i in range(r.shape[0]):
        data_value1 = r[i][0]*data1*np.exp(1j*r[i][3])  
        # data_value1 = r[i][0]*data1*np.exp(1j*0)  # HE11e
        data_value2 = r[i][1]*data2*np.exp(1j*r[i][4])  
        data_value3 = r[i][2]*data3*np.exp(1j*r[i][5])  

        measure_data = m[i][0]*data1*np.exp(1j*m[i][3])+m[i][1]*data2*np.exp(1j*m[i][4])+m[i][2]*data3*np.exp(1j*m[i][5])
        # measure_data = m[i][0]*data1*np.exp(1j*0)+m[i][1]*data2*np.exp(1j*m[i][3])+m[i][2]*data3*np.exp(1j*m[i][4])
        orignal_data.append(measure_data)
        result_no_zhengfu = data_value1+data_value2+data_value3
        result = corr2(np.abs(result_no_zhengfu)**2, np.abs(measure_data)**2)      
        max_cor.append(result)
        best_value.append(concat_np6(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5]))
        # best_value.append(concat_np5(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4]))
        max_data.append(result_no_zhengfu)
    return np.array(orignal_data), np.mean(max_cor), np.array(max_data), best_value

def concat_np4(x1,x2,x3,x4):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    return np.array(result)

def concat_np6(x1,x2,x3,x4,x5,x6):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    return np.array(result)

def concat_np5(x1,x2,x3,x4,x5):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    return np.array(result)

def concat_np7(x1,x2,x3,x4,x5,x6,x7):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    result.append(x7)
    return np.array(result)

def concat_np8(x1,x2,x3,x4,x5,x6,x7,x8):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    result.append(x7)
    result.append(x8)
    return np.array(result)

def concat_np9(x1,x2,x3,x4,x5,x6,x7,x8,x9):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    result.append(x7)
    result.append(x8)
    result.append(x9)
    return np.array(result)

def concat_np11(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    result.append(x7)
    result.append(x8)
    result.append(x9)
    result.append(x10)
    result.append(x11)
    return np.array(result)

def concat_np15(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    result.append(x7)
    result.append(x8)
    result.append(x9)
    result.append(x10)
    result.append(x11)
    result.append(x12)
    result.append(x13)
    result.append(x14)
    result.append(x15)
    return np.array(result)

def concat_np19(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    result.append(x7)
    result.append(x8)
    result.append(x9)
    result.append(x10)
    result.append(x11)
    result.append(x12)
    result.append(x13)
    result.append(x14)
    result.append(x15)
    result.append(x16)
    result.append(x17)
    result.append(x18)
    result.append(x19)
    return np.array(result)

def correction_4(m, r, data1, data2, data3, data4):
    max_cor = []
    best_value = []
    orignal_data = []
    print(r[0])
    print(m[0])
    max_data = []
    for i in range(r.shape[0]):
        # data_value1 = r[i][0]*data1*np.exp(1j*0)  #HE11e
        data_value1 = r[i][0]*data1*np.exp(1j*r[i][4])  
        data_value2 = r[i][1]*data2*np.exp(1j*r[i][5])  
        data_value3 = r[i][2]*data3*np.exp(1j*r[i][6])  
        data_value4 = r[i][3]*data4*np.exp(1j*r[i][7])  

        # result_list1 = data_value1+data_value2+data_value3+data_value4
        # result_value1 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6], r[i][7])
        # result_list1 = np.array(result_list1).reshape(128, 128,1)

        # result_list2 = data_value5+data_value2+data_value3+data_value4
        # result_value2 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][8], r[i][5], r[i][6], r[i][7])
        # result_list2 = np.array(result_list2).reshape(128, 128,1)

        # result_list3 = data_value1+data_value6+data_value3+data_value4
        # result_value3 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][9], r[i][6], r[i][7])
        # result_list3 = np.array(result_list3).reshape(128, 128,1)

        # result_list4 = data_value1+data_value2+data_value7+data_value4
        # result_value4 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][10], r[i][7])
        # result_list4 = np.array(result_list4).reshape(128, 128,1)

        # result_list5 = data_value1+data_value2+data_value3+data_value8
        # result_value5 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6], r[i][11])
        # result_list5 = np.array(result_list5).reshape(128, 128,1)

        # result_list6 = data_value5+data_value6+data_value3+data_value4
        # result_value6 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][8], r[i][9], r[i][6], r[i][7])
        # result_list6 = np.array(result_list6).reshape(128, 128,1)

        # result_list7 = data_value5+data_value2+data_value7+data_value4
        # result_value7 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][8], r[i][5], r[i][10], r[i][7])
        # result_list7 = np.array(result_list7).reshape(128,128,1)

        # result_list8 = data_value5+data_value2+data_value3+data_value8
        # result_value8 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][8], r[i][5], r[i][6], r[i][11])
        # result_list8 = np.array(result_list8).reshape(128,128,1)

        # result_list9 = data_value1+data_value6+data_value7+data_value4
        # result_value9 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][9], r[i][10], r[i][7])
        # result_list9 = np.array(result_list9).reshape(128,128,1)

        # result_list10 = data_value1+data_value6+data_value3+data_value8
        # result_value10 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][9], r[i][6], r[i][11])
        # result_list10 = np.array(result_list10).reshape(128,128,1)

        # result_list11 = data_value1+data_value2+data_value7+data_value8
        # result_value11 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][10], r[i][11])
        # result_list11 = np.array(result_list11).reshape(128,128,1)

        # result_list12 = data_value1+data_value6+data_value7+data_value8
        # result_value12 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][9], r[i][10], r[i][11])
        # result_list12 = np.array(result_list12).reshape(128,128,1)

        # result_list13 = data_value5+data_value6+data_value7+data_value4
        # result_value13 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][8], r[i][9], r[i][10], r[i][7])
        # result_list13 = np.array(result_list13).reshape(128,128,1)

        # result_list14 = data_value5+data_value2+data_value7+data_value8
        # result_value14 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][8], r[i][5], r[i][10], r[i][11])
        # result_list14 = np.array(result_list14).reshape(128,128,1)

        # result_list15 = data_value5+data_value6+data_value3+data_value8
        # result_value15 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][8], r[i][9], r[i][6], r[i][11])
        # result_list15 = np.array(result_list15).reshape(128,128,1)

        # result_list16 = data_value5+data_value6+data_value7+data_value8
        # result_value16 = concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][8], r[i][9], r[i][10], r[i][11])
        # result_list16 = np.array(result_list16).reshape(128,128,1)

        # # r1 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][4],r[i][5],r[i][6],r[i][7])
        # # r2 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][8],r[i][5],r[i][6],r[i][7])
        # # r3 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][4],r[i][9],r[i][6],r[i][7])
        # # r4 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][4],r[i][5],r[i][10],r[i][7])
        # # r5 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][4],r[i][5],r[i][6],r[i][11])

        # # r6 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][8],r[i][9],r[i][6],r[i][7])
        # # r7 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][8],r[i][5],r[i][10],r[i][7])
        # # r8 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][8],r[i][5],r[i][6],r[i][11])
        # # r9 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][4],r[i][9],r[i][10],r[i][7])
        # # r10 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][4],r[i][9],r[i][6],r[i][11])
        # # r11 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][4],r[i][5],r[i][10],r[i][11])

        # # r12 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][4],r[i][9],r[i][10],r[i][11])
        # # r13 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][8],r[i][9],r[i][10],r[i][7])
        # # r14 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][8],r[i][5],r[i][10],r[i][11])
        # # r15 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][8],r[i][9],r[i][6],r[i][11])

        # # r16 = concat_np8(r[i][0],r[i][1],r[i][2],r[i][3],r[i][8],r[i][9],r[i][10],r[i][11])

        # result_list_all = concat_np16(result_list1, result_list2, result_list3, result_list4, result_list5, result_list6, result_list7, result_list8, result_list9, result_list10, result_list11, result_list12, result_list13, result_list14, result_list15, result_list16)
        # result_value = concat_np16(result_value1, result_value2, result_value3, result_value4, result_value5, result_value6, result_value7, result_value8, result_value9, result_value10, result_value11, result_value12, result_value13, result_value14, result_value15, result_value16)

        # measure_data = m[i][0]*data1*np.exp(1j*0)+m[i][1]*data2*np.exp(1j*m[i][4])+m[i][2]*data3*np.exp(1j*m[i][5])+m[i][3]*data4*np.exp(1j*m[i][6])
        measure_data = m[i][0]*data1*np.exp(1j*r[i][4])+m[i][1]*data2*np.exp(1j*m[i][5])+m[i][2]*data3*np.exp(1j*m[i][6])+m[i][3]*data4*np.exp(1j*m[i][7])
        orignal_data.append(measure_data)
        result_no_zhengfu = data_value1+data_value2+data_value3+data_value4
        result = corr2(np.abs(result_no_zhengfu)**2,np.abs(measure_data)**2)
        # result1 = corr2(np.abs(result_list1),np.abs(measure_data))
        # result2 = corr2(np.abs(result_list2),np.abs(measure_data))
        # result3 = corr2(np.abs(result_list3),np.abs(measure_data))
        # result4 = corr2(np.abs(result_list4),np.abs(measure_data))
        # result5 = corr2(np.abs(result_list5),np.abs(measure_data))
        # result6 = corr2(np.abs(result_list6),np.abs(measure_data))
        # result7 = corr2(np.abs(result_list7),np.abs(measure_data))   
        # result8 = corr2(np.abs(result_list8),np.abs(measure_data))
        # result9 = corr2(np.abs(result_list9),np.abs(measure_data))
        # result10 = corr2(np.abs(result_list10),np.abs(measure_data))
        # result11 = corr2(np.abs(result_list11),np.abs(measure_data))
        # result12 = corr2(np.abs(result_list12),np.abs(measure_data))
        # result13 = corr2(np.abs(result_list13),np.abs(measure_data))
        # result14 = corr2(np.abs(result_list14),np.abs(measure_data))
        # result15 = corr2(np.abs(result_list15),np.abs(measure_data))
        # result16 = corr2(np.abs(result_list16),np.abs(measure_data))

        # result1 = corr2(np.angle(result_list1),np.angle(measure_data))
        # result2 = corr2(np.angle(result_list2),np.angle(measure_data))
        # result3 = corr2(np.angle(result_list3),np.angle(measure_data))
        # result4 = corr2(np.angle(result_list4),np.angle(measure_data))
        # result5 = corr2(np.angle(result_list5),np.angle(measure_data))
        # result6 = corr2(np.angle(result_list6),np.angle(measure_data))
        # result7 = corr2(np.angle(result_list7),np.angle(measure_data))
        # result8 = corr2(np.angle(result_list8),np.angle(measure_data))
        # result9 = corr2(np.angle(result_list9),np.angle(measure_data))
        # result10 = corr2(np.angle(result_list10),np.angle(measure_data))
        # result11 = corr2(np.angle(result_list11),np.angle(measure_data))
        # result12 = corr2(np.angle(result_list12),np.angle(measure_data))
        # result13 = corr2(np.angle(result_list13),np.angle(measure_data))
        # result14 = corr2(np.angle(result_list14),np.angle(measure_data))
        # result15 = corr2(np.angle(result_list15),np.angle(measure_data))
        # result16 = corr2(np.angle(result_list16),np.angle(measure_data))

        # result1 = corr2(result_list1,measure_data)
        # result2 = corr2(result_list2,measure_data)
        # result3 = corr2(result_list3,measure_data)
        # result4 = corr2(result_list4,measure_data)
        # result5 = corr2(result_list5,measure_data)
        # result6 = corr2(result_list6,measure_data)
        # result7 = corr2(result_list7,measure_data)
        # result8 = corr2(result_list8,measure_data)
        # result9 = corr2(result_list1,measure_data)
        # result10 = corr2(result_list2,measure_data)
        # result11 = corr2(result_list3,measure_data)
        # result12 = corr2(result_list4,measure_data)
        # result13 = corr2(result_list5,measure_data)
        # result14 = corr2(result_list6,measure_data)
        # result15 = corr2(result_list7,measure_data)
        # result16 = corr2(result_list8,measure_data)

        # result1 = corr2(r1,m[i])
        # result2 = corr2(r2,m[i])
        # result3 = corr2(r3,m[i])
        # result4 = corr2(r4,m[i])
        # result5 = corr2(r5,m[i])
        # result6 = corr2(r6,m[i])
        # result7 = corr2(r7,m[i])
        # result8 = corr2(r8,m[i])
        # result9 = corr2(r9,m[i])
        # result10 = corr2(r10,m[i])
        # result11 = corr2(r11,m[i])
        # result12 = corr2(r12,m[i])
        # result13 = corr2(r13,m[i])
        # result14 = corr2(r14,m[i])
        # result15 = corr2(r15,m[i])
        # result16 = corr2(r16,m[i])

        # result = np.array([result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11, result12, result13, result14, result15, result16])
        # max_cor.append(np.max(result))
        # index = np.argmax(result)
        max_cor.append(result)
        # best_value.append(result_value[index])
        best_value.append(concat_np8(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6], r[i][7]))
        # best_value.append(concat_np7(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6]))
        # max_data.append(result_list_all[index])
        max_data.append(result_no_zhengfu)
    return np.array(orignal_data), np.mean(max_cor), np.array(max_data), best_value

def correction_6(m, r, data1, data2, data3, data4, data5, data6):
    max_cor = []
    best_value = []
    orignal_data = []
    print(r[0])
    print(m[0])
    max_data = []
    for i in range(r.shape[0]):
        data_value1 = r[i][0]*data1*np.exp(1j*0)  
        data_value2 = r[i][1]*data2*np.exp(1j*r[i][6])  
        data_value3 = r[i][2]*data3*np.exp(1j*r[i][7]) # HE11e
        data_value4 = r[i][3]*data4*np.exp(1j*r[i][8])  
        data_value5 = r[i][4]*data5*np.exp(1j*r[i][9])
        data_value6 = r[i][5]*data6*np.exp(1j*r[i][10]) 

        measure_data = m[i][0]*data1*np.exp(1j*0)+m[i][1]*data2*np.exp(1j*m[i][6])+m[i][2]*data3*np.exp(1j*m[i][7])+m[i][3]*data4*np.exp(1j*m[i][8])+m[i][4]*data5*np.exp(1j*m[i][9])+m[i][5]*data6*np.exp(1j*m[i][10])
        orignal_data.append(measure_data)
        result_no_zhengfu = data_value1+data_value2+data_value3+data_value4+data_value5+data_value6
        result = corr2(np.abs(result_no_zhengfu)**2,np.abs(measure_data)**2)
        max_cor.append(result)
        best_value.append(concat_np11(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6], r[i][7], r[i][8], r[i][9], r[i][10]))
        max_data.append(result_no_zhengfu)
    return np.array(orignal_data), np.mean(max_cor), np.array(max_data), best_value

def correction_8(m, r, data1, data2, data3, data4, data5, data6, data7, data8):
    max_cor = []
    best_value = []
    orignal_data = []
    print(r[0])
    print(m[0])
    max_data = []
    for i in range(r.shape[0]):
        data_value1 = r[i][0]*data1*np.exp(1j*r[i][8])  
        data_value2 = r[i][1]*data2*np.exp(1j*r[i][9])  
        data_value3 = r[i][2]*data3*np.exp(1j*0) # HE11e
        data_value4 = r[i][3]*data4*np.exp(1j*r[i][10])  
        data_value5 = r[i][4]*data5*np.exp(1j*r[i][11]) 
        data_value6 = r[i][5]*data6*np.exp(1j*r[i][12]) 
        data_value7 = r[i][6]*data7*np.exp(1j*r[i][13])
        data_value8 = r[i][7]*data8*np.exp(1j*r[i][14]) 

        measure_data = m[i][0]*data1*np.exp(1j*m[i][8])+m[i][1]*data2*np.exp(1j*m[i][9])+m[i][2]*data3*np.exp(1j*0)+m[i][3]*data4*np.exp(1j*m[i][10])+m[i][4]*data5*np.exp(1j*m[i][11])+m[i][5]*data6*np.exp(1j*m[i][12])+m[i][6]*data7*np.exp(1j*m[i][13])+m[i][7]*data8*np.exp(1j*m[i][14])
        orignal_data.append(measure_data)
        result_no_zhengfu = data_value1+data_value2+data_value3+data_value4+data_value5+data_value6+data_value7+data_value8
        result = corr2(np.abs(result_no_zhengfu)**2,np.abs(measure_data)**2)
        max_cor.append(result)
        best_value.append(concat_np15(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6], r[i][7], r[i][8], r[i][9], r[i][10], r[i][11], r[i][12], r[i][13], r[i][14]))
        max_data.append(result_no_zhengfu)
    return np.array(orignal_data), np.mean(max_cor), np.array(max_data), best_value

def correction_10(m, r, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10):
    max_cor = []
    best_value = []
    orignal_data = []
    print(r[0])
    print(m[0])
    max_data = []
    for i in range(r.shape[0]):
        data_value1 = r[i][0]*data1*np.exp(1j*r[i][10])  
        data_value2 = r[i][1]*data2*np.exp(1j*r[i][11])  
        data_value3 = r[i][2]*data3*np.exp(1j*r[i][12])  
        data_value4 = r[i][3]*data4*np.exp(1j*r[i][13])  
        data_value5 = r[i][4]*data5*np.exp(1j*0) # HE11e
        data_value6 = r[i][5]*data6*np.exp(1j*r[i][14]) 
        data_value7 = r[i][6]*data7*np.exp(1j*r[i][15])
        data_value8 = r[i][7]*data8*np.exp(1j*r[i][16]) 
        data_value9 = r[i][8]*data9*np.exp(1j*r[i][17])
        data_value10 = r[i][9]*data10*np.exp(1j*r[i][18])

        measure_data = m[i][0]*data1*np.exp(1j*m[i][10])+m[i][1]*data2*np.exp(1j*m[i][11])+m[i][2]*data3*np.exp(1j*m[i][12])+m[i][3]*data4*np.exp(1j*m[i][13])+m[i][4]*data5*np.exp(1j*0)+m[i][5]*data6*np.exp(1j*m[i][14])+m[i][6]*data7*np.exp(1j*m[i][15])+m[i][7]*data8*np.exp(1j*m[i][16])+m[i][8]*data9*np.exp(1j*m[i][17])+m[i][9]*data10*np.exp(1j*m[i][18])
        orignal_data.append(measure_data)
        result_no_zhengfu = data_value1+data_value2+data_value3+data_value4+data_value5+data_value6+data_value7+data_value8+data_value9+data_value10
        result = corr2(np.abs(result_no_zhengfu)**2,np.abs(measure_data**2))
        max_cor.append(result)
        best_value.append(concat_np19(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6], r[i][7], r[i][8], r[i][9], r[i][10], r[i][11], r[i][12], r[i][13], r[i][14], r[i][15], r[i][16], r[i][17], r[i][18]))
        max_data.append(result_no_zhengfu)
    return np.array(orignal_data), np.mean(max_cor), np.array(max_data), best_value

def get_data(folder):  # folder是根目录文件夹
    result_list = []
    label_list = []
    file_list = []
    angle_temp2 = []
    fudu = []
    angle_list = []
    for file in os.listdir(folder):
        file_list.append(file)
        # img = Image.open(folder + file)
        img = scipy.io.loadmat(folder + file)
        data_values1 = img['E'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
        data_label = np.array(img['mode_coef'].reshape(8, ))
        abs_temp = np.abs(data_values1)
        angle_temp = np.angle(data_values1)
        data = tf.concat([abs_temp, angle_temp], axis = 2)
        # anglelabel_temp = data_label[2:4]
        # anglelabel_temp = data_label[3:6]
        # anglelabel_temp = data_label[3:5] # not jianbing modal(3)
        anglelabel_temp = data_label[4:8]
        # anglelabel_temp = data_label[4:7] # not jianbing modal(4)
        # anglelabel_temp = data_label[6:11] #six modal
        # anglelabel_temp = data_label[8:15] #eight modal
        # anglelabel_temp = data_label[10:19] #ten modal
        angle_list.append(anglelabel_temp)
        angle_temp2.append(data_label)
        anglelabel_temp2 = np.sin(anglelabel_temp/2)
        anglelabel_temp3 = np.divide(anglelabel_temp2 + 1, 2)
        # anglelabel_temp3 = np.divide(anglelabel_temp + math.pi, 2*math.pi)
        # label = [data_label[0]**2, data_label[1]**2, anglelabel_temp3[0], anglelabel_temp3[1]]
        # label = [data_label[0]**2, data_label[1]**2, data_label[2]**2, anglelabel_temp3[0], anglelabel_temp3[1], anglelabel_temp3[2]]
        # label = [data_label[0]**2, data_label[1]**2, data_label[2]**2, anglelabel_temp3[0], anglelabel_temp3[1]] # not jianbing modal(3)
        label = [data_label[0]**2, data_label[1]**2, data_label[2]**2, data_label[3]**2, anglelabel_temp3[0], anglelabel_temp3[1], anglelabel_temp3[2], anglelabel_temp3[3]]
        # label = [data_label[0]**2, data_label[1]**2, data_label[2]**2, data_label[3]**2, anglelabel_temp3[0], anglelabel_temp3[1], anglelabel_temp3[2]] # not jianbing modal(4)
        # label = [data_label[0]**2, data_label[1]**2, data_label[2]**2, data_label[3]**2, data_label[4]**2, data_label[5]**2, anglelabel_temp3[0], anglelabel_temp3[1], anglelabel_temp3[2], anglelabel_temp3[3], anglelabel_temp3[4]]
        # label = [data_label[0]**2, data_label[1]**2, data_label[2]**2, data_label[3]**2, data_label[4]**2, data_label[5]**2, data_label[6]**2, data_label[7]**2, anglelabel_temp3[0], anglelabel_temp3[1], anglelabel_temp3[2], anglelabel_temp3[3], anglelabel_temp3[4], anglelabel_temp3[5], anglelabel_temp3[6]]
        # label = [data_label[0]**2, data_label[1]**2, data_label[2]**2, data_label[3]**2, data_label[4]**2, data_label[5]**2, data_label[6]**2, data_label[7]**2,  data_label[8]**2, data_label[9]**2, anglelabel_temp3[0], anglelabel_temp3[1], anglelabel_temp3[2], anglelabel_temp3[3], anglelabel_temp3[4], anglelabel_temp3[5], anglelabel_temp3[6], anglelabel_temp3[7], anglelabel_temp3[8]]   
        fudu.append(label[0:4])
        label_list.append(label)
        result_list.append(data)
    return result_list, label_list, file_list, angle_temp2, fudu, angle_list

class Error(keras.metrics.Metric):
    def __init__(self, name = 'Error'):
        super(Error, self).__init__(name = name)
        self.true_positives = self.add_weight(name = 'ctp', initializer = 'zeros')

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')
        values = tf.abs(tf.abs(y_true) - tf.abs(y_pred))
        values = tf.cast(values, 'float32')
        self.true_positives.assign(tf.reduce_mean(values))

    def result(self):
        return self.true_positives
    
    def reset_state(self):
        self.true_positives.assign(0.0)

# TM01 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode3/TM01/E_TM011')
# TM01_data = TM01['E_TM01_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE21e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode3/HE21e/E_HE21e1')
# HE21e_data = HE21e['E_HE21e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# data_image, data_label, file_list, data_label2, fudu, angle_list = get_data('/tf/lijianjun/桌面/test/mode3/image/')

# TE01 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode4/TE01/E_TE011')
# TE01_data = TE01['E_TE01_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# TM01 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode4/TM01/E_TM011')
# TM01_data = TM01['E_TM01_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE21e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode4/HE21e/E_HE21e1')
# HE21e_data = HE21e['E_HE21e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# data_image, data_label, file_list, data_label2, fudu, angle_list = get_data('/tf/lijianjun/桌面/test/mode4/image/')

TE01 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode5/TE01/E_TE011')
TE01_data = TE01['E_TE01_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
TM01 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode5/TM01/E_TM011')
TM01_data = TM01['E_TM01_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
HE21e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode5/HE21e/E_HE21e1')
HE21e_data = HE21e['E_HE21e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
HE21o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode5/HE21o/E_HE21o1')
HE21o_data = HE21o['E_HE21o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
data_image, data_label, file_list, data_label2, fudu, angle_list = get_data('/tf/lijianjun/桌面/test/mode5/image/')

# HE11e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode6/HE11e/E_HE11e1')
# HE11e_data = HE11e['E_HE11e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE31o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode6/HE31o/E_HE31o1')
# HE31o_data = HE31o['E_HE31o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE41o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode6/HE41o/E_HE41o1')
# HE41o_data = HE41o['E_HE41o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# data_image, data_label, file_list, data_label2, fudu, angle_list = get_data('/tf/lijianjun/桌面/test/mode6/image/')
# data_image = np.array(data_image)

# concat intensity and phase
# temp = [to_one(np.abs(data_image[1])), np.angle(data_image[1])]
# fig = plt.figure()
# gs = gridspec.GridSpec(1,2)
# ax1 = plt.subplot(gs[0,0])
# ax2 = plt.subplot(gs[0,1])
# ax = [ax1, ax2]
# norm1 = matplotlib.colors.Normalize(vmin = 0.0, vmax = 1.0)
# norm2 = matplotlib.colors.Normalize(vmin = -math.pi, vmax = math.pi)
# cmap = plt.cm.jet
# for i in range(1):
#     img = temp[i]
#     im = ax[i].imshow(img, norm = norm1, cmap = cmap)
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
# ax[0].set_xlabel('Intensity')
# clb = fig.colorbar(im, ax = ax[0], ticks = [0.0, 0.5, 1.0],shrink=0.48)
# clb.set_ticklabels([0.0, 0.5, 1.0])
# for i in range(1,2):
#     img = temp[i]
#     im = ax[i].imshow(img, norm = norm2, cmap = cmap)
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
# ax[1].set_xlabel('Phase')
# clb = fig.colorbar(im, ax = ax[1], ticks = [-math.pi, 0.0, math.pi],shrink=0.48)
# clb.set_ticklabels([r'$-\pi$',0,r'$\pi$'])
# plt.savefig('/tf/lijianjun/桌面/test/mode6/HE11e/concat.svg')

# HE11e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode7/HE11e/E_HE11e1')
# HE11e_data = HE11e['E_HE11e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE31o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode7/HE31o/E_HE31o1')
# HE31o_data = HE31o['E_HE31o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE41o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode7/HE41o/E_HE41o1')
# HE41o_data = HE41o['E_HE41o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE42e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode7/HE42e/E_HE42e1')
# HE42e_data = HE42e['E_HE42e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# data_image, data_label, file_list, data_label2, fudu, angle_list = get_data('/tf/lijianjun/桌面/test/mode7/image/')

# HE11e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode8/HE11e/E_HE11e1')
# HE11e_data = HE11e['E_HE11e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE31o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode8/HE31o/E_HE31o1')
# HE31o_data = HE31o['E_HE31o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE32e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode8/HE32e/E_HE32e1')
# HE32e_data = HE32e['E_HE32e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE33o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode8/HE33o/E_HE33o1')
# HE33o_data = HE33o['E_HE33o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE41o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode8/HE41o/E_HE41o1')
# HE41o_data = HE41o['E_HE41o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE42e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode8/HE42e/E_HE42e1')
# HE42e_data = HE42e['E_HE42e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# data_image, data_label, file_list, data_label2, fudu, angle_list = get_data('/tf/lijianjun/桌面/test/mode8/image/')

# TE01 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode9/TE01/E_TE011')
# TE01_data = TE01['E_TE01_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# TE02 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode9/TE02/E_TE021')
# TE02_data = TE02['E_TE02_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE11e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode9/HE11e/E_HE11e1')
# HE11e_data = HE11e['E_HE11e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE31o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode9/HE31o/E_HE31o1')
# HE31o_data = HE31o['E_HE31o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE32e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode9/HE32e/E_HE32e1')
# HE32e_data = HE32e['E_HE32e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE33o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode9/HE33o/E_HE33o1')
# HE33o_data = HE33o['E_HE33o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE41o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode9/HE41o/E_HE41o1')
# HE41o_data = HE41o['E_HE41o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE42e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode9/HE42e/E_HE42e1')
# HE42e_data = HE42e['E_HE42e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# data_image, data_label, file_list, data_label2, fudu, angle_list = get_data('/tf/lijianjun/桌面/test/mode9/image/')

# TE01 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/TE01/E_TE011')
# TE01_data = TE01['E_TE01_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# TE02 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/TE02/E_TE021')
# TE02_data = TE02['E_TE02_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# TM03 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/TM03/E_TM031')
# TM03_data = TM03['E_TM03_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# TM04 = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/TM04/E_TM041')
# TM04_data = TM04['E_TM04_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE11e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/HE11e/E_HE11e1')
# HE11e_data = HE11e['E_HE11e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE31o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/HE31o/E_HE31o1')
# HE31o_data = HE31o['E_HE31o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE32e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/HE32e/E_HE32e1')
# HE32e_data = HE32e['E_HE32e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE33o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/HE33o/E_HE33o1')
# HE33o_data = HE33o['E_HE33o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE41o = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/HE41o/E_HE41o1')
# HE41o_data = HE41o['E_HE41o_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# HE42e = scipy.io.loadmat('/tf/lijianjun/桌面/test/mode10/HE42e/E_HE42e1')
# HE42e_data = HE42e['E_HE42e_temp'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
# data_image, data_label, file_list, data_label2, fudu, angle_list = get_data('/tf/lijianjun/桌面/test/mode10/image/')

data_image = np.array(data_image)
data_label = np.array(data_label)
fudu = np.array(fudu)
angle_list = np.array(angle_list)
print(file_list[34000])
data_image_train = data_image[0:33000]
data_label_train = data_label[0:33000]
data_image_val = data_image[33000:34000]
data_label_val = data_label[33000:34000]
data_image_test = data_image[34000:35000]
data_label_test = data_label[34000:35000]
data_image_test2 = np.empty((data_image_test.shape[0],data_image_test.shape[1],data_image_test.shape[2],data_image_test.shape[3]))
# data_image_test3 = np.empty((data_image_test.shape[0],data_image_test.shape[1],data_image_test.shape[2],data_image_test.shape[3]))
# data_image_test4 = np.empty((data_image_test.shape[0],data_image_test.shape[1],data_image_test.shape[2],data_image_test.shape[3]))
# data_image_test5 = np.empty((data_image_test.shape[0],data_image_test.shape[1],data_image_test.shape[2],data_image_test.shape[3]))
for i in range(data_image_test.shape[0]):
    noise = np.random.randn(128,128,2)*0.0
    data_image_test2[i] = data_image_test[i]*(1+noise)  # Not prefect signal(add noise)

# for i in range(data_image_test.shape[0]):
#     noise = np.random.randn(128,128,2)*0.08
#     data_image_test3[i] = data_image_test[i]*(1+noise)  # Not prefect signal(add noise)

# for i in range(data_image_test.shape[0]):
#     noise = np.random.randn(128,128,2)*0.16
#     data_image_test4[i] = data_image_test[i]*(1+noise)  # Not prefect signal(add noise)

# for i in range(data_image_test.shape[0]):
#     noise = np.random.randn(128,128,2)*0.32
#     data_image_test5[i] = data_image_test[i]*(1+noise)  # Not prefect signal(add noise)

print(data_label_test[0])
print(data_label2[34000])
# 适用Resnet的basic_block
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

img = Input(shape=(128, 128, 2), name='input')
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
dense = Dense(8, activation='sigmoid')(avg_pool)  # 最后一个fc层, 10根据图片的类别数而改变

model = Model(img, dense)
# model.summary()
def scheduler(epoch):
    if epoch < 30:
        return 0.0001
    else:
        return 0.00001
    
adam = Adam(0.01)
loss_func = tf.keras.losses.MeanSquaredError()
# error = tf.keras.metrics.MeanSquaredError()
# train_loss = tf.keras.metrics.Mean(name = 'train_loss')
# train_loss = tf.keras.metrics.Mean(name = 'train_loss')
# train_error = tf.keras.metrics.MeanSquaredError(name = 'train_error')
# test_loss = tf.keras.metrics.Mean(name = 'test_loss')
# test_error = tf.keras.metrics.MeanSquaredError(name = 'test_error')
# 当val_acc不增大, 那么会在2个epoch之后减小learning_rate
# EarlyStop = EarlyStopping(monitor = 'loss',
#                         patience = 0, verbose = 1, mode = 'auto')
corr_result = []
learning_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
class Printcor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        loss_test, _ = model.evaluate(data_image_test2, data_label_test)
        output_result = model.predict(data_image_test2)
        output = []
        output_angle1 = []
        output_angle2 = []
        output2 = []
        for i in range(output_result.shape[0]):
            output_temp = np.arcsin((output_result[i][4:8])*2-1)*2
            output.append(np.sqrt(output_result[i][0:4]))
            output2.append(output_result[i][0:4])
            # output_angle1.append(output_temp)
            # output_temp = output_result[i][3:6]*2*math.pi-math.pi
            output_angle1.append(output_temp)
            # output_angle2.append(-output_temp) 
        # output_data = tf.concat([output, output_angle1, output_angle2], axis = 1)
        output_data = tf.concat([output, output_angle1], axis = 1)
        output_data = np.array(output_data)
        output_angle1 = np.array(output_angle1)
        fudu_error = np.mean(np.abs(fudu[34000:35000] - output2))
        angle_error = np.mean(np.abs(angle_list[34000:35000] - output_angle1)/(2*math.pi))
        # print(fudu_error)
        # print(angle_error)
        print(output_data[0])
        print(data_label2[34000])
        # # print(data_image_test2[0]-data_image_test[0])
        print('test_mse =  %f'%loss_test)
        # _, cor_result, _ , _= correction_2(data_label2[34000:35000], output_data, TM01_data, HE21e_data)
        _, cor_result, _, _ = correction_4(data_label2[34000:35000], output_data, TE01_data, TM01_data, HE21e_data, HE21o_data)
        # _, cor_result, _, _ = correction_4(data_label2[34000:35000], output_data, HE11e_data, HE31o_data, HE41o_data, HE42e_data)
        # _, cor_result, _, _ = correction_3(data_label2[34000:35000], output_data, TE01_data, TM01_data, HE21e_data)
        # _, cor_result, _, _ = correction_3(data_label2[34000:35000], output_data, HE11e_data, HE31o_data, HE41o_data)
        # _, cor_result, _, _ = correction_6(data_label2[34000:35000], output_data, HE11e_data, HE31o_data, HE32e_data, HE33o_data, HE41o_data, HE42e_data)
        # _, cor_result, _, _ = correction_8(data_label2[34000:35000], output_data, TE01_data, TE02_data, HE11e_data, HE31o_data, HE32e_data, HE33o_data, HE41o_data, HE42e_data)
        # _, cor_result, _, _ = correction_10(data_label2[34000:35000], output_data, TE01_data, TE02_data, TM03_data, TM04_data, HE11e_data, HE31o_data, HE32e_data, HE33o_data, HE41o_data, HE42e_data)       
        print(cor_result) 
        corr_result.append(cor_result)

model.compile(
    optimizer = adam, 
    loss = loss_func,
    metrics = [Error()]
    )
t5 = time.time()
print(model.fit(data_image_train, data_label_train, validation_data = (data_image_val, data_label_val), epochs = 60, batch_size = 32, callbacks = [learning_scheduler, Printcor()]))
t6 = time.time()
print('train time:', (t6 - t5))
t1 = time.time()
output_result = model.predict(data_image_test2)
t2 = time.time()
print('predicted time:',(t2 - t1))
# output_result3 = model.predict(data_image_test3) # nosie=0.08
# output_result4 = model.predict(data_image_test4) # nosie=0.16
# output_result5 = model.predict(data_image_test5) # nosie=0.32
# t3 = time.time()
output = []
output_angle1 = []
output_angle2 = []
output2 = []
for i in range(output_result.shape[0]):
    output_temp = np.arcsin((output_result[i][4:8])*2-1)*2
    output.append(np.sqrt(output_result[i][0:4]))
    output2.append(output_result[i][0:4])
    # output_angle1.append(output_temp)
    # output_temp = output_result[i][3:6]*2*math.pi-math.pi
    output_angle1.append(output_temp)
    # output_angle2.append(-output_temp) 
    # output_data = tf.concat([output, output_angle1, output_angle2], axis = 1)
    output_data = tf.concat([output, output_angle1], axis = 1)
# t4 = time.time()
# print('select phase and calculate time:', (t4 - t3))
output_data = np.array(output_data)
output_angle1 = np.array(output_angle1)
output2 = np.array(output2)

# TM01_fudu_error = np.mean(np.abs(fudu[34000:35000, 0] - output2[:,0]))
# HE21e_fudu_error = np.mean(np.abs(fudu[34000:35000, 1] - output2[:,1]))
# TM01_angle_error = np.mean(np.abs(angle_list[34000:35000, 0] - output_angle1[:,0])/(2*math.pi))
# HE21e_angle_error = np.mean(np.abs(angle_list[34000:35000, 1] - output_angle1[:,1])/(2*math.pi))

# TE01_fudu_error = np.mean(np.abs(fudu[34000:35000, 0] - output2[:,0]))
# TM01_fudu_error = np.mean(np.abs(fudu[34000:35000, 1] - output2[:,1]))
# HE21e_fudu_error = np.mean(np.abs(fudu[34000:35000, 2] - output2[:,2]))
# TE01_angle_error = np.mean(np.abs(angle_list[34000:35000, 0] - output_angle1[:,0])/(2*math.pi))
# TM01_angle_error = np.mean(np.abs(angle_list[34000:35000, 1] - output_angle1[:,1])/(2*math.pi))
# HE21e_angle_error = np.mean(np.abs(angle_list[34000:35000, 2] - output_angle1[:,2])/(2*math.pi))

# TE01_fudu_error = np.mean(np.abs(fudu[34000:35000, 0] - output2[:,0]))
# TM01_fudu_error = np.mean(np.abs(fudu[34000:35000, 1] - output2[:,1]))
# HE21e_fudu_error = np.mean(np.abs(fudu[34000:35000, 2] - output2[:,2]))
# HE21o_fudu_error = np.mean(np.abs(fudu[34000:35000, 3] - output2[:,3]))
# TE01_angle_error = np.mean(np.abs(angle_list[34000:35000, 0] - output_angle1[:,0])/(2*math.pi))
# TM01_angle_error = np.mean(np.abs(angle_list[34000:35000, 1] - output_angle1[:,1])/(2*math.pi))
# HE21e_angle_error = np.mean(np.abs(angle_list[34000:35000, 2] - output_angle1[:,2])/(2*math.pi))
# HE21o_angle_error = np.mean(np.abs(angle_list[34000:35000, 3] - output_angle1[:,3])/(2*math.pi))

# # HE11e_fudu_error = np.mean(np.abs(fudu[34000:35000, 0] - output2[:,0]))
# # HE31o_fudu_error = np.mean(np.abs(fudu[34000:35000, 1] - output2[:,1]))
# # HE41o_fudu_error = np.mean(np.abs(fudu[34000:35000, 2] - output2[:,2]))
# # HE31o_angle_error = np.mean(np.abs(angle_list[34000:35000, 0] - output_angle1[:,0])/(2*math.pi))
# # HE41o_angle_error = np.mean(np.abs(angle_list[34000:35000, 1] - output_angle1[:,1])/(2*math.pi))

# # HE11e_fudu_error = np.mean(np.abs(fudu[34000:35000, 0] - output2[:,0]))
# # HE31o_fudu_error = np.mean(np.abs(fudu[34000:35000, 1] - output2[:,1]))
# # HE41o_fudu_error = np.mean(np.abs(fudu[34000:35000, 2] - output2[:,2]))
# # HE42e_fudu_error = np.mean(np.abs(fudu[34000:35000, 3] - output2[:,3]))
# # HE31o_angle_error = np.mean(np.abs(angle_list[34000:35000, 0] - output_angle1[:,0])/(2*math.pi))
# # HE41o_angle_error = np.mean(np.abs(angle_list[34000:35000, 1] - output_angle1[:,1])/(2*math.pi))
# # HE42e_angle_error = np.mean(np.abs(angle_list[34000:35000, 2] - output_angle1[:,2])/(2*math.pi))

# # HE11e_fudu_error = np.mean(np.abs(fudu[34000:35000, 0] - output2[:,0]))
# # HE31o_fudu_error = np.mean(np.abs(fudu[34000:35000, 1] - output2[:,1]))
# # HE32e_fudu_error = np.mean(np.abs(fudu[34000:35000, 2] - output2[:,2]))
# # HE33o_fudu_error = np.mean(np.abs(fudu[34000:35000, 3] - output2[:,3]))
# # HE41o_fudu_error = np.mean(np.abs(fudu[34000:35000, 4] - output2[:,4]))
# # HE42e_fudu_error = np.mean(np.abs(fudu[34000:35000, 5] - output2[:,5]))
# # HE31o_angle_error = np.mean(np.abs(angle_list[34000:35000, 0] - output_angle1[:,0])/(2*math.pi))
# # HE32e_angle_error = np.mean(np.abs(angle_list[34000:35000, 1] - output_angle1[:,1])/(2*math.pi))
# # HE33o_angle_error = np.mean(np.abs(angle_list[34000:35000, 2] - output_angle1[:,2])/(2*math.pi))
# # HE41o_angle_error = np.mean(np.abs(angle_list[34000:35000, 3] - output_angle1[:,3])/(2*math.pi))
# # HE42e_angle_error = np.mean(np.abs(angle_list[34000:35000, 4] - output_angle1[:,4])/(2*math.pi))

# # TE01_fudu_error = np.mean(np.abs(fudu[34000:35000, 0] - output2[:,0]))
# # TE02_fudu_error = np.mean(np.abs(fudu[34000:35000, 1] - output2[:,1]))
# # HE11e_fudu_error = np.mean(np.abs(fudu[34000:35000, 2] - output2[:,2]))
# # HE31o_fudu_error = np.mean(np.abs(fudu[34000:35000, 3] - output2[:,3]))
# # HE32e_fudu_error = np.mean(np.abs(fudu[34000:35000, 4] - output2[:,4]))
# # HE33o_fudu_error = np.mean(np.abs(fudu[34000:35000, 5] - output2[:,5]))
# # HE41o_fudu_error = np.mean(np.abs(fudu[34000:35000, 6] - output2[:,6]))
# # HE42e_fudu_error = np.mean(np.abs(fudu[34000:35000, 7] - output2[:,7]))
# # TE01_angle_error = np.mean(np.abs(angle_list[34000:35000, 0] - output_angle1[:,0])/(2*math.pi))
# # TE02_angle_error = np.mean(np.abs(angle_list[34000:35000, 1] - output_angle1[:,1])/(2*math.pi))
# # HE31o_angle_error = np.mean(np.abs(angle_list[34000:35000, 2] - output_angle1[:,2])/(2*math.pi))
# # HE32e_angle_error = np.mean(np.abs(angle_list[34000:35000, 3] - output_angle1[:,3])/(2*math.pi))
# # HE33o_angle_error = np.mean(np.abs(angle_list[34000:35000, 4] - output_angle1[:,4])/(2*math.pi))
# # HE41o_angle_error = np.mean(np.abs(angle_list[34000:35000, 5] - output_angle1[:,5])/(2*math.pi))
# # HE42e_angle_error = np.mean(np.abs(angle_list[34000:35000, 6] - output_angle1[:,6])/(2*math.pi))

# # TE01_fudu_error = np.mean(np.abs(fudu[34000:35000, 0] - output2[:,0]))
# # TE02_fudu_error = np.mean(np.abs(fudu[34000:35000, 1] - output2[:,1]))
# # TM03_fudu_error = np.mean(np.abs(fudu[34000:35000, 2] - output2[:,2]))
# # TM04_fudu_error = np.mean(np.abs(fudu[34000:35000, 3] - output2[:,3]))
# # HE11e_fudu_error = np.mean(np.abs(fudu[34000:35000, 4] - output2[:,4]))
# # HE31o_fudu_error = np.mean(np.abs(fudu[34000:35000, 5] - output2[:,5]))
# # HE32e_fudu_error = np.mean(np.abs(fudu[34000:35000, 6] - output2[:,6]))
# # HE33o_fudu_error = np.mean(np.abs(fudu[34000:35000, 7] - output2[:,7]))
# # HE41o_fudu_error = np.mean(np.abs(fudu[34000:35000, 8] - output2[:,8]))
# # HE42e_fudu_error = np.mean(np.abs(fudu[34000:35000, 9] - output2[:,9]))
# # TE01_angle_error = np.mean(np.abs(angle_list[34000:35000, 0] - output_angle1[:,0])/(2*math.pi))
# # TE02_angle_error = np.mean(np.abs(angle_list[34000:35000, 1] - output_angle1[:,1])/(2*math.pi))
# # TM03_angle_error = np.mean(np.abs(angle_list[34000:35000, 2] - output_angle1[:,2])/(2*math.pi))
# # TM04_angle_error = np.mean(np.abs(angle_list[34000:35000, 3] - output_angle1[:,3])/(2*math.pi))
# # HE31o_angle_error = np.mean(np.abs(angle_list[34000:35000, 4] - output_angle1[:,4])/(2*math.pi))
# # HE32e_angle_error = np.mean(np.abs(angle_list[34000:35000, 5] - output_angle1[:,5])/(2*math.pi))
# # HE33o_angle_error = np.mean(np.abs(angle_list[34000:35000, 6] - output_angle1[:,6])/(2*math.pi))
# # HE41o_angle_error = np.mean(np.abs(angle_list[34000:35000, 7] - output_angle1[:,7])/(2*math.pi))
# # HE42e_angle_error = np.mean(np.abs(angle_list[34000:35000, 8] - output_angle1[:,8])/(2*math.pi))

# fudu_error = np.mean(np.abs(fudu[34000:35000] - output2))
# angle_error = np.mean(np.abs(angle_list[34000:35000] - output_angle1)/(2*math.pi))

# print(TE01_fudu_error)
# # print(TE02_fudu_error)
# print(TM01_fudu_error)
# # print(TM03_fudu_error)
# # print(TM04_fudu_error)
# # print(HE11e_fudu_error)
# print(HE21e_fudu_error)
# print(HE21o_fudu_error)
# # print(HE31o_fudu_error)
# # print(HE32e_fudu_error)
# # print(HE33o_fudu_error)
# # print(HE41o_fudu_error)
# # print(HE42e_fudu_error)

# print(TE01_angle_error)
# # print(TE02_angle_error)
# print(TM01_angle_error)
# # print(TM03_angle_error)
# # print(TM04_angle_error)
# # print(HE12o_angle_error)
# print(HE21e_angle_error)
# print(HE21o_angle_error)
# # print(HE31o_angle_error)
# # print(HE32e_angle_error)
# # print(HE33o_angle_error)
# # print(HE41o_angle_error)
# # print(HE42e_angle_error)

# print(fudu_error)
# print(angle_error)

# add noise output
# print("noise begin")
# print("noise weight = 0.08")
# output_data3 = notjianbing_4(fudu, angle_list, output_result3)
# _, cor_result3, _, _ = correction_3(data_label2[34000:35000], output_data3, HE11e_data, HE31o_data, HE41o_data)
# print(cor_result3)
# print("noise weight = 0.16")
# output_data4 = notjianbing_4(fudu, angle_list, output_result4)
# _, cor_result4, _, _ = correction_3(data_label2[34000:35000], output_data4, HE11e_data, HE31o_data, HE41o_data)
# print(cor_result4)
# print("noise weight = 0.32")
# output_data5 = notjianbing_4(fudu, angle_list, output_result5)
# _, cor_result5, _, _ = correction_3(data_label2[34000:35000], output_data5, HE11e_data, HE31o_data, HE41o_data)
# print(cor_result5)
# print('noise end')
# orignal_data, cor_result, restore_data, Y2 = correction_2(data_label2[34000:35000], output_data, TM01_data, HE21e_data)
orignal_data, cor_result, restore_data, Y2 = correction_4(data_label2[34000:35000], output_data, TE01_data, TM01_data, HE21e_data, HE21o_data)
# orignal_data, cor_result, restore_data, Y2 = correction_4(data_label2[34000:35000], output_data, HE11e_data, HE31o_data, HE41o_data, HE42e_data)
# orignal_data, cor_result, restore_data, Y2 = correction_3(data_label2[34000:35000], output_data, TE01_data, TM01_data, HE21e_data)
# orignal_data, cor_result, restore_data, Y2 = correction_3(data_label2[34000:35000], output_data, HE11e_data, HE31o_data, HE41o_data)
# orignal_data, cor_result, restore_data, Y2 = correction_6(data_label2[34000:35000], output_data, HE11e_data, HE31o_data,  HE32e_data, HE33o_data, HE41o_data, HE42e_data)
# orignal_data, cor_result, restore_data, Y2 = correction_8(data_label2[34000:35000], output_data, TE01_data, TE02_data, HE11e_data, HE31o_data,  HE32e_data, HE33o_data, HE41o_data, HE42e_data)
# orignal_data, cor_result, restore_data, Y2 = correction_10(data_label2[34000:35000], output_data, TE01_data, TE02_data, TM03_data, TM04_data, HE11e_data, HE31o_data, HE32e_data, HE33o_data, HE41o_data, HE42e_data)
print(cor_result)

temppp = []      
for i in range(1000):
    phase_temp2 = data_label2[34000+i]
    phase = phase_temp2[4:8]
    phase2_temp2 = Y2[i]
    phase22 = phase2_temp2[4:8]
    # print(phase,phase2)
    if((np.sign(phase)==np.sign(phase22)).all()):
        # print(phase,phase2)
        # if(phase_temp2[3]<0.75):
        if(phase_temp2[2]<0.75) and (phase_temp2[6]<0.6*math.pi):
            temppp.append(i)
        # print(i)

# print(data_label2[34000:34010])
# print(Y2[0:10])
# paint modal_weights and phase_values
print(np.max(np.angle(orignal_data[0])))
fig = plt.figure(figsize=(50,40))
gs = gridspec.GridSpec(6,6)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[0,3])
ax5 = plt.subplot(gs[0,4])
ax6 = plt.subplot(gs[0,5])
ax7 = plt.subplot(gs[1,0])
ax8 = plt.subplot(gs[1,1])
ax9 = plt.subplot(gs[1,2])
ax10 = plt.subplot(gs[1,3])
ax11 = plt.subplot(gs[1,4])
ax12 = plt.subplot(gs[1,5])
ax13 = plt.subplot(gs[2:4,0:2])
ax14 = plt.subplot(gs[2:4,2:4])
ax15 = plt.subplot(gs[2:4,4:6])
ax16 = plt.subplot(gs[4:6,0:2])
ax17 = plt.subplot(gs[4:6,2:4])
ax18 = plt.subplot(gs[4:6,4:6])
ax = np.array([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12])
# for jianbing modal(4 modal)
# x1 = to_one(np.abs(orignal_data[temppp[110]]))
# x2 = to_one(np.abs(restore_data[temppp[110]]))
# x3 = np.angle(orignal_data[temppp[110]])
# x4 = np.angle(restore_data[temppp[110]])
# x5 = to_one(np.abs(orignal_data[temppp[61]]))
# x6 = to_one(np.abs(restore_data[temppp[61]]))
# x7 = np.angle(orignal_data[temppp[61]])
# x8 = np.angle(restore_data[temppp[61]])
# x9 = to_one(np.abs(orignal_data[temppp[64]]))
# x10 = to_one(np.abs(restore_data[temppp[64]]))
# x11 = np.angle(orignal_data[temppp[64]])
# x12 = np.angle(restore_data[temppp[64]])

x1 = to_one(np.abs(orignal_data[temppp[0]]))
x2 = to_one(np.abs(restore_data[temppp[0]]))
x3 = np.angle(orignal_data[temppp[0]])
x4 = np.angle(restore_data[temppp[0]])
x5 = to_one(np.abs(orignal_data[temppp[5]]))
x6 = to_one(np.abs(restore_data[temppp[5]]))
x7 = np.angle(orignal_data[temppp[5]])
x8 = np.angle(restore_data[temppp[5]])
x9 = to_one(np.abs(orignal_data[temppp[3]]))
x10 = to_one(np.abs(restore_data[temppp[3]]))
x11 = np.angle(orignal_data[temppp[3]])
x12 = np.angle(restore_data[temppp[3]])
temp = np.array([x1, x2, x5, x6, x9, x10, x3, x4, x7, x8, x11, x12])
norm1 = matplotlib.colors.Normalize(vmin = 0.0, vmax = 1.0)
norm2 = matplotlib.colors.Normalize(vmin = -math.pi, vmax = math.pi)
cmap = plt.cm.jet
for i in range(2):
    img = temp[i]
    im = ax[i].imshow(img, norm = norm1, cmap = cmap)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
ax[0].set_ylabel('Intensity',fontsize=52)
clb = plt.colorbar(im, ticks = [0.0, 0.5, 1.0])
clb.set_ticklabels([0.0, 0.5, 1.0],fontsize = 52)
ax[0].text(5,-5,'(a)',fontsize=65)
ax[0].set_title('Actual',fontsize=52)
ax[1].set_title('Reconstructed',fontsize=52)
plt.tight_layout()
for i in range(2,4):
    img = temp[i]
    im = ax[i].imshow(img, norm = norm1, cmap = cmap)
    # ax[i].axis('off')
clb = plt.colorbar(im, ticks = [0.0, 0.5, 1.0])
clb.set_ticklabels([0.0, 0.5, 1.0],fontsize = 52)
ax[2].set_title('Actual',fontsize=52)
ax[3].set_title('Reconstructed',fontsize=52)
ax[2].text(5,-5,'(b)',fontsize=52)
for i in range(4,6):
    img = temp[i]
    im = ax[i].imshow(img, norm = norm1, cmap = cmap)
    # ax[i].axis('off')
clb = plt.colorbar(im, ticks = [0.0, 0.5, 1.0])
clb.set_ticklabels([0.0, 0.5, 1.0],fontsize = 52)
ax[4].set_title('Actual',fontsize=52)
ax[5].set_title('Reconstructed',fontsize=52)
ax[4].text(5,-5,'(c)',fontsize=52)
plt.tight_layout()
for i in range(6,8):
    img = temp[i]
    im = ax[i].imshow(img, norm = norm2, cmap = cmap)
    # ax[i].set_xticks([])
    # ax[i].set_yticks([])
ax[6].set_ylabel('Phase',fontsize=52)
clb = plt.colorbar(im, ticks = [-math.pi, 0.0, math.pi])
clb.set_ticklabels([r'$-\pi$',0,r'$\pi$'], fontsize = 52)
plt.tight_layout()
for i in range(8,10):
    img = temp[i]
    im = ax[i].imshow(img, norm = norm2, cmap = cmap)
    # ax[i].axis('off')
clb = plt.colorbar(im, ticks = [-math.pi, 0.0, math.pi])
clb.set_ticklabels([r'$-\pi$',0,r'$\pi$'], fontsize = 52)
plt.tight_layout()
for i in range(10,12):
    img = temp[i]
    im = ax[i].imshow(img, norm = norm2, cmap = cmap)
    # ax[i].axis('off')
clb = plt.colorbar(im, ticks = [-math.pi, 0.0, math.pi])
clb.set_ticklabels([r'$-\pi$',0,r'$\pi$'], fontsize = 52)
plt.tight_layout()
# # zhu zhuang tu
# actual_1 = data_label2[34000+temppp[110]][0:4]
# actual_2 = data_label2[34000+temppp[61]][0:4]
# actual_3 = data_label2[34000+temppp[64]][0:4]
# actual_4 = data_label2[34000+temppp[110]][4:8]
# actual_5 = data_label2[34000+temppp[61]][4:8]
# actual_6 = data_label2[34000+temppp[64]][4:8]

actual_1 = data_label2[34000+temppp[0]][0:4]
actual_2 = data_label2[34000+temppp[5]][0:4]
actual_3 = data_label2[34000+temppp[3]][0:4]
actual_4 = data_label2[34000+temppp[0]][4:8]
actual_5 = data_label2[34000+temppp[5]][4:8]
actual_6 = data_label2[34000+temppp[3]][4:8]

# actual_11 = data_label2[34000+temppp[60]][0:4]
# actual_22 = data_label2[34000+temppp[61]][0:4]
# actual_33 = data_label2[34000+temppp[62]][0:4]
# actual_111 = data_label2[34000+temppp[63]][0:4]
# actual_222 = data_label2[34000+temppp[64]][0:4]
# actual_333 = data_label2[34000+temppp[65]][0:4]
# actual_44 = data_label2[34000+temppp[60]][4:8]
# actual_55 = data_label2[34000+temppp[61]][4:8]
# actual_66 = data_label2[34000+temppp[62]][4:8]
# actual_444 = data_label2[34000+temppp[63]][4:8]
# actual_555 = data_label2[34000+temppp[64]][4:8]
# actual_666 = data_label2[34000+temppp[65]][4:8]

actual_value = np.array([actual_1, actual_2, actual_3, actual_4, actual_5, actual_6])
# # zhu zhuang tu
# predicted_1 = Y2[temppp[110]][0:4]
# predicted_2 = Y2[temppp[61]][0:4]
# predicted_3 = Y2[temppp[64]][0:4]
# predicted_4 = Y2[temppp[110]][4:8]
# predicted_5 = Y2[temppp[61]][4:8]
# predicted_6 = Y2[temppp[64]][4:8]


predicted_1 = Y2[temppp[0]][0:4]
predicted_2 = Y2[temppp[5]][0:4]
predicted_3 = Y2[temppp[3]][0:4]
predicted_4 = Y2[temppp[0]][4:8]
predicted_5 = Y2[temppp[5]][4:8]
predicted_6 = Y2[temppp[3]][4:8]
# # predicted_11 = Y2[temppp[60]][0:4]
# # predicted_22 = Y2[temppp[61]][0:4]
# # predicted_33 = Y2[temppp[62]][0:4]
# # predicted_111 = Y2[temppp[63]][0:4]
# # predicted_222 = Y2[temppp[64]][0:4]
# # predicted_333 = Y2[temppp[65]][0:4]
# # predicted_44 = Y2[temppp[60]][4:8]
# # predicted_55 = Y2[temppp[61]][4:8]
# # predicted_66 = Y2[temppp[62]][4:8]
# # predicted_444 = Y2[temppp[63]][4:8]
# # predicted_555 = Y2[temppp[64]][4:8]
# # predicted_666 = Y2[temppp[65]][4:8]
# # print(np.abs(actual_11-predicted_11), np.abs(actual_44-predicted_44))
# # print(np.abs(actual_22-predicted_22), np.abs(actual_55-predicted_55))
# # print(np.abs(actual_33-predicted_33), np.abs(actual_66-predicted_66))
# # print(np.abs(actual_111-predicted_111), np.abs(actual_444-predicted_444))
# # print(np.abs(actual_222-predicted_222), np.abs(actual_555-predicted_555))
# # print(np.abs(actual_333-predicted_333), np.abs(actual_666-predicted_666))

predicted_value = np.array([predicted_1, predicted_2, predicted_3, predicted_4, predicted_5, predicted_6])
X1 = np.arange(4)
X2 = np.arange(4)
bar_width = 0.35
# tick_label_amp = ["TM01","HE21e"]
# tick_label_amp = ["TE01","TM01","HE21e"]
tick_label_amp = ["TE01","TM01","HE21e","HE21o"]
# tick_label_amp = ["HE11e","HE31o","HE41o"]
# tick_label_amp = ["HE11e","HE31o","HE41o","HE42e"]
# tick_label_amp = ["HE11e","HE31o","HE32e","HE33o","HE41o","HE42e"]
# tick_label_amp = ["TE01","TE02","HE11e","HE31o","HE32e","HE33o","HE41o","HE42e"]
# tick_label_amp = ["TE01","TE02","TM03","TM04","HE11e","HE31o","HE32e","HE33o","HE41o","HE42e"]

# tick_label_pha = ["TM01","HE21e"]
# tick_label_pha = ["TE01","TM01","HE21e"]
tick_label_pha = ["TE01","TM01","HE21e","HE21o"]
# tick_label_pha = ["HE31o","HE41o"]
# tick_label_pha = ["HE31o","HE41o","HE42e"]
# tick_label_pha = ["HE31o","HE32e","HE33o","HE41o","HE42e"]
# tick_label_pha = ["TE01","TE02","HE31o","HE32e","HE33o","HE41o","HE42e"]
# tick_label_pha = ["TE01","TE02","TM03","TM04","HE31o","HE32e","HE33o","HE41o","HE42e"]
axx = np.array([ax13,ax14,ax15,ax16,ax17,ax18])
for i in range(3):
    axx[i].bar(X1,actual_value[i],bar_width,color="black",align="center")
    axx[i].bar(X1+bar_width,predicted_value[i],bar_width,color="red",align="center")
    axx[i].set_ylabel("Modal weigths", fontsize = 52)
    axx[i].set_xticks(X1+bar_width/2,tick_label_amp)
    axx[i].set_xticklabels(tick_label_amp,fontsize=52)
    axx[i].set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    axx[i].set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize = 52)
    axx[i].legend(["Actual", "Predicted"], fontsize = 52)
    # ax2[i].tight_layout()
axx[0].text(-0.253,0.95,'(a1)',fontsize=52)
axx[1].text(-0.253,0.95,'(b1)',fontsize=52)
axx[2].text(-0.253,0.95,'(c1)',fontsize=52)
for i in range(3,6):
    axx[i].bar(X2,actual_value[i],bar_width,color="black",align="center")
    axx[i].bar(X2+bar_width,predicted_value[i],bar_width,color="red",align="center")
    axx[i].set_ylabel("Modal phases", fontsize = 52)
    axx[i].set_xticks(X2+bar_width/2,tick_label_pha)
    axx[i].set_xticklabels(tick_label_pha,fontsize=52)
    axx[i].set_yticks([-math.pi,0,math.pi])
    # ax2[i].set_yticks([0,2*math.pi])
    axx[i].set_yticklabels([r'$-\pi$',0,r'$\pi$'],fontsize=52)
    # ax2[i].set_yticklabels([0,r'$2\pi$'],fontsize=20)
    axx[i].legend(["Actual", "Predicted"], fontsize =52)
axx[3].text(-0.25,math.pi-0.35,'(a2)',fontsize=52)
axx[4].text(-0.25,math.pi-0.35,'(b2)',fontsize=52)
axx[5].text(-0.25,math.pi-0.35,'(c3)',fontsize=52)
plt.tight_layout()
plt.savefig('/tf/lijianjun/桌面/test/mode5/TM01/modal_weights.svg',bbox_inches = 'tight')
# plt.savefig('/tf/lijianjun/桌面/test/mode10/HE11e/modal_weights.svg',bbox_inches = 'tight')

# paint cor line
X = np.arange(60)
print(corr_result)
plt.figure()
plt.plot(X, np.array(corr_result), color = 'red', linestyle = '-', marker = '.')
plt.text(0,0.99,'(a)',fontsize=13)
plt.xlabel('Epochs',fontsize=13)
plt.ylabel('Correlation',fontsize=13)
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],fontsize=13)
# plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],fontsize=13)
plt.yticks([0.80,0.82,0.84,0.86,0.88,0.90],fontsize=13)
# plt.yticks([0.80, 0.84, 0.88,0.92,0.96,1.00],fontsize=13)
plt.grid(linewidth = 0.1)
plt.savefig('/tf/lijianjun/桌面/test/mode5/TM01/cor_result_2.svg')
# plt.savefig('/tf/lijianjun/桌面/test/mode10/HE11e/cor_result_2.svg')

corr_temp = []
for i in range(1000):
    # print(i, np.round(corr2(np.abs(restore_data[i])**2, np.abs(orignal_data[i])**2), 4))
    # For Two modal
#   if((np.round(corr2(np.abs(restore_data[i])**2, np.abs(orignal_data[i])**2), 4)<0.9983) & (np.round(corr2(np.abs(restore_data[i])**2, np.abs(orignal_data[i])**2), 4)>0.985)):
#       corr_temp.append(i)
    #For three modal
    if((np.round(corr2(np.abs(restore_data[i])**2, np.abs(orignal_data[i])**2), 4)<(cor_result+0.002)) & (np.round(corr2(np.abs(restore_data[i])**2, np.abs(orignal_data[i])**2), 4)>(cor_result-0.005))):
      corr_temp.append(i)
# print(corr_temp)
fig, ax = plt.subplots(3, 6)
ax = ax.flatten()
x1 = to_one(np.abs(orignal_data[corr_temp[0]]))
x2 = to_one(np.abs(restore_data[corr_temp[0]]))
x3 = to_one(np.abs(orignal_data[corr_temp[1]]))
x4 = to_one(np.abs(restore_data[corr_temp[1]]))
x5 = to_one(np.abs(orignal_data[corr_temp[2]]))
x6 = to_one(np.abs(restore_data[corr_temp[2]]))
x7 = to_one(np.abs(orignal_data[corr_temp[3]]))
x8 = to_one(np.abs(restore_data[corr_temp[3]]))
x9 = to_one(np.abs(orignal_data[corr_temp[4]]))
x10 = to_one(np.abs(restore_data[corr_temp[4]]))
x11 = to_one(np.abs(orignal_data[corr_temp[5]]))
x12 = to_one(np.abs(restore_data[corr_temp[5]]))
temp = np.array([x1, x3, x5, x7, x9, x11, x2, x4, x6, x8, x10, x12, np.abs(x1 - x2), np.abs(x3 - x4), np.abs(x5 - x6), np.abs(x7 - x8), np.abs(x9 - x10), np.abs(x11 - x12)])

# x111 = to_one(np.abs(orignal_data[6]))
# x22 = to_one(np.abs(restore_data[6]))
# x33 = to_one(np.abs(orignal_data[7]))
# x44 = to_one(np.abs(restore_data[7]))
# x55 = to_one(np.abs(orignal_data[8]))
# x66 = to_one(np.abs(restore_data[8]))
# x77 = to_one(np.abs(orignal_data[9]))
# x88 = to_one(np.abs(restore_data[9]))
# x99 = to_one(np.abs(orignal_data[10]))
# x1010 = to_one(np.abs(restore_data[10]))
# x1111 = to_one(np.abs(orignal_data[11]))
# x1212 = to_one(np.abs(restore_data[11]))
# temp2 = np.array([x11, x33, x55, x77, x99, x1111, x22, x44, x66, x88, x1010, x1212, np.abs(x11 - x22), np.abs(x33 - x44), np.abs(x55 - x66), np.abs(x77 - x88), np.abs(x99 - x1010), np.abs(x1111 - x1212)])
norm = matplotlib.colors.Normalize(vmin = 0.0, vmax = 1.0)
print(np.mean(np.abs(x1 - x2)), np.max(np.abs(x1 - x2)))
print(np.mean(np.abs(x3 - x4)), np.max(np.abs(x3 - x4)))
print(np.mean(np.abs(x5 - x6)), np.max(np.abs(x5 - x6)))
print(np.mean(np.abs(x7 - x8)), np.max(np.abs(x7 - x8)))
print(np.mean(np.abs(x9 - x10)), np.max(np.abs(x9 - x10)))
print(np.mean(np.abs(x11 - x12)), np.max(np.abs(x11 - x12)))
cmap = plt.cm.jet
for i in range(18):
    img = temp[i]
    im = ax[i].imshow(img, norm = norm, cmap = cmap)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
# ax[12].set_xlabel(np.round(np.max(np.abs(x1 - x2)),2))
# ax[13].set_xlabel(np.round(np.max(np.abs(x3 - x4)),2))
# ax[14].set_xlabel(np.round(np.max(np.abs(x5 - x6)),2))
# ax[15].set_xlabel(np.round(np.max(np.abs(x7 - x8)),2))
# ax[16].set_xlabel(np.round(np.max(np.abs(x9 - x10)),2))
# ax[17].set_xlabel(np.round(np.max(np.abs(x11 - x12)),2))
ax[0].text(-105,80,'Actual',fontsize=10)
ax[0].text(-55,30,'(a)',fontsize=10)
ax[6].text(-228,80,'Reconstructed',fontsize=10)
ax[6].text(-55,30,'(b)',fontsize=10)
ax[12].text(-135,80,'Residual',fontsize=10)
ax[12].text(-55,30,'(c)',fontsize=10)
ax[12].text(-175,165,'Correlation',fontsize=10)
ax[12].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[0]])**2, np.abs(orignal_data[corr_temp[0]])**2), 4),fontsize=10)
ax[13].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[1]])**2, np.abs(orignal_data[corr_temp[1]])**2), 4),fontsize=10)
ax[14].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[2]])**2, np.abs(orignal_data[corr_temp[2]])**2), 4),fontsize=10)
ax[15].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[3]])**2, np.abs(orignal_data[corr_temp[3]])**2), 4),fontsize=10)
ax[16].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[4]])**2, np.abs(orignal_data[corr_temp[4]])**2), 4),fontsize=10)
ax[17].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[5]])**2, np.abs(orignal_data[corr_temp[5]])**2), 4),fontsize=10)
plt.colorbar(im, ax =[ax[i] for i in range(18)], ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.savefig('/tf/lijianjun/桌面/test/mode5/TM01/34000_34004_2.svg')
# plt.savefig('/tf/lijianjun/桌面/test/mode10/HE11e/34000_34004_2.svg',bbox_inches = 'tight')


# plt.figure()
# plt.grid(linewidth=0.1)
# mode_num = np.array([3,4,6,8,10])
# cor_value = np.array([0.9938,0.9928,0.9885,0.9838,0.9784])
# plt.plot(mode_num, cor_value, color = 'red', linestyle = '-', marker = '.')
# plt.xticks([2,3,4,5,6,7,8,9,10],fontsize=13)
# plt.yticks([0.96,0.97,0.98,0.99,1.00],fontsize=13)
# plt.xlabel('Mode number',fontsize=13)
# plt.ylabel('Correlation',fontsize=13)
# plt.savefig('/tf/lijianjun/桌面/test/cor_mode.svg')