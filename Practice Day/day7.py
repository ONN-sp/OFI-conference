import tensorflow as tf
import numpy as np
import math
# x = [[2]]
# m = tf.matmul(x, x)
# print(m)

# a = tf.constant([1, 2,
#                 3, 4])
# print(a)
# print(tf.add(a, 1))

# w = tf.constant([1.0])
# with tf.GradientTape() as tape:
#     tape.watch(w)
#     loss = w * w
# grad = tape.gradient(loss, w)
# print(grad)

# a = tf.keras.metrics.Mean()
# print(a([1, 3]))

# sparse_tensor = tf.sparse.SparseTensor(indices = [[0, 0], [1, 2]],
#                                        values = [1, 2],
#                                        dense_shape = [3, 4])
# print(sparse_tensor)
# print(tf.sparse.to_dense(sparse_tensor))

# b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# print(b)
# a = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
# print(a)

# @tf.function
# def f(x):
#     print("The function is running in Python")
#     print(x)
# f(1.0)
# f(1)

# a = tf.ragged.constant([[1], []])
# print(tf.add(a, 1))

# t1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])
# print(tf.slice(t1,
#                begin=[1],
#                size=[3]).numpy())
# print(t1[1:4].numpy())

# a = tf.constant([5])
# b = tf.constant([[1], [3]])
# values = tf.constant([2, 4])
# print(tf.scatter_nd(indices = b, updates = values, shape = a))

# a = tf.cast(tf.constant([0, 1, 2, 4, 5]), 'float32')
# b = tf.cast(tf.constant([1, 2, 3, 4, 6]), 'float32')
# c = tf.abs(a - b)
# c - tf.cast(c, 'float32')
# print(tf.reduce_mean(c))

t1 = np.array([1, 1, 1, 1, 1, 1, 1, 1]).reshape(2, 4, 1)
t2 = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(2, 4, 1)
t3 = np.empty((2,4,1))
for i in range(t1.shape[0]):
    noise = np.random.randn(4,1)*0.3
    t3[i] = t1[i]*(1 + noise)
print(t3[0])
print(t3[1])