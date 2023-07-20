# 电影文本评论
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, Sequential, preprocessing, regularizers, Model
from tensorflow.keras.layers import Flatten, Dense, Embedding, Dropout

max_words = 10000  # 词汇量
max_len = 20  # 在评论中第20个单词处截断
embedding_dim = 16  # 嵌入层输出维度
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_words)
print(x_test.shape)
# 统计序列长度，将数据集转换成形状为（samples，maxlen）的二维整数张量
# maxlen=20 取后20个单词
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen = max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen = max_len)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(32)

# # 下面几行是将数值转换为实际的评论
# # word_index is a dictionary mapping words to an integer index
# word_index = imdb.get_word_index()
# # We reverse it, mapping integer indices to words
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# # We decode the review; note that our indices were offset by 3
# # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])  # 第一条实际评论

# 构建模型
# 特征单词数 max_words = 10000
# 后20个单词 max_len = 20
# 嵌入维度 embedding_dim = 8
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = Embedding(max_words, embedding_dim, input_length = max_len)
        self.flatten = Flatten()
        self.d1 = Dense(16, activation='relu', kernel_regularizer = regularizers.l2(0.001))
        self.drop1 = Dropout(0.2)
        self.d2 = Dense(1, activation='sigmoid', kernel_regularizer = regularizers.l2(0.001))
        self.drop2 = Dropout(0.2)
    def call(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.drop1(x)
        x = self.d2(x)
        return self.drop2(x)
model = MyModel()
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits = False)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name = 'test_accuracy')
@tf.function
def train_step(words, labels):
    with tf.GradientTape() as tape:
        preds = model(words, training = True)
        loss = loss_func(labels, preds)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, preds)
@tf.function
def test_step(words, labels):
    preds = model(words, training = False)
    loss = loss_func(labels, preds)
    test_loss(loss)
    test_accuracy(labels, preds) 
EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    for words, labels in train_ds:
        train_step(words, labels)
    for words, labels in test_ds:
        test_step(words, labels)
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )




