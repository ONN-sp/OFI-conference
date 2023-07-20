# 电影文本积极、消极评论的分类
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, GRU
import matplotlib.pyplot as plt
from tensorflow.keras import Model

num_words = 30000
max_len = 200
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words = num_words)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, max_len, padding='post')
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, max_len, padding='post')

class lstm_model(Model):
    def __init__(self):
        super(lstm_model, self).__init__()
        self.embd = Embedding(input_dim=30000, output_dim=32, input_length=max_len)
        self.lstm1 = LSTM(32, return_sequences=True)  # return_sequences为true,则输出所有时间状态的值,否则只输出最后一个状态的值
        self.lstm2 = LSTM(1, activation='sigmoid', return_sequences=False)
    def call(self, x):
        x = self.embd(x)
        x = self.lstm1(x)
        return self.lstm2(x)
class gru_model(Model):
    def __init__(self):
        super(gru_model, self).__init__()
        self.embd = Embedding(input_dim=30000, output_dim=32, input_length=max_len)
        self.gru1 = GRU(32, return_sequences=True)  # return_sequences为true,则输出所有时间状态的值,否则只输出最后一个状态的值
        self.gru2 = GRU(1, activation='sigmoid', return_sequences=False)
    def call(self, x):
        x = self.embd(x)
        x = self.gru1(x)
        return self.gru2(x)

# model = lstm_model()
model = gru_model()
# model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['trainning', 'valivation'])
plt.show()
