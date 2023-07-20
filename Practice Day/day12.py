#词嵌入(其实是一个文本分类, 积极和消极)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, BatchNormalization

(train_x, train_y), (test_x, text_y) = keras.datasets.imdb.load_data(num_words=10000)
word_index = tf.keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = {v:k for k, v in word_index.items()}
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
max_len = 500
train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, value=word_index['<PAD>'],
                                                        padding='post', maxlen=max_len)
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, value=word_index['<PAD>'],
                                                        padding='post', maxlen=max_len) 
embedding_dim = 16
class word_Embedding(Model):
    def __init__(self):
        super(word_Embedding, self).__init__()
        self.embed = Embedding(10000, embedding_dim, input_length=max_len)
        self.pool = GlobalAveragePooling1D()
        self.d1 = Dense(16, activation='relu')                                                       
        self.d2 = Dense(1, activation='sigmoid')  
    def call(self, x):
        x = self.embed(x)
        x = self.pool(x)
        x = self.d1(x)
        return self.d2(x)
model =  word_Embedding()
model.compile(optimizer=keras.optimizers.Adam(),
               loss=keras.losses.BinaryCrossentropy(),
               metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=30, batch_size=64, validation_split=0.1) 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

