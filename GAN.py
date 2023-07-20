import tensorflow as tf
from tensorflow.keras import datasets, Model, regularizers
from tensorflow.keras.layers import Conv2DTranspose, Reshape, GlobalMaxPooling2D, Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()   # train_images = (50000, 32, 32, 3)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.reshape(-1, 28, 28, 1) / 255.0, test_images.reshape(-1, 28, 28, 1) / 255.0
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(10000).batch(32)
# Hyparameters
IDEAS = 128
EPOCHS = 5

class GAN_G(Model):
    def __init__(self):
        super(GAN_G, self).__init__()
        self.d1 = Dense(input_shape = (IDEAS, ), units = 7 * 7 * 128)
        self.leakyrelu1 = LeakyReLU(alpha = 0.2)
        self.reshape = Reshape((7, 7, 128))
        self.convt1 = Conv2DTranspose(128, (4, 4), strides = 2, padding = 'same')
        self.leakyrelu2 = LeakyReLU(alpha = 0.2)
        self.convt2 = Conv2DTranspose(128, (4, 4), strides = 2, padding = 'same')
        self.leakyrelu3 = LeakyReLU(alpha = 0.2)
        self.conv1 = Conv2D(1, (7, 7), padding = 'same', activation = 'sigmoid')
    def call(self, x):
        x = self.d1(x)
        x = self.leakyrelu1(x)
        x = self.reshape(x)
        x = self.convt1(x)
        x = self.leakyrelu2(x)
        x = self.convt2(x)
        x = self.leakyrelu3(x)
        return self.conv1(x)
class GAN_D(Model):
    def __init__(self):
        super(GAN_D, self).__init__()
        self.conv1 = Conv2D(input_shape = (28, 28, 1), filters = 64, kernel_size = 3, strides = 2, padding = 'same')
        self.leakyrelu1 = LeakyReLU(alpha = 0.2)
        self.conv2 = Conv2D(128, 3, 2, padding = 'same')
        self.leakyrelu2 = LeakyReLU(alpha = 0.2)
        self.pool = GlobalMaxPooling2D()
        self.d1 = Dense(1)
    def call(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.pool(x)
        return self.d1(x)
model_generator = GAN_G()
model_discriminator = GAN_D()

g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits = True)
g_train_acc = tf.keras.metrics.BinaryAccuracy()
g_test_acc = tf.keras.metrics.BinaryAccuracy()
d_train_acc_true = tf.keras.metrics.BinaryAccuracy()
d_train_acc_fake = tf.keras.metrics.BinaryAccuracy()
d_test_acc_fake = tf.keras.metrics.BinaryAccuracy()
d_test_acc_true = tf.keras.metrics.BinaryAccuracy()

@tf.function
def train_step(images):
    noise = tf.random.normal(shape = (images.shape[0], IDEAS))
    generator_images = model_generator(noise)
    true_labels = tf.ones((images.shape[0], 1))
    fake_labels = tf.zeros((images.shape[0], 1))
    # 鉴别器训练
    with tf.GradientTape() as tape:
        pred_true = model_discriminator(images)
        pred_fake = model_discriminator(generator_images)
        d_loss_true = loss_func(true_labels, pred_true)
        d_loss_fake = loss_func(fake_labels, pred_fake)
        d_loss = 0.5 * d_loss_true + 0.5 * d_loss_fake
    gradients = tape.gradient(d_loss, model_discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(gradients, model_discriminator.trainable_variables))
    d_train_acc_true(true_labels, pred_true)
    d_train_acc_fake(fake_labels, pred_fake)
    # 生成器训练
    with tf.GradientTape() as tape:
        preds = model_discriminator(model_generator(noise))
        g_loss = loss_func(true_labels, preds)
    gradients = tape.gradient(g_loss, model_generator.trainable_variables)  # 不更新鉴别器的权重
    g_optimizer.apply_gradients(zip(gradients, model_generator.trainable_variables))
    g_train_acc(true_labels, preds)
    return d_loss, g_loss
@tf.function
def test_step(images):
    noise = tf.random.normal(shape = (images.shape[0], IDEAS))
    true_labels = tf.ones((images.shape[0], 1))
    fake_labels = tf.zeros((images.shape[0], 1))
    preds = model_discriminator(images)
    preds_fake = model_discriminator(model_generator(noise))
    d_true_test_loss = loss_func(true_labels, preds)
    d_fake_test_loss = loss_func(fake_labels, preds_fake)
    d_test_loss = 0.5 * d_true_test_loss + 0.5 * d_fake_test_loss
    d_test_acc_true(true_labels, preds)
    d_test_acc_fake(fake_labels, preds_fake)
    g_test_loss = loss_func(true_labels, preds_fake)
    g_test_acc(true_labels, preds_fake)
    return d_test_loss, g_test_loss

for epoch in range(EPOCHS):
    g_train_acc.reset_states()
    g_test_acc.reset_states()
    d_train_acc_true.reset_states()
    d_train_acc_fake.reset_states()
    d_test_acc_fake.reset_states()
    d_test_acc_true.reset_states()
    for x_train, y_train in train_ds:
        d_loss, g_loss = train_step(x_train)
    for x_test, y_test in test_ds:
        d_test_loss, g_test_loss = test_step(x_test)
    print(
        f'Epoch {epoch + 1}, '
        f'd_train_loss: {d_loss}, '
        f'g_train_loss: {g_loss}, '
        f'd_test_loss: {d_test_loss}, '
        f'g_test_loss: {g_test_loss}, '
        f'g_train_acc: {g_train_acc.result()},'
        f'g_test_acc: {g_test_acc.result()}, '
        f'd_train_acc: {d_train_acc_true.result() * 0.5 + d_train_acc_fake.result() * 0.5},'
        f'd_test_acc: {d_test_acc_true.result() * 0.5 + d_test_acc_fake.result() * 0.5},'
    )