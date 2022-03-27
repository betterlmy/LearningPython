import tensorflow as tf
import matplotlib.pyplot as plt

# 下载MNIST数据集
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)

# 处理数据集，将数据集以二维表形式训练
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)

# 归一化处理
x_train = tf.cast(x_train, tf.float32) / 255
x_test = tf.cast(x_test, tf.float32) / 255

# 设定输入层、隐藏层和输出层的单元数
input_size = 784
hidden_size = 32
output_size = 784

input = tf.keras.layers.Input(shape=(input_size,))
# Encoder
en = tf.keras.layers.Dense(hidden_size, activation='relu')(input)
# Decoder
de = tf.keras.layers.Dense(output_size, activation='sigmoid')(en)
model = tf.keras.Model(inputs=input, outputs=de)
model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train,
          epochs=50,
          batch_size=256,
          shuffle=True,
          validation_data=(x_test, x_test))

encode = tf.keras.Model(inputs=input, outputs=en)
input_de = tf.keras.layers.Input(shape=(hidden_size,))
output = model.layers[-1](input_de)
decode = tf.keras.Model(inputs=input_de, outputs=output)
encode_test = encode(x_test)
decode_test = decode.predict(encode_test)
x_test = x_test.numpy()
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # 展示原始图像
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    # 展示自编码器重构后的图像
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decode_test[i].reshape(28, 28))
plt.show()
