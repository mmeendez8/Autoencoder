import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# Network parameters
latent_dim = 10
reshaped_dim = [-1, 7, 7, 1]
inputs_decoder = 49
batch_size = 128
epochs = 100

test_image_n = 5

# Get data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Create tf dataset
with tf.variable_scope("DataPipe"):
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.map(lambda x: tf.image.convert_image_dtype([x], dtype=tf.float32))
    dataset = dataset.batch(batch_size=batch_size).prefetch(batch_size)

    iterator = dataset.make_initializable_iterator()
    input_batch = iterator.get_next()
    input_batch = tf.reshape(input_batch, shape=[-1, 28, 28, 1])


def encoder(X):
    activation = tf.nn.relu
    with tf.variable_scope("Encoder"):
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.flatten(x)

        # Local latent variables
        mean_ = tf.layers.dense(x, units=latent_dim, name='mean')
        std_dev = tf.nn.softplus(tf.layers.dense(x, units=latent_dim), name='std_dev')  # softplus to force >0

        # Reparametrization trick
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], latent_dim]), name='epsilon')
        z = mean_ + tf.multiply(epsilon, std_dev)

        return z, mean_, std_dev


def decoder(z):
    activation = tf.nn.relu
    with tf.variable_scope("Decoder"):
        x = tf.layers.dense(z, units=inputs_decoder, activation=activation)
        x = tf.layers.dense(x, units=inputs_decoder, activation=activation)
        x = tf.reshape(x, reshaped_dim)

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28, 1])

        return img


z, mean_, std_dev = encoder(input_batch)
output = decoder(z)


# TRAINING
flat_output = tf.reshape(output, [-1, 28*28])
flat_input = tf.reshape(input_batch, [-1, 28*28])

with tf.name_scope('loss'):
    img_loss = tf.reduce_sum(tf.squared_difference(flat_output, flat_input))
    latent_loss = 0.5 * tf.reduce_sum(tf.square(mean_) + tf.square(std_dev) - tf.log(tf.square(std_dev)) - 1, 1)
    loss = tf.reduce_mean(img_loss + latent_loss)
    tf.summary.scalar('batch_loss', loss)


optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

merged_summary_op = tf.summary.merge_all()

init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
gpu_options = tf.GPUOptions(allow_growth=True)


# Training loop
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init_vars)
    writer = tf.summary.FileWriter('./logs', sess.graph)

    for epoch in range(epochs):
        sess.run(iterator.initializer)
        print(epoch)
        flag = True  # Show only first image of epoch
        while True:
            try:
                sess.run(optimizer)
                if not epoch % 1 and flag:
                    summ, target, output_ = sess.run([merged_summary_op, input_batch, output])
                    fig = plt.figure()
                    for j in range(test_image_n):
                        for pos, im in enumerate([target, output_]):
                            fig.add_subplot(test_image_n, 2, (2*j+pos)+1)
                            plt.imshow(im[j].reshape((28, 28)), cmap='gray')
                    plt.savefig('Results/Train/Epoch_{}'.format(epoch))

                    flag = False
                    writer.add_summary(summ, epoch)

                    # Create random image
                    artificial_image = sess.run(output, feed_dict={z: np.random.normal(0, 1, (1, latent_dim))})
                    plt.figure()
                    plt.imshow(artificial_image[0].reshape((28, 28)), cmap='gray')
                    plt.savefig('Results/Test/{}'.format(epoch))

            except tf.errors.OutOfRangeError:
                break

