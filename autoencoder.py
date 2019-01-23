import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import seaborn as sns


# Network parameters
tf.flags.DEFINE_float('learning_rate', .0005, 'Initial learning rate.')
tf.flags.DEFINE_integer('epochs', 100, 'Number of steps to run trainer.')
tf.flags.DEFINE_integer('batch_size', 128, 'Minibatch size')
tf.flags.DEFINE_integer('latent_dim', 10, 'Number of latent dimensions')
tf.flags.DEFINE_integer('test_image_number', 5, 'Number of test images to recover during training')
tf.flags.DEFINE_integer('inputs_decoder', 49, 'Size of decoder input layer')
tf.flags.DEFINE_string('dataset', 'mnist', 'Dataset name [mnist, fashion-mnist]')
tf.flags.DEFINE_string('logdir', './logs', 'Logs folder')

FLAGS = tf.flags.FLAGS

# Define and create results folders
results_folder = os.path.join('Results', FLAGS.dataset)
[os.makedirs(os.path.join(results_folder, folder)) for folder in ['Test', 'Train']
    if not os.path.exists(os.path.join(results_folder, folder))]

# Empty log folder
try:
    if not len(os.listdir(FLAGS.logdir)) == 0:
        shutil.rmtree(FLAGS.logdir)
except:
    pass

# Get data
data = keras.datasets.mnist if FLAGS.dataset == 'mnist' else keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Create tf dataset
with tf.variable_scope("DataPipe"):
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.map(lambda x: tf.image.convert_image_dtype([x], dtype=tf.float32))
    dataset = dataset.batch(batch_size=FLAGS.batch_size).prefetch(FLAGS.batch_size)

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
        mean_ = tf.layers.dense(x, units=FLAGS.latent_dim, name='mean')
        std_dev = tf.nn.softplus(tf.layers.dense(x, units=FLAGS.latent_dim), name='std_dev')  # softplus to force >0

        # Reparametrization trick
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], FLAGS.latent_dim]), name='epsilon')
        z = mean_ + tf.multiply(epsilon, std_dev)

        return z, mean_, std_dev


def decoder(z, training=False):
    activation = tf.nn.relu
    with tf.variable_scope("Decoder"):
        x = tf.layers.dense(z, units=FLAGS.inputs_decoder, activation=activation)
        x = tf.layers.dense(x, units=FLAGS.inputs_decoder, activation=activation)
        recovered_size = int(np.sqrt(FLAGS.inputs_decoder))
        x = tf.reshape(x, [-1, recovered_size, recovered_size, 1])

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=None)

        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28, 1])
        return img

# Link encoder and decoder
z, mean_, std_dev = encoder(input_batch)
output = decoder(z)

# Reshape input and output to flat vectors
flat_output = tf.reshape(output, [-1, 28 * 28])
flat_input = tf.reshape(input_batch, [-1, 28 * 28])

with tf.name_scope('loss'):
    # img_loss = tf.reduce_sum(tf.squared_difference(flat_output, flat_input))
    img_loss = tf.reduce_sum(flat_input * -tf.log(flat_output) + (1 - flat_input) * -tf.log(1 - flat_output), 1)
    # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    # img_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_input, logits=flat_output),1)
    latent_loss = 0.5 * tf.reduce_sum(tf.square(mean_) + tf.square(std_dev) - tf.log(tf.square(std_dev)) - 1, 1)
    loss = tf.reduce_mean(img_loss + latent_loss)
    tf.summary.scalar('batch_loss', loss)

optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)


init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
gpu_options = tf.GPUOptions(allow_growth=True)

# Training loop
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    writer = tf.summary.FileWriter('./logs', sess.graph)
    sess.run(init_vars)
    merged_summary_op = tf.summary.merge_all()

    for epoch in range(FLAGS.epochs):
        sess.run(iterator.initializer)
        print(epoch)
        flag = True  # Show only first image of epoch
        while True:
            try:
                sess.run(optimizer)
                if not epoch % 1 and flag:
                    summ, target, output_, ilos = sess.run([merged_summary_op, input_batch, output, img_loss])
                    print(ilos)
                    f, axarr = plt.subplots(FLAGS.test_image_number, 2)
                    for j in range(FLAGS.test_image_number):
                        for pos, im in enumerate([target, output_]):
                            axarr[j, pos].imshow(im[j].reshape((28, 28)), cmap='gray')
                            axarr[j, pos].axis('off')
                    plt.savefig(os.path.join(results_folder, 'Train/Epoch_{}').format(epoch))
                    plt.close(f)
                    flag = False
                    writer.add_summary(summ, epoch)

                    # Create random image
                    artificial_image = sess.run(output, feed_dict={z: np.random.normal(0, 1, (1, FLAGS.latent_dim))})
                    plt.figure()
                    plt.imshow(artificial_image[0].reshape((28, 28)), cmap='gray')
                    plt.savefig(os.path.join(results_folder, 'Test/{}'.format(epoch)))
                    plt.close()
            except tf.errors.OutOfRangeError:
                break
