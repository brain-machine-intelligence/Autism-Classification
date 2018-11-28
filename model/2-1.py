import tensorflow as tf
from spatial_transformer import transformer
from transformer import spatial_transformer_network
# from mytransformer import spatial_transformer_network
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import matplotlib.pylab as plt
from datetime import datetime
from tqdm import trange
import os

nGlimpse = 4

SMALL_NUM = 1e-6

data_path = './Individuals_0.25/'
data_path_list = os.listdir(data_path)
accuracyFolderName = 'MRI_hos_stn_0.25/'

images_info = np.load(data_path + '1.npy')
no_hot_labels = np.load('Hos_label.npy')


mri_patient = len(data_path_list)
number_files = mri_patient
mri_height, mri_width, mri_channel = images_info.shape

batch_size = int(mri_patient * 0.2)


tf.set_random_seed(1)
x = tf.placeholder(tf.float32, [None, mri_height, mri_width, mri_channel, 1])
y = tf.placeholder(tf.float32, [None, 2])

# Identity transformation
initial = np.array([[1., 0, 0], [0, 1., 0]])
initial = initial.astype('float32')
initial = initial.flatten()

param = 6
# Create variables for fully connected layer for the localisation network

# def weight_variable(shape):
#     initial = tf.random_uniform(shape, minval=0, maxval = 0.1)
#     # initial = tf.random_normal(shape, mean= 0, stddev= 0.1)
#     # initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1) * tf.sqrt(1.0/shape[0])
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.random_uniform(shape, minval=0, maxval = 0.1)
#     # initial = tf.random_normal(shape, mean= 0, stddev= 0.1)
#     # initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1) * tf.sqrt(1.0/shape[0])
#     return tf.Variable(initial)


keep_prob = tf.placeholder(tf.float32)
x_total = []
for i in range(mri_height):
    x_flat = tf.reshape(x[:,i,:,:], (-1, mri_width * mri_channel))

    W_fc_loc1 = tf.Variable(initial_value=tf.zeros([mri_width * mri_channel, 50]))
    b_fc_loc1 = tf.Variable(initial_value=tf.zeros([50]))
    W_fc_loc2 = tf.Variable(initial_value=tf.zeros([50, param]))
    b_fc_loc2 = tf.Variable(initial_value=initial)

    h_fc_loc1 = tf.nn.tanh(tf.matmul(x_flat, W_fc_loc1) + b_fc_loc1)

    # Dropout for regularization

    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

    # Second layer
    h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

    h_trans = spatial_transformer_network(x[:,i,:,:], h_fc_loc2)
    x_total.append(h_trans)

x_total = tf.stack(x_total, axis=1)

h_trans_ext = tf.reshape(x_total, (-1, mri_height, mri_width, mri_channel))

h_conv1 = tf.layers.conv2d(h_trans_ext, 16, 3, strides=1, padding='same', activation=tf.nn.relu)
h_conv2 = tf.layers.conv2d(h_conv1, 16, 3, strides=1, padding='same', activation=tf.nn.relu)


h_flat = tf.layers.flatten(h_conv2)

# Create the fully-connected layer
n_fc = 1024
# W_fc1 = weight_variable([16 * 16 * n_filters_2, n_fc])
# b_fc1 = bias_variable([n_fc])
# h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
h_fc1 = tf.nn.relu(tf.layers.dense(h_flat, n_fc))
n_class = 2
# Add additional regularization
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# W_fc2 = weight_variable([n_fc, n_class])
# b_fc2 = bias_variable([n_class])
# y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_logits = tf.layers.dense(h_fc1_drop, n_class)

scores = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y)


# Loss
cross_entropy = tf.reduce_mean(scores)

# Optimizer
lr = 1e-4
opt = tf.train.AdamOptimizer(lr)

# Minimize
optimizer = opt.minimize(cross_entropy)

# Gradients
# grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])


predictions = tf.argmax(y_logits, 1)
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

results_fmt = '[{:%H:%M:%S}] EPOCH {}/{} LOSS {:.3f}, DEV: {:.3f}'

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# We'll now train in minibatches and report accuracy, loss:

iter_per_epoch = 1000
n_epochs = 100 # Feel free to increase the epochs for better results

results_fmt = '[{:%H:%M:%S}] EPOCH {}, ITERATION {}/{}, LOSS {:.3f}'




def compute_accuracy(index):
    batches_in_epoch = len(index) // batch_size

    total_acc = 0

    for iter_i in range(batches_in_epoch):

        batch_samples = np.random.choice(index, batch_size)
        images = []
        for id in range(len(batch_samples)):
            image_temp = np.load(data_path + str(batch_samples[id]) + '.npy')
            image_temp = image_temp - np.min(image_temp)
            image_temp = image_temp / (np.max(image_temp) + SMALL_NUM)
            images.append(image_temp)
        images = np.stack(images)

        batch_xs = images[:, :, :, :, np.newaxis]
        batch_ys = no_hot_labels[batch_samples]

        batch_ys = sess.run(tf.one_hot(batch_ys, 2))

        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        total_acc = total_acc + acc


    return total_acc/batches_in_epoch



test_acc_list = []
cross_validation_times = 10
for exp in trange(cross_validation_times):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    full_index = np.arange(mri_patient)
    test_choice = batch_size
    test_index = np.random.choice(mri_patient, test_choice, replace=False)
    train_index = np.setdiff1d(full_index, test_index)

    train_batches_in_epoch = len(train_index) // batch_size

    for epoch_i in trange(n_epochs):
        for iter_i in range(train_batches_in_epoch - 1):

            batch_samples = np.random.choice(train_index, batch_size)

            train_images = []
            for id in range(len(batch_samples)):
                image_temp = np.load(data_path + str(batch_samples[id]) + '.npy')
                image_temp = image_temp - np.min(image_temp)
                image_temp = image_temp / (np.max(image_temp) + SMALL_NUM)
                train_images.append(image_temp)
            train_images = np.stack(train_images)

            batch_xs = train_images[:, :, :, :, np.newaxis]
            batch_ys = no_hot_labels[batch_samples]

            batch_ys = sess.run(tf.one_hot(batch_ys, 2))

            if iter_i % 2 == 0:
                loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                print(results_fmt.format(datetime.now(), epoch_i, iter_i, train_batches_in_epoch, loss))
                # print(param_list_value[0])

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})

        # val_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid, keep_prob: 1.0})

        # print('--------> {}/{} EPOCHS, VAL ACC: {:.3f}'.format(epoch_i + 1, n_epochs, val_acc))

        train_acc = compute_accuracy(train_index)
        print('train accuracy is %.5f'%train_acc)
        test_acc = compute_accuracy(test_index)
        print('test accuracy is %.5f'%test_acc)
    test_acc = compute_accuracy(test_index)
    sess.close()
    test_acc_list.append(test_acc)
print('the mean is %.5f, the std is %.5f' % (np.mean(test_acc_list), np.std(test_acc_list)/np.sqrt(cross_validation_times)))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
test_index = np.random.choice(mri_patient, batch_size, replace=False)
batch_samples = np.random.choice(test_index, batch_size)
test_images = []
for id in range(len(batch_samples)):
    image_temp = np.load(data_path + str(batch_samples[id]) + '.npy')
    image_temp = image_temp - np.min(image_temp)
    image_temp = image_temp / (np.max(image_temp) + SMALL_NUM)
    test_images.append(image_temp)
test_images = np.stack(test_images)

batch_xs = test_images[:, :, :, :, np.newaxis]
batch_ys = no_hot_labels[batch_samples]
batch_ys = sess.run(tf.one_hot(batch_ys, 2))

batch_xs_orig = batch_xs


tranlated_img = sess.run([h_trans_ext], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

# print(h_fc_loc2_mat_value)
np.save('2C_2D_original_mri_images', np.reshape(batch_xs_orig, (-1,  mri_height, mri_width, mri_channel)))
np.save('2C_2D_translated_mri_images', tranlated_img)

# test_acc, test_predictions = sess.run([accuracy, predictions], feed_dict={x: X_test, y: Y_test, keep_prob: 1.0})
