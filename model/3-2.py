import tensorflow as tf
from spatial_transformer import transformer
from mytransformer import spatial_transformer_network
import numpy as np
# from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import matplotlib.pylab as plt
from datetime import datetime
from tqdm import trange
import os

nGlimpse = 4



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
initial = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0]])
initial = initial.astype('float32')
initial = initial.flatten()


def weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1)
    # initial = tf.random_normal(shape, mean= 0, stddev= 0.1)
    # initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1) * tf.sqrt(1.0/shape[0])
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1)
    # initial = tf.random_normal(shape, mean= 0, stddev= 0.1)
    # initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1) * tf.sqrt(1.0/shape[0])
    return tf.Variable(initial)



param = 12
# Create variables for fully connected layer for the localisation network
W_fc_loc1 = tf.Variable(initial_value=tf.zeros([128, 50]), name='W_fc_loc1')
b_fc_loc1 = tf.Variable(initial_value=tf.zeros([50]), name='b_fc_loc1')
W_fc_loc2 = tf.Variable(initial_value=tf.zeros([50, param]), name='W_fc_loc2')
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')






filter_size = 3
n_filters_1 = 32
n_filters_2 = 16


W_conv1 = weight_variable([filter_size, filter_size, filter_size, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
W_conv2 = weight_variable([filter_size, filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
keep_prob = tf.placeholder(tf.float32)

# Weight matrix is [height x width x input_channels x output_channels]

# h_trans_ext = tf.reshape(h_trans, (-1, 64,64,1))
#
# h_conv1 = tf.layers.conv2d(h_trans_ext, 16, 3, strides=2, padding='same', activation= tf.nn.relu)
# h_conv2 = tf.layers.conv2d(h_conv1, 16, 3, strides=2, padding='same', activation= tf.nn.relu)

# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(num_units=128) for _ in range(2)])

inputs = [0] * nGlimpse
param_list = [0] * nGlimpse
h0 = multi_cell.zero_state(batch_size, np.float32)

h11_list = [0] * nGlimpse
out_h = [0] * nGlimpse

for i in range(nGlimpse):

    h_fc_loc1 = tf.nn.tanh(tf.matmul(h0[1], W_fc_loc1) + b_fc_loc1)
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
    h_fc_loc2 = tf.nn.leaky_relu(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)


    # h_trans = transformer(x_tensor, h_fc_loc2, out_size)
    h_trans = spatial_transformer_network(x, h_fc_loc2, [mri_height, mri_width, mri_channel])

    trans_matrix = tf.reshape(h_fc_loc2, (-1, 3, 4))
    param_list[i] = trans_matrix
    inputs[i] = h_trans
    h_trans_ext = tf.reshape(h_trans, (-1, mri_height, mri_width, mri_channel,1))

    h_conv1 = tf.nn.relu(tf.nn.conv3d(input=h_trans_ext, filter=W_conv1, strides=[1, 2, 2, 2, 1], padding='SAME') + b_conv1)
    h_conv1 = tf.nn.max_pool3d(h_conv1,[1,2,2,2,1],[1,2,2,2,1],padding="SAME")
    h_conv2 = tf.nn.relu(tf.nn.conv3d(input=h_conv1, filter=W_conv2, strides=[1, 2, 2, 2, 1], padding='SAME') + b_conv2)
    h_conv2 = tf.nn.max_pool3d(h_conv2,[1,2,2,2,1],[1,2,2,2,1], padding="SAME")
    # if i == 0:
    #     h_conv1 = tf.layers.conv2d(h_trans_ext, 16, 3, strides=1, padding='same', activation= tf.nn.relu, reuse=None, name='conv1')
    #     h_conv2 = tf.layers.conv2d(h_conv1, 16, 3, strides=1, padding='same', activation= tf.nn.relu, reuse= None, name='conv2')
    # else:
    #     h_conv1 = tf.layers.conv2d(h_trans_ext, 16, 3, strides=1, padding='same', activation= tf.nn.relu, reuse=True, name='conv1')
    #     h_conv2 = tf.layers.conv2d(h_conv1, 16, 3, strides=1, padding='same', activation= tf.nn.relu, reuse= True, name='conv2')
    #
    # print(tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel'))
    # print(tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2/kernel'))
    h_conv2_flat = tf.reshape(h_conv2, (batch_size, -1))
    print(h_conv2.name)

    output, h1 = multi_cell.call(h_conv2_flat, h0)

    out_h[i] = output
    h11_list[i] = h1[0]

    h0 = h1

h_flat = tf.layers.flatten(h0[0])

inputs = tf.concat(axis=0, values=inputs)
inputs = tf.reshape(inputs, (nGlimpse, batch_size , mri_height, mri_width, mri_channel))
inputs = tf.transpose(inputs, [1, 0, 2, 3, 4])

param_list = tf.concat(axis=0, values=param_list)
param_list = tf.reshape(param_list, (nGlimpse, batch_size , 3, 4))
param_list = tf.transpose(param_list, [1, 0, 2, 3])

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
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])


predictions = tf.argmax(y_logits, 1)
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

results_fmt = '[{:%H:%M:%S}] EPOCH {}/{} LOSS {:.3f}, DEV: {:.3f}'

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# We'll now train in minibatches and report accuracy, loss:

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
print('the mean is %.5f, the std is %.5f' % (np.mean(test_acc_list), np.std(test_acc_list) / np.sqrt(cross_validation_times)))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
test_index = np.random.choice(mri_patient, batch_size, replace=False)
batch_samples = np.random.choice(test_index, batch_size)
test_images = []
for id in range(len(batch_samples)):
    image_temp = np.load(data_path + str(batch_samples[id]) + '.npy')
    test_images.append(image_temp)
test_images = np.stack(test_images)

batch_xs = test_images[:, :, :, :, np.newaxis]
batch_ys = no_hot_labels[batch_samples]
batch_ys = sess.run(tf.one_hot(batch_ys, 2))

batch_xs_orig = batch_xs


tranlated_img = sess.run(inputs, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

np.save('L_2C_3D_original_mri_images', np.reshape(batch_xs_orig, (-1,  mri_height, mri_width, mri_channel)))
np.save('L_2C_3D_translated_mri_images', tranlated_img)

# test_acc, test_predictions = sess.run([accuracy, predictions], feed_dict={x: X_test, y: Y_test, keep_prob: 1.0})
