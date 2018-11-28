from __future__ import print_function
import tensorflow as tf
import pickle as pk
from numpy import array
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.misc
import matplotlib
import os
from tqdm import trange
from sklearn.metrics import roc_auc_score
# from dicoms_to_pixels import dicoms_to_3D_pixel_lists

print('%x' %sys.maxsize,sys.maxsize > 2 ** 32)



# path = '/media/king/DATA/Web/TOTAL/ABIDE-I-II-individuals-roted/'
path = './Individuals_0.25/'
accuracyFolderName = 'MRI_hos_2DCNN_0.25/'
ckptFolderName = "MRI_web_2DCNN_chckPts/"
if os.path.isdir(ckptFolderName) == False:
    os.makedirs(ckptFolderName)


list = os.listdir(path) # dir is your directory path
number_files = len(list)


def compute_accuracy(index):

    batches_in_epoch = len(index) // batch_size
    acc_eval = 0
    auc_eval = 0
    sen_eval = 0
    spe_eval = 0

    np.random.shuffle(index)

    p_y_value_list = []

    for i in range(batches_in_epoch):

        index_temp = index[i * batch_size: (i + 1) * batch_size]

        test_images = []
        for id in range(len(index_temp)):
            image_temp = np.load(path + str(index_temp[id]) + '.npy')
            # image_temp = (image_temp - np.max(image_temp)) / (np.max(image_temp) - np.min(image_temp))
            test_images.append(image_temp)
        test_images = np.stack(test_images)

        nextX = test_images
        nextY = no_hot_labels[index_temp]

        feed_dict = {xs: nextX, tf_is_training: False}
        p_y_value = sess.run(p_y, feed_dict=feed_dict)

        predicted_labels_value = np.argmax(p_y_value, axis= 1)
        correct_labels_value = nextY

        acc_eval += np.mean(np.equal(predicted_labels_value,correct_labels_value))

        p_y_value_list.append(p_y_value[:,1])
        auc_eval += roc_auc_score(correct_labels_value, p_y_value[:, 1])

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for j in range(batch_size):
            if predicted_labels_value[j] == 1 and correct_labels_value[j] == 1:
                TP += 1
            elif predicted_labels_value[j] == 0 and correct_labels_value[j] == 0:
                TN += 1
            elif predicted_labels_value[j] == 1 and correct_labels_value[j] == 0:
                FP += 1
            elif predicted_labels_value[j] == 0 and correct_labels_value[j] == 1:
                FN += 1

        TPR = TP / float(TP + FN)
        TNR = TN / float(TN + FP)
        FPR = FP / float(FP + TN)
        FNR = FN / float(TP + FN)
        sen_eval += TPR
        spe_eval += TNR

    ACC = acc_eval / batches_in_epoch

    AUC = auc_eval / batches_in_epoch

    SEN = sen_eval / batches_in_epoch

    SPE = spe_eval / batches_in_epoch

    return ACC, AUC, SEN, SPE



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    # initial = tf.random_uniform(shape, minval=-1, maxval=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W, stride = [1, 1, 1, 1, 1]):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv3d(x, W, strides= stride, padding='SAME')


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('param'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('param_stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


no_hot_labels = np.load('Hos_label.npy')

# no_hot_labels = np.load('ABIDE-I-II_label_sorted.npy')

n_class = 2
cross_validation_times = 30
mri_patient = number_files
max_iters = 5000
batch_size = int(mri_patient * 0.2)
SMALL_NUM = 1e-10

mri_info = np.load(path + '1.npy')

first_layer = 128
second_layer = 128
third_layer = 64


mri_depth, mri_height, mri_width = mri_info.shape
print(mri_info.shape)

xs = tf.placeholder(tf.float32,[None, mri_depth, mri_height, mri_width])
# ys = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, n_class])

tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing


print(xs.shape)



conv1 = tf.layers.conv2d(xs, first_layer, 3, name='conv1', padding= 'same', activation= tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
batch1 = tf.contrib.layers.batch_norm(pool1, center=True, scale=True, is_training=tf_is_training)
print(batch1.shape)
conv2 = tf.layers.conv2d(batch1, second_layer, 3, name='conv2', padding= 'same', activation= tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
batch2 = tf.contrib.layers.batch_norm(pool2, center=True, scale=True, is_training=tf_is_training)
print(conv2.shape)
conv3 = tf.layers.conv2d(batch2, third_layer, 3, name='conv3', padding= 'same', activation= tf.nn.leaky_relu)
pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
batch3 = tf.contrib.layers.batch_norm(pool3, center=True, scale=True, is_training=tf_is_training)
print(batch3.shape)

# gap = tf.layers.average_pooling3d(conv2, [conv2.shape[1], conv2.shape[2], conv2.shape[3]], strides=[1,1,1])
#
# gap = tf.reshape(gap,[-1, mri_channel * (2 ** 3)])

flat = tf.layers.flatten(batch3)
flat_drop = tf.layers.dropout(flat, rate= 0.2, training= tf_is_training)
print(flat_drop.shape)

fc1 = tf.layers.dense(flat_drop, 1024)
fc1_drop = tf.layers.dropout(fc1, rate= 0.2, training= tf_is_training)

fc2 = tf.layers.dense(fc1_drop, n_class)

p_y = tf.nn.softmax(fc2)

# loss = tf.losses.softmax_cross_entropy(onehot_labels= ys , logits= prediction)    # loss

loss = - tf.log(p_y + SMALL_NUM) * (ys)

cost = tf.reduce_mean(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cost)




for exp in trange(cross_validation_times):


    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    labels = sess.run(tf.one_hot(no_hot_labels, 2))

    full_index = np.arange(mri_patient)
    test_choice = batch_size
    test_index = np.random.choice(mri_patient, test_choice, replace=False)
    train_index = np.setdiff1d(full_index, test_index)

    test_baseline = 1 - np.mean(np.argmax(labels[test_index], axis=1))

    # print(test_baseline)
    print('\nCross Validation Start: Experiment ' + str(exp) + ' ' + 'Max_step = ' + str(max_iters) + ' Patients_num = ' + str(number_files))
    print('Test index are :')
    print(test_index)
    print('The test accuracy for the zero baseline is %.5f' % test_baseline)

    test_acc = []
    test_auc = []
    test_sen = []
    test_spe = []

    for i in trange(max_iters):

        i = i + 1

        batch_samples = np.random.choice(train_index, batch_size)

        train_images = []
        for id in range(len(batch_samples)):
            image_temp = np.load(path  + str(batch_samples[id]) + '.npy')
            # print(path  + str(batch_samples[id]) + '.npy')
            # image_temp = image_temp - np.min(image_temp) / (np.max(image_temp) - np.min(image_temp))
            # image_temp = (image_temp - np.mean(image_temp)) / np.std(image_temp)
            # print(np.std(image_temp))
            train_images.append(image_temp)
        train_images = np.stack(train_images)

        batch_xs = train_images
        batch_ys = labels[batch_samples]

        _, cost_value = sess.run([train_step, cost], feed_dict={xs: batch_xs, ys: batch_ys, tf_is_training:True})

        # print('experiment %d, step %d, training loss %.5f' % (exp, i, cost_value))

        if i % 500 == 0:
             train_accuracy,_,_,_ = compute_accuracy(train_index)
             print('\n#####################################')
             print('exp %d step %d, total training accuracy %.5f' %(exp, i, train_accuracy))
             print('#####################################')

        if i % 1000 == 0:

            acc, auc, sen, spe = compute_accuracy(test_index)
            print('\n##########################################################')
            print('exp %d step %d, total testing ACC %.5f AUC %.5f SEN %.5f SPE %.5f' % (exp, i, acc, auc, sen, spe))
            print('##########################################################')

    test_acc.append(acc)
    test_auc.append(auc)
    test_sen.append(sen)
    test_spe.append(spe)

    sess.close()

    path_cv = accuracyFolderName

    if os.path.isdir(path_cv):
        pass
    else:
        os.mkdir(path_cv)

    np.save(path_cv + 'test_acc_experiment_' + str(exp), test_acc)

    np.save(path_cv + 'test_auc_experiment_' + str(exp), test_auc)

    np.save(path_cv + 'test_sen_experiment_' + str(exp), test_sen)

    np.save(path_cv + 'test_spe_experiment_' + str(exp), test_spe)





