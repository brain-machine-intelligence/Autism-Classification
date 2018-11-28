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
import cv2

from scipy.io import savemat
# from dicoms_to_pixels import dicoms_to_3D_pixel_lists

print('%x' %sys.maxsize,sys.maxsize > 2 ** 32)


path = '/media/king/DATA/Hos/Individuals_0.25/'
accuracyFolderName = 'MRI_hos_2DCNN/'

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

        nextX = test_images[:,:,:,:,np.newaxis]
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



def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


no_hot_labels = np.load('Hos_label.npy')

# no_hot_labels = np.load('ABIDE-I-II_label_sorted.npy')

n_class = 2
cross_validation_times = 1
mri_patient = number_files
max_iters = 15
batch_size = int(mri_patient * 0.2)
SMALL_NUM = 1e-10

mri_info = np.load(path + '1.npy')

first_layer = 64
second_layer = 64
third_layer = 32


mri_depth, mri_height, mri_width = mri_info.shape

xs = tf.placeholder(tf.float32,[None, mri_depth, mri_height, mri_width, 1])
# ys = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, n_class])

tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing


# print(xs.shape)


gap_list = []
batch_list = []
w_gap_list = []

for i in trange(mri_depth):

    conv1 = tf.layers.conv2d(xs[:,i,:,:,:], first_layer, 3, name=('conv1_' + str(i)), padding= 'same', activation= tf.nn.leaky_relu)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name=('pool1_' + str(i)))
    batch1 = tf.contrib.layers.batch_norm(pool1, center=True, scale=True, is_training=tf_is_training)
    # print(batch1.shape)
    conv2 = tf.layers.conv2d(batch1, second_layer, 3, name=('conv2_' + str(i)), padding= 'same', activation= tf.nn.leaky_relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name=('pool2_' + str(i)))
    batch2 = tf.contrib.layers.batch_norm(pool2, center=True, scale=True, is_training=tf_is_training)
    # print(batch2.shape)
    conv3 = tf.layers.conv2d(batch2, third_layer, 3, name=('conv3_' + str(i)), padding= 'same', activation= tf.nn.leaky_relu)
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name=('pool3_' + str(i)))
    batch3 = tf.contrib.layers.batch_norm(pool3, center=True, scale=True, is_training=tf_is_training)
    # print(batch3.shape)

    batch_list.append(batch3)


    gap = tf.layers.average_pooling2d(batch3, [batch3.shape[1], batch3.shape[2]], strides=[1,1], name='gap_' + str(i))

    gap_drop = tf.layers.dropout(gap, rate=0.2, training=tf_is_training)

    gap_drop = tf.layers.dense(gap_drop, 1, name = ('gap_individual_' + str(i)))

    with tf.variable_scope(('gap_individual_' + str(i)), reuse=True):
        w_gap_ind = tf.get_variable('kernel')
        w_gap_list.append(w_gap_ind)

    gap_list.append(gap_drop)

gap_list = tf.stack(gap_list, axis=1)

gap_squeeze = tf.reshape(gap_list, [-1, mri_depth])

fc1_drop = tf.layers.dropout(gap_squeeze, rate= 0.2, training= tf_is_training)

fc2 = tf.layers.dense(fc1_drop, n_class, name = 'gap_total')

with tf.variable_scope('gap_total', reuse=True):
    w_gap_total = tf.get_variable('kernel')

p_y = tf.nn.softmax(fc2)

# loss = tf.losses.softmax_cross_entropy(onehot_labels= ys , logits= prediction)    # loss

loss = - tf.log(p_y + SMALL_NUM) * (ys)

cost = tf.reduce_mean(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cost)




for exp in range(cross_validation_times):


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

        batch_xs = train_images[:,:,:,:,np.newaxis]
        batch_ys = labels[batch_samples]

        _, cost_value = sess.run([train_step, cost], feed_dict={xs: batch_xs, ys: batch_ys, tf_is_training:True})


        if i % 500 == 0:
             train_accuracy,_,_,_ = compute_accuracy(train_index)
             print('\nexp %d, step %d, total training accuracy %.5f' %(exp, i, train_accuracy))

        if i % 1000 == 0:

            acc, auc, sen, spe = compute_accuracy(test_index)
            print('\nexp %d, step %d, testing accuracy %.5f' % (exp, i, acc))

    for p in trange(mri_patient):

        mri = np.load(path + str(p) +'.npy')

        img = mri[np.newaxis, :, :, :]
        img = img[:,:,:,:,np.newaxis]


        result = sess.run([batch_list, w_gap_list, w_gap_total], feed_dict={xs:img, tf_is_training:False})

        cam_feature_ind, w_gap_ind_value, w_gap_total_value = result

        cam_3d = np.zeros((mri_depth, mri_height, mri_width))

        for d in range(mri_depth):

            cam_feature = np.squeeze(cam_feature_ind[d])

            cam_imgs = np.zeros((mri_height, mri_width))
            cam_img_temp = np.zeros((mri_height, mri_width))

            for n in range(cam_feature.shape[2]):
                cam_img_temp = scipy.misc.imresize(cam_feature[:,:,n], (mri_height, mri_width))

                cam_imgs += w_gap_total_value[d, no_hot_labels[p]] * w_gap_ind_value[d][n] * cam_img_temp


            mri_temp = mri[d,:,:] - np.min(mri[d,:,:])
            mri_temp = mri_temp / np.max(mri_temp)
            mri_temp = np.uint8(255 * mri_temp)
            mri_3d   = [mri_temp for _ in range(3)]
            mri_3d   = np.stack(mri_3d, 2)

            cam_temp = cam_imgs - np.min(cam_imgs)
            cam_temp = cam_temp / np.max(cam_temp)
            cam_3d[d,:,:] = cam_temp

            cam_temp = np.uint8(255 * cam_temp)

            path_img = './Hos/2DCAM_imgs/patients_images_' + str(p)
            if os.path.isdir(path_img):
                pass
            else:
                os.makedirs(path_img)

            heat_map = cv2.applyColorMap(cam_temp, cv2.COLORMAP_JET)
            result = heat_map * 0.5 + mri_3d * 0.5
            scipy.misc.imsave(path_img + '/' + 'image_' + str(d) + '.jpg', result)




        path_data = './Hos/2DCAM_data/'
        if os.path.isdir(path_data):
            pass
        else:
            os.makedirs(path_data)

        # cam_orig = np.zeros((170,256,256))
        #
        # cam_orig_temp = np.zeros((mri_depth, 256,256))
        #
        # for j in range(mri_depth):
        #     cam_orig_temp[j, :, :] = scipy.misc.imresize(cam_3d[j, :, :], (256, 256))
        # for k in range(256):
        #     cam_orig[:, k, :] = scipy.misc.imresize(cam_3d[:, k, :], (170, 256))


        # cam_3d[cam_3d >=0.5] = 1
        # cam_3d[cam_3d < 0.5] = 0
        #
        # cam_3d = np.moveaxis(cam_3d, 0, -1)

        # savemat(path_data + 'patient_' + str(p) + '_heatmap', {'heatmap': cam_3d})
        np.save(path_data + 'patient_' + str(p) + '_heatmap', cam_3d)


