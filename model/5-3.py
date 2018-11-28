import tensorflow as tf
# import tf_mnist_loader
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import sys
import os
import scipy.misc
from tqdm import trange

try:
    xrange
except NameError:
    xrange = range

#dataset = tf_mnist_loader.read_data_sets("mnist_data")
save_dir = "chckPts/"
save_prefix = "save"
summaryFolderName = "ZeroRes_summary/"
chckPtsFolderName = 'ZeroRes_chckPts/'
accuracyFolderName = 'ZeroRes_Cross_Validation/'



if len(sys.argv) == 2:
    simulationName = str(sys.argv[1])
    print("Simulation name = " + simulationName)
    summaryFolderName = summaryFolderName + simulationName + "/"
    saveImgs = True
    imgsFolderName = "imgs/" + simulationName + "/"
    if os.path.isdir(summaryFolderName) == False:
        os.makedirs(summaryFolderName)
    # if os.path.isdir(imgsFolderName) == False:
    #     os.mkdir(imgsFolderName)
else:
    saveImgs = False
    print("#######################################################################################")
    print("!!!!!!Training Start!!!!!!!")
    print("#######################################################################################")
    print("Reading MRI Data...........")
    # images_norm = np.load('image_norm.npy')

    # images_clip = np.load('image_clip.npy')
    # images_clip = np.load('image_clip_norm.npy')

    # images_clip = np.load('image_norm.npy')



    # images_norm = np.empty(images.shape)
    #
    #
    # for i in np.arange(images.shape[0]):
    #     img_max , img_min = images[i].max(), images[i].min()
    #     images_norm[i] = (images[i] - img_min) / (img_max - img_min)

    # labels = np.load('labels_data.npy')
    # np.save('image_norm', images_norm)

    # images = np.load('mri_sim_1.npy')

    images = np.load('Hos_data_0.25.npy')
    # images = np.load('image_crop_large.npy')
    #
    no_hot_labels = np.load('Hos_label.npy')

    # images = np.load('GU_data.npy')
    # no_hot_labels = np.load('GU_label.npy')

    # images = np.load('ABIDE-II_data.npy')
    # no_hot_labels = np.load('ABIDE-II_label.npy')

    # no_hot_labels = np.load('label_sim_1.npy')

    mri_patient, mri_channel, mri_height, mri_width = images.shape



    # images_norm = np.empty(images_clip.shape)
    #
    # for i in np.arange(images_clip.shape[0]):
    #     path = '/home/king/PycharmProjects/Clip_Images_Hist/patients_images_' + str(i)
    #     os.makedirs(path)
    #     for j in np.arange(images_clip.shape[1]):
    #         img_max , img_min = images_clip[i][j].max(), images_clip[i][j].min()
    #         images_norm[i][j] = (images_clip[i][j] - img_min) / (img_max - img_min)
    #         scipy.misc.imsave(path + '/' + 'image_' + str(j) + '.jpg', images_norm[i][j])
    #
    # np.save('image_hist_equal', images_norm)




####################histogram equalization###########################

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


# images_hist_equal = np.zeros(images_clip.shape)

# for i in range(images_clip.shape[0]):
#     path = '/home/king/PycharmProjects/Clip_Images_Hist_Equal/patients_images_' + str(i)
#     os.makedirs(path)
#     for j in range(images_clip.shape[1]):
#         image = images_clip[i,j,:,:]
#         images_hist_equal[i,j,:,:] = image_histogram_equalization(image)[0]
#         scipy.misc.imsave(path + '/' + 'image_' + str(j) + '.jpg', images_hist_equal[i,j,:,:])
#
# np.save('images_hist_equal', images_hist_equal)


start_step = 0
#load_path = None
load_path = save_dir + save_prefix + str(start_step) + ".ckpt"
# to enable visualization, set draw to True
eval_only = False
draw = False
animate = False






#########################All the Changes Are Here ####################################
channels = mri_channel               # mnist are grayscale images
img_size = mri_height
minRadius = 4  # zooms -> minRadius * 2**<depth_level>
depth = 3  # number of zooms
nGlimpses = 8               # number of glimpses
# batch_size = 10
batch_size = int(mri_patient * 0.2)
cross_validation_times = 30
max_iters = 100000
test_step = 1000
test_num = 100

sensorBandwidth = minRadius * 2

totalSensorBandwidth = depth * (sensorBandwidth ** 3)

dropout_rate = 0.5

initLr = 1e-3
lr_min = 1e-4
lrDecayRate = .999
lrDecayFreq = 1000



fixed_learning_rate = 0.001


loc_size = 3






momentumValue = .9

# initloc = 0.2               # std when setting the location
# loc_min = 1e-2
# locDecayRate = .999
# locDecayFreq = 5000


# model parameters




LOC_DIM = 3

# network units
hg_size = 128               #
hl_size = 128               #
g_size = 256                #
cell_size = 256             #
cell_out_size = cell_size   #

# paramters about the training examples
n_classes = 2              # card(Y)

# training parameters

preTraining = 0
preTraining_epoch = 5000
drawReconsturction = 0
SMALL_NUM = 1e-10

# resource prellocation
mean_locs = []              # expectation of locations
sampled_locs = []           # sampled locations ~N(mean_locs[.], loc_sd)
pixel_locs = []             # pixel locations
adjusted_locs = []             # adjusted pixel locations
d_crops = []
baselines = []              # baseline, the value prediction
glimpse_images = []         # to show in window
total_TPR = []               # to show true positive
total_TNR = []               # to show true negative
total_FPR = []               # to show false positive
total_FNR = []               # to show false negative
cross_validation_accuracy = [] # the cross-validation-accuracy
baseline_accuracy = []         # to record the baseline accuracy of zeros for the test samples

layer_output = []

print('The patients num: %d, channel: %d, height: %d, width: %d, Glimpse num: %d, '
      'depth: %d, minRadius: %d, sensorBandwidth: %d, batch_size: %d, CV_times: %d, dropout_rate: %.2f'
      %(mri_patient, mri_channel, mri_height, mri_width, nGlimpses,
        depth, minRadius, sensorBandwidth, batch_size, cross_validation_times, dropout_rate))


# set the weights to be small random values, with truncated normal distribution
def weight_variable(shape, myname, train):
    initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1)
    # initial = tf.random_normal(shape, mean= 0, stddev= 0.1)
    # initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1) * tf.sqrt(1.0/shape[0])
    return tf.Variable(initial, name=myname, trainable=train)

# get local glimpses
def glimpseSensor(img, normLoc):

    loc0 = tf.floor(((normLoc[:, 0] + 1) / 2.0) * channels)  # normLoc coordinates are between -1 and 1
    loc1 = tf.floor(((normLoc[:, 1] + 1) / 2.0) * img_size)
    loc2 = tf.floor(((normLoc[:, 2] + 1) / 2.0) * img_size)
    #
    # loc = tf.convert_to_tensor([loc0, loc1, loc2])
    loc = tf.stack([loc0, loc1, loc2], axis=1)
    # loc = tf.round(((normLoc + 1) / 2.0) * img_size)
    loc = tf.cast(loc, tf.int32)

    pixel_locs.append(loc)

    # img = tf.reshape(img, (batch_size, img_size, img_size, channels))


    # process each image individually
    zooms = []
    for k in range(batch_size):
        # with tf.device('/gpu:%d' % k):
        imgZooms = []
        one_img = img[k, :, :, :]
        # max_radius = minRadius * (2 ** (depth - 1))
        # offset = max_radius
        # with tf.device('/gpu:1'):
        # pad image with zeros
        # one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, max_radius * 2 + img_size, max_radius * 2 + img_size)
        # with tf.device('/gpu:2'):

        # pad_zeros = tf.zeros([one_img.get_shape()[0], one_img.get_shape()[1], offset])

        # one_img = tf.concat([pad_zeros, one_img, pad_zeros], axis=2)
        # with tf.device('/gpu:3'):
        for i in range(depth):
            r = int(minRadius * (2 ** (i)))

            d_raw = 2 * r
            d = tf.constant(d_raw, shape=[1])
            d = tf.tile(d, [3])
            loc_k = loc[k,:]

            loc_k0 = tf.clip_by_value(loc_k[0], r, one_img.get_shape()[0] - r)
            loc_k1 = tf.clip_by_value(loc_k[1], r, one_img.get_shape()[1] - r)
            loc_k2 = tf.clip_by_value(loc_k[2], r, one_img.get_shape()[2] - r)

            loc_k = tf.convert_to_tensor([loc_k0, loc_k1, loc_k2])

            adjusted_loc = loc_k - r
            adjusted_locs.append(adjusted_loc)

            # one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value, one_img.get_shape()[1].value))

            # crop image to (d x d x d)

            d_crop0 = tf.minimum(d[0], one_img.get_shape()[0])
            d_crop1 = tf.minimum(d[1], one_img.get_shape()[1])
            d_crop2 = tf.minimum(d[2], one_img.get_shape()[2])

            d_crop = tf.convert_to_tensor([d_crop0, d_crop1, d_crop2])
            d_crops.append(d_crop)
            zoom = tf.slice(one_img, adjusted_loc, d_crop)

            # resize cropped image to (sensorBandwidth x sensorBandwidth)
            zoom = tf.transpose(zoom,[1,2,0])
            zoom = tf.image.resize_images(zoom, (sensorBandwidth, sensorBandwidth))
            zoom = tf.transpose(zoom, [2,0,1])
            zoom = tf.image.resize_images(zoom,(sensorBandwidth,sensorBandwidth))
            # zoom = tf.transpose(zoom,[2,3,1])
            zoom = tf.reshape(zoom, (sensorBandwidth, sensorBandwidth, sensorBandwidth))
            imgZooms.append(zoom)

        zooms.append(tf.stack(imgZooms))

    zooms = tf.stack(zooms)

    glimpse_images.append(zooms)

    return zooms

# implements the input network
def get_glimpse(loc, glimpse_step):
    # get input using the previous location

    # inputs_placeholder = tf.contrib.layers.batch_norm(inputs_placeholder, center= True, scale = True, is_training = tf_is_training)


    glimpse_input = glimpseSensor(inputs_placeholder, loc)

    glimpse_input = tf.reshape(glimpse_input, (batch_size, totalSensorBandwidth))

    # tf.summary.histogram('glimpse_image_step_' + str(glimpse_step), glimpse_input[0,:])

    glimpse_input = tf.contrib.layers.batch_norm(glimpse_input, center= True, scale = True, is_training = tf_is_training)

    tf.summary.histogram('glimpse_image_step_' + str(glimpse_step), glimpse_input[0,:])

    # the hidden units that process location & the input
    act_glimpse_hidden = tf.matmul(glimpse_input, Wg_g_h) + Bg_g_h

    act_glimpse_hidden = tf.contrib.layers.batch_norm(act_glimpse_hidden, center=True, scale=True, is_training=tf_is_training)

    act_glimpse_hidden = tf.nn.leaky_relu(act_glimpse_hidden)

    tf.summary.histogram('glimpse_feature_step_' + str(glimpse_step), act_glimpse_hidden[0, :])

    act_glimpse_hidden = tf.layers.dropout(act_glimpse_hidden, rate = dropout_rate, training= tf_is_training)

    act_loc_hidden = tf.matmul(loc, Wg_l_h) + Bg_l_h

    act_loc_hidden = tf.contrib.layers.batch_norm(act_loc_hidden, center=True, scale=True, is_training=tf_is_training)

    act_loc_hidden = tf.nn.leaky_relu(act_loc_hidden)

    tf.summary.histogram('location_feature_step' + str(glimpse_step), act_loc_hidden[0,:])

    act_loc_hidden = tf.layers.dropout(act_loc_hidden, rate= dropout_rate, training = tf_is_training)            ###dropout

    # the hidden units that integrates the location & the glimpses

    glimpseFeature = tf.matmul(act_glimpse_hidden, Wg_hg_gf) + tf.matmul(act_loc_hidden, Wg_hl_gf) + Bg_hlhg_gf
    # glimpseFeature = tf.matmul(act_glimpse_hidden, Wg_hg_gf)  + Bg_hlhg_gf

    glimpseFeature = tf.contrib.layers.batch_norm(glimpseFeature, center=True, scale=True, is_training=tf_is_training)

    glimpseFeature = tf.nn.leaky_relu(glimpseFeature)

    tf.summary.histogram('loc_glimpse_feature_step' + str(glimpse_step), glimpseFeature[0, :])


    glimpseFeature = tf.layers.dropout(glimpseFeature, rate=dropout_rate,  training = tf_is_training)           ###dropout
    # return g
    # glimpseFeature2 = tf.matmul(glimpseFeature1, Wg_gf1_gf2) + Bg_gf1_gf2
    return glimpseFeature


def get_next_input(output, glimpse_step):
    # the next location is computed by the location network
    core_net_out = tf.stop_gradient(output)

    # baseline = tf.sigmoid(tf.matmul(core_net_out, Wb_h_b) + Bb_h_b)
    baseline = tf.sigmoid(tf.matmul(core_net_out, Wb_h_b) + Bb_h_b)
    baselines.append(baseline)

    # compute the next location, then impose noise

    # def f1(): return tf.random_normal((batch_size, LOC_DIM), mean = 0, stddev= 0.3)
    # def f2(): return tf.clip_by_value(tf.matmul(core_net_out, Wl_h_l) + Bl_h_l, -1, 1)
    #
    # mean_loc = tf.cond(tf.less(training_step, 30000), f1, f2)


    mean_loc = tf.maximum(-1.0, tf.minimum(1.0, tf.matmul(core_net_out, Wl_h_l) + Bl_h_l))


    # mean_loc = tf.matmul(core_net_out, Wl_h_l) + Bl_h_l
    # mean_loc = tf.clip_by_value(mean_loc, -1, 1)
    # mean_loc = tf.stop_gradient(mean_loc)
    mean_locs.append(mean_loc)

    # add noise
    # sample_loc = tf.tanh(mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd))
    sample_loc = tf.maximum(-1.0, tf.minimum(1.0, mean_loc + tf.random_normal([batch_size, LOC_DIM], 0, loc_sd)))

    # don't propagate throught the locations
    sample_loc = tf.stop_gradient(sample_loc)
    sampled_locs.append(sample_loc)

    return get_glimpse(sample_loc, glimpse_step)


def affineTransform(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim])
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b


def model():

    # initialize the location under unif[-1,1], for all example in the batch
    # initial_loc = tf.random_uniform((batch_size, LOC_DIM), minval=-1, maxval=1)

    # initial_loc = tf.random_normal((batch_size, LOC_DIM), mean = 0, stddev= init_loc_sd)

    initial_loc = tf.zeros((batch_size, LOC_DIM))

    mean_locs.append(initial_loc)

    # initial_loc = tf.tanh(initial_loc + tf.random_normal(initial_loc.get_shape(), 0, loc_sd))
    initial_loc = tf.clip_by_value(initial_loc + tf.random_normal((batch_size, LOC_DIM), 0, loc_sd), -1, 1)

    sampled_locs.append(initial_loc)

    # get the input using the input network
    initial_glimpse = get_glimpse(initial_loc, 0)

    # set up the recurrent structure
    inputs = [0] * nGlimpses
    outputs = [0] * nGlimpses
    glimpse = initial_glimpse
    # REUSE = None
    for t in range(nGlimpses):
        if t == 0:  # initialize the hidden state to be the zero vector
            hiddenState_prev = tf.zeros((batch_size, cell_size))
        else:
            hiddenState_prev = outputs[t-1]

        # forward prop
        # with tf.variable_scope("coreNetwork", reuse=REUSE):
        # the next hidden state is a function of the previous hidden state and the current glimpse
        h1 = tf.matmul(hiddenState_prev, Wc_r_h) + Bc_r_h

        tf.summary.histogram('CoreNetwork_h1_step_' + str(t), tf.expand_dims(h1[0, :], axis=0))

        h2 = tf.matmul(glimpse, Wc_g_h) + Bc_g_h

        tf.summary.histogram('CoreNetwork_h2_step_' + str(t), tf.expand_dims(h2[0,:], axis= 0))

        hiddenState = h1 + h2

        hiddenState = tf.contrib.layers.batch_norm(hiddenState, center=True, scale=True, is_training=tf_is_training)

        hiddenState = tf.nn.leaky_relu(hiddenState)

        tf.summary.histogram('CoreNetwork_h1+h2_step_' + str(t), hiddenState[0,:])

        hiddenState = tf.layers.dropout(hiddenState, rate=dropout_rate,  training = tf_is_training)

        # save the current glimpse and the hidden state
        inputs[t] = glimpse
        outputs[t] = hiddenState
        # get the next input glimpse
        if t != nGlimpses -1:
            glimpse = get_next_input(hiddenState, t + 1)
        else:
            first_hiddenState = tf.stop_gradient(hiddenState)
            # baseline = tf.sigmoid(tf.matmul(first_hiddenState, Wb_h_b) + Bb_h_b)
            baseline = tf.sigmoid(tf.matmul(first_hiddenState, Wb_h_b) + Bb_h_b)
            baselines.append(baseline)
        # REUSE = True  # share variables for later recurrence

    return outputs


def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# to use for maximum likelihood with input location
def gaussian_pdf(mean, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
    return Z * tf.exp(a)


def calc_reward(outputs):

    # correct_y = tf.cast(labels_placeholder, tf.int64)
    # consider the action at the last time step
    # for step in range(nGlimpses - 1):
    #     output_temp = outputs[step]
    #     output_temp = tf.reshape(output_temp, (batch_size, cell_out_size))
    #     tf.summary.histogram('before_softmax_function_step_' + str(step), output_temp)
    #     p_y_temp  = tf.nn.softmax(tf.matmul(output_temp, Wa_h_a) + Ba_h_a)
    #     tf.summary.histogram('after_softmax_function_step_' + str(step), p_y_temp)
    #
    #     max_p_y_temp = tf.argmax(p_y_temp, 1)
    #
    #     # reward for all examples in the batch
    #     R_temp = tf.cast(tf.equal(max_p_y_temp, correct_y), tf.float32)
    #     reward_temp = tf.reduce_mean(R_temp)  # mean reward
    #
    #     tf.summary.scalar('accuracy_step_' + str(step), reward_temp)


    outputs = tf.transpose(tf.stack(outputs),[1,0,2])
    # outputs = outputs[0] # look at ONLY THE END of the sequence
    outputs = tf.reshape(outputs, (batch_size, 8 * cell_out_size))

    tf.summary.histogram('before_softmax_function_step_7', outputs)

    # get the baseline
    b = tf.stack(baselines)
    b = tf.concat(axis=2, values=[b, b, b])
    b = tf.reshape(b, (batch_size, (nGlimpses) * loc_size))
    # no_grad_b = tf.stop_gradient(b)

    # get the action(classification)
    outputs_drop = tf.layers.dropout(outputs, rate=dropout_rate,  training = tf_is_training)
    p_y = tf.nn.softmax(tf.matmul(outputs_drop, Wa_h_a) + Ba_h_a)

    tf.summary.histogram('after_softmax_function_step_7', p_y)


    max_p_y = tf.argmax(p_y, 1)
    correct_y = tf.cast(labels_placeholder, tf.int64)

    # reward for all examples in the batch
    R = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)
    reward = tf.reduce_mean(R) # mean reward
    R = tf.reshape(R, (batch_size, 1))
    R = tf.tile(R, [1, (nGlimpses)*loc_size])
    no_grad_R = tf.stop_gradient(R)

    # get the location

    p_loc = gaussian_pdf(mean_locs, sampled_locs)
    # p_loc = tf.tanh(p_loc)

    # p_loc_orig = p_loc
    p_loc = tf.reshape(p_loc, (batch_size, (nGlimpses) * loc_size))

    # define the cost function
    # J = tf.concat(axis=1, values=[tf.log(p_y + SMALL_NUM) * (onehot_labels_placeholder), tf.log(p_loc + SMALL_NUM) * (R - no_grad_b)])

    J1 = tf.log(p_y + SMALL_NUM) * (onehot_labels_placeholder)
    J2 = tf.log(p_loc + SMALL_NUM) * (no_grad_R - b)

    J21 = tf.log(p_loc + SMALL_NUM)
    J22 = no_grad_R - b



    # JP1 = tf.log(p_loc + SMALL_NUM)
    # JP2 = (no_grad_R - b)

    temp = (mean_locs + 1)/2
    # J3 = tf.nn.relu(2.5 * (1 - tf.exp(-100 * (temp - 0.8))))
    scale = 2.5

    J3 = tf.nn.relu(scale * (1 - tf.exp(-100 * (temp - 0.8)))) + tf.nn.relu(
        scale * (1 - tf.exp(100 * (temp - 0.2))))
    # J3_1 = tf.nn.relu(scale * (1 - tf.exp(-100*(temp[:,:,0] - 0.9))))
    #
    # J3_1 = tf.reshape(J3_1 , [batch_size, nGlimpses, 1])
    #
    # J3_2 = tf.nn.relu(scale * (1 - tf.exp(-100*(temp[:,:,1:] - 0.8)))) + tf.nn.relu(scale * (1 - tf.exp(100*(temp[:,:,1:] - 0.2))))
    #
    # J3 = tf.concat([J3_1, J3_2], axis= 2)

    J3 = tf.reduce_sum(J3, axis=2)
    # J3 = tf.zeros((batch_size, nGlimpses))
    # for i in range(batch_size):
    #     for j in range(nGlimpses):
    #         x = sampled_locs[i][j][0]
    #         y = sampled_locs[i][j][1]
    #         z = sampled_locs[i][j][2]
    #         and_x = tf.logical_and(tf.greater_equal(x, -1.0), tf.less_equal(x, 0.8))
    #         and_y = tf.logical_and(tf.greater_equal(y, -0.6), tf.less_equal(y, 0.6))
    #         and_z = tf.logical_and(tf.greater_equal(z, -0.6), tf.less_equal(z, 0.6))
    #         and_xy = tf.logical_and(and_x, and_y)
    #         and_xyz = tf.logical_and(and_xy, and_z)
            # def f1():  return
            # def f2(): J3[i,j] = 50
            # sess_temp = tf.Session()
            # init_temp = tf.global_variables_initializer()
            # sess_temp.run(init_temp)
            # result_temp = tf.where(and_xyz, tf.constant(0), tf.constant(50))
            # J3[i,j] = result_temp.eval()
            # sess_temp.close()
            # if and_xyz is True:
            #     pass
            # else:
            #     J3[i,j] = 50

    J4 = tf.square(no_grad_R - b)
    J = tf.concat(axis=1, values=[J1, J2, -J3, -J4])
    J_total = [tf.reduce_mean(J1), tf.reduce_mean(J2), tf.reduce_mean(J3), tf.reduce_mean(J4)]


    tf.summary.scalar('maximize classification cost', tf.reduce_mean(J1))
    tf.summary.scalar('maximize reinforcement cost', tf.reduce_mean(J2))
    tf.summary.scalar('minimize location cost', tf.reduce_mean(J3))
    tf.summary.scalar('minimize (reward - b) ** 2', tf.reduce_mean(J4))




    J = tf.reduce_sum(J, 1)


    # J = J - 0.5 * tf.reduce_sum(tf.reduce_sum(tf.square(sampled_locs),1),1)
    # J = J - tf.reduce_sum(tf.square(R - b), 1)

    # J = J - tf.reduce_sum(tf.square(no_grad_R - b), 1) # for variance reduction


    J = tf.reduce_mean(J, 0)

    # J_temp = tf.reduce_sum(J3,1)
    # J_temp = tf.reduce_mean(J_temp,0)

    cost = -J
    # var_list = tf.trainable_variables()
    # grads = tf.gradients(cost, var_list)
    # grads, _ = tf.clip_by_global_norm(grads, 0.5)
    # define the optimizer
    lr_max = tf.maximum(lr, lr_min)
    optimizer = tf.train.AdamOptimizer(lr_max)
    # train_op = optimizer.minimize(cost, global_step)


    ##############batch_normailization###########
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(cost, global_step)
    # optimizer = tf.train.MomentumOptimizer(lr, momentumValue)
    # train_op = optimizer.minimize(cost, global_step)
    # train_op = optimizer.apply_gradients(zip(grads, var_list), global_step=global_step)

    return cost, reward, max_p_y, correct_y, train_op, b, tf.reduce_mean(b), tf.reduce_mean(R - b), lr_max, J_total, J2, J21, J22


def preTrain(outputs):
    lr_r = 1e-3
    # consider the action at the last time step
    outputs = outputs[-1] # look at ONLY THE END of the sequence
    outputs = tf.reshape(outputs, (batch_size, cell_out_size))
    # if preTraining:
    reconstruction = tf.sigmoid(tf.matmul(outputs, Wr_h_r) + Br_h_r)
    inputs_temp = tf.reshape(inputs_placeholder,[batch_size, channels * img_size **2])
    reconstructionCost = tf.reduce_mean(tf.square(inputs_temp - reconstruction))

    train_op_r = tf.train.RMSPropOptimizer(lr_r).minimize(reconstructionCost)
    return reconstructionCost, reconstruction, train_op_r


def evaluate():

    # accuracy = 0
    # TP = 0
    # TN = 0
    # FP = 0
    # FN = 0
    #
    # test_index = np.arange(0, 8)
    # test_index = np.append(test_index, [80, 81])


    # index = test_index.reshape([-1,batch_size])
    index = test_index
    nextX = images[index]
    # nextX = images_norm[index[i]]
    nextY = no_hot_labels[index]

    # tf.summary.scalar('test_accuracy', reward)

    feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY,\
                     onehot_labels_placeholder: dense_to_one_hot(nextY),\
                     tf_is_training : False, loc_sd : var_loc, init_loc_sd: init_var_loc, training_step: epoch}
    summary_test, r_value, y_value = sess.run([accuracy_summary, reward, predicted_labels], feed_dict=feed_dict)
    # print(('Test_sample = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d] Labels = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d] Average_reward = %.5f'\
    #        % (index[i][0],index[i][1], index[i][2],index[i][3],index[i][4],index[i][5], index[i][6],index[i][7],index[i][8],index[i][9],\
    #           nextY[0], nextY[1], nextY[2], nextY[3], nextY[4], nextY[5], nextY[6], nextY[7],nextY[8], nextY[9], r_value)))
    # print(y_value)
    # # print(nextY)
    # for j in range(batch_size):
    #     if y_value[j] == 1 and nextY[j] == 1:
    #         TP +=1
    #     elif y_value[j] == 0 and nextY[j] == 0:
    #         TN +=1
    #     elif y_value[j] == 1 and nextY[j] == 0:
    #         FP +=1
    #     elif y_value[j] == 0 and nextY[j] == 1:
    #         FN +=1
    # TPR = TP / float(TP + FN)
    # TNR = TN / float(TN + FP)
    # FPR = FP / float(FP + TN)
    # FNR = FN / float(TP + FN)

    # print(TP)
    # print(TN)
    # print(FP)
    # print(FN)
    # summary_str = sess.run(accuracy_summary, feed_dict=feed_dict)


    # print(('Test_sample = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]'\
    #     % (index[0],index[1], index[2],index[3], index[4], \
    #         index[5], index[6],index[7],index[8],index[9])))
    # print(('Labels      = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]' \
    #        % (nextY[0], nextY[1], nextY[2], nextY[3], nextY[4], \
    #           nextY[5], nextY[6], nextY[7],nextY[8], nextY[9])))
    # print(('Predicted   = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]' \
    #        % (y_value[0], y_value[1], y_value[2], y_value[3], y_value[4], \
    #           y_value[5], y_value[6], y_value[7], y_value[8], y_value[9])))

    accuracy = r_value



    # TPR /= float(n)
    # TNR /= float(n)
    # FPR /= float(n)
    # FNR /= float(n)

    # tf.summary.scalar('test_accuracy', reward)

    # total_acc.append(accuracy)
    # total_TPR.append(TPR)
    # total_TNR.append(TNR)
    # total_FPR.append(FPR)
    # total_FNR.append(FNR)

    # print(("TOTAL ACCURACY: %.5f" %(accuracy)))
    # print(("TOTAL True Positive Rate: %.5f" %(TPR)))
    # print(("TOTAL True Negative Rate: %.5f" %(TNR)))
    # print(("TOTAL False Positive Rate: %.5f"  %(FPR)))
    # print(("TOTAL False Negative Rate: %.5f" %(FNR)))
    return summary_test, accuracy



def toImgCoordinates(coordinate_tanh):
    '''
    Transform coordinate in [-1,1] to mnist
    :param coordinate_tanh: vector in [-1,1] x [-1,1]
    :return: vector in the corresponding mnist coordinate
    '''
    return np.round(((coordinate_tanh + 1) / 2.0) * img_size)


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


def plotWholeImg(img, img_size, sampled_locs_fetched):
    plt.imshow(np.reshape(img, [img_size, img_size]),
               cmap=plt.get_cmap('gray'), interpolation="nearest")

    plt.ylim((img_size - 1, 0))
    plt.xlim((0, img_size - 1))

    # transform the coordinate to mnist map
    sampled_locs_mnist_fetched = toImgCoordinates(sampled_locs_fetched)
    # visualize the trace of successive nGlimpses (note that x and y coordinates are "flipped")
    plt.plot(sampled_locs_mnist_fetched[0, :, 1], sampled_locs_mnist_fetched[0, :, 0], '-o',
             color='lawngreen')
    plt.plot(sampled_locs_mnist_fetched[0, -1, 1], sampled_locs_mnist_fetched[0, -1, 0], 'o',
             color='red')

with tf.device('/device:GPU:0'):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(initLr, global_step, lrDecayFreq, lrDecayRate, staircase=True)
        # loc_sd = tf.train.exponential_decay(initloc, global_step, locDecayFreq, locDecayRate, staircase=True)

        # preallocate x, y, baseline
        # labels = tf.placeholder("float32", shape=[batch_size, n_classes])
        labels_placeholder = tf.placeholder(tf.float32, shape=(None), name="labels_raw")
        onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(None, n_classes), name="labels_onehot")
        inputs_placeholder = tf.placeholder(tf.float32, shape=(None, mri_channel, mri_height, mri_width), name="images")
        tf_is_training = tf.placeholder(tf.bool, None, name= 'normalization')
        loc_sd = tf.placeholder(tf.float32, None, name= 'location_sd')
        init_loc_sd = tf.placeholder(tf.float32, None, name= 'initial_location_sd')
        training_step = tf.placeholder(tf.int32, None, name= 'training_step')


        # J3 = tf.placeholder(tf.float32, (batch_size, nGlimpses), name='location_penalty')

        # declare the model parameters, here're naming rule:
        # the 1st captical letter: weights or bias (W = weights, B = bias)
        # the 2nd lowercase letter: the network (e.g.: g = glimpse network)
        # the 3rd and 4th letter(s): input-output mapping, which is clearly written in the variable name argument
        Wg_l_h = weight_variable((loc_size, hl_size), "glimpseNet_wts_location_hidden", True)
        Bg_l_h = weight_variable((1,hl_size), "glimpseNet_bias_location_hidden", True)

        Wg_g_h = weight_variable((totalSensorBandwidth, hg_size), "glimpseNet_wts_glimpse_hidden", True)
        Bg_g_h = weight_variable((1,hg_size), "glimpseNet_bias_glimpse_hidden", True)

        Wg_hg_gf = weight_variable((hg_size, g_size), "glimpseNet_wts_hiddenGlimpse_glimpseFeature", True)
        Wg_hl_gf = weight_variable((hl_size, g_size), "glimpseNet_wts_hiddenLocation_glimpseFeature", True)
        Bg_hlhg_gf = weight_variable((1,g_size), "glimpseNet_bias_hGlimpse_hLocs_glimpseFeature", True)

        Wc_g_h = weight_variable((cell_size, g_size), "coreNet_wts_glimpse_hidden", True)
        Bc_g_h = weight_variable((1,g_size), "coreNet_bias_glimpse_hidden", True)

        Wc_r_h = weight_variable((cell_size, cell_size), "coreNet_wts_recurrent_hidden", True)
        Bc_r_h = weight_variable((1, cell_size), "coreNet_wts_recurrent_hidden", True)

        # Wr_h_r = weight_variable((cell_out_size, channels * img_size**2), "reconstructionNet_wts_hidden_action", True)
        # Br_h_r = weight_variable((1, channels * img_size**2), "reconstructionNet_bias_hidden_action", True)

        # with tf.device('/gpu:1'):
        Wb_h_b = weight_variable((g_size, 1), "baselineNet_wts_hiddenState_baseline", True)
        Bb_h_b = weight_variable((1,1), "baselineNet_bias_hiddenState_baseline", True)

        Wl_h_l = weight_variable((cell_out_size, loc_size), "locationNet_wts_hidden_location", True)
        Bl_h_l = weight_variable((1, loc_size), "locationNet_bias_hidden_location", True)

        Wa_h_a = weight_variable((8 * cell_out_size, n_classes), "actionNet_wts_hidden_action", True)
        Ba_h_a = weight_variable((1,n_classes),  "actionNet_bias_hidden_action", True)

        # query the model ouput
        # with tf.device('/gpu:2'):
        outputs = model()

        # convert list of tensors to one big tensor
        sampled_locs = tf.concat(axis=0, values=sampled_locs)
        sampled_locs = tf.reshape(sampled_locs, (nGlimpses, batch_size, loc_size))
        sampled_locs = tf.transpose(sampled_locs, [1, 0, 2])
        mean_locs = tf.concat(axis=0, values=mean_locs)
        mean_locs = tf.reshape(mean_locs, (nGlimpses, batch_size, loc_size))
        mean_locs = tf.transpose(mean_locs, [1, 0, 2])
        glimpse_images = tf.concat(axis=0, values=glimpse_images)

        # compute the reward
        # reconstructionCost, reconstruction, train_op_r = preTrain(outputs)
        cost, reward, predicted_labels, correct_labels, train_op, b, avg_b, rminusb, lr, J_total, J2, J21, J22 = calc_reward(outputs)

        # tf.summary.scalar("reconstructionCost", reconstructionCost)
        accuracy_summary = tf.summary.scalar("accuracy", reward)
        tf.summary.scalar("cost", cost)
        tf.summary.scalar("mean_b", avg_b)
        tf.summary.scalar("mean_R-b", rminusb)

        tf.summary.histogram('MRI_image', inputs_placeholder)

        print("Start Evaluating the Recurrent Attention Neural Network ........")


        for exp in range(20, cross_validation_times):
            full_index = np.arange(mri_patient)
            test_choice = int(mri_patient * 0.2)
            test_index = np.random.choice(mri_patient, test_choice, replace= False)
            train_index = np.setdiff1d(full_index, test_index)

            test_baseline = 1 - np.mean(no_hot_labels[test_index])

            # print(test_baseline)
            print('Cross Validation Start: Experiment ' + str(exp) + ' ' + 'Max_step = ' + str(max_iters))

            mean_acc = []  # to show the accuracy
            std_acc = []

            # tensorboard visualization for the parameters
            variable_summaries(Wg_l_h, "glimpseNet_wts_location")
            variable_summaries(Bg_l_h, "glimpseNet_bias_location")
            variable_summaries(Wg_g_h, "glimpseNet_wts_glimpse")
            variable_summaries(Bg_g_h, "glimpseNet_bias_glimpse")
            variable_summaries(Wg_hg_gf, "glimpseNet_wts_Glimpse_glimpseFeature")
            variable_summaries(Wg_hl_gf, "glimpseNet_wts_Location_glimpseFeature")
            variable_summaries(Bg_hlhg_gf, "glimpseNet_bias_Glimpse_Locs_glimpseFeature")

            variable_summaries(Wc_g_h, "coreNet_wts_glimpse")
            variable_summaries(Bc_g_h, "coreNet_bias_glimpse")

            variable_summaries(Wc_r_h, "coreNet_wts_recurrent")
            variable_summaries(Bc_r_h, "coreNet_bias_recurrent")

            variable_summaries(Wb_h_b, "baselineNet_wts_")
            variable_summaries(Bb_h_b, "baselineNet_bias")

            variable_summaries(Wl_h_l, "locationNet_wts")
            variable_summaries(Bl_h_l, "locationNet_bias")

            variable_summaries(Wa_h_a, 'actionNet_wts')
            variable_summaries(Ba_h_a, 'actionNet_bias')

            # tensorboard visualization for the performance metrics
            # tf.summary.scalar("reconstructionCost", reconstructionCost)

            #drawing the attention image in tensorboard

            # accuracy_summary = tf.summary.scalar("accuracy", reward)
            # tf.summary.scalar("cost", cost)
            # tf.summary.scalar("mean(b)", avg_b)
            # tf.summary.scalar("mean(R - b)", rminusb)
            #
            # tf.summary.histogram('MRI_image', inputs_placeholder)

            # inputs_placeholder = tf.contrib.layers.batch_norm(inputs_placeholder, center=True, scale=True, is_training=tf_is_training)
            #
            # tf.summary.histogram('MRI_image_norm', inputs_placeholder)


            img_mri = inputs_placeholder[0, :, :, :]
            mean_one = (mean_locs[0, :, :] + 1) / 2
            sample_one = (sampled_locs[0, :, :] + 1) / 2

            min_loc_imgsize = depth * minRadius / (1.0 * img_size)
            max_loc_imgsize = 1 - depth * minRadius / (1.0 * img_size)

            min_loc_channel = depth * minRadius / (1.0 * channels)
            max_loc_channel = 1 - depth * minRadius / (1.0 * channels)

            mean_one_0 = tf.clip_by_value(mean_one[:, 0], min_loc_channel, max_loc_channel)
            mean_one_12 = tf.clip_by_value(mean_one[:, 1:], min_loc_imgsize, max_loc_imgsize)

            sample_one_0 = tf.clip_by_value(sample_one[:, 0], min_loc_channel, max_loc_channel)
            sample_one_12 = tf.clip_by_value(sample_one[:, 1:], min_loc_imgsize, max_loc_imgsize)

            glimpse = []
            for i in range(nGlimpses):
                glimpse_num = tf.floor(mean_one_0[i] * channels)
                glimpse_num = tf.cast(glimpse_num, tf.int32)
                temp_image = img_mri[glimpse_num,:,:]
                temp_image = tf.expand_dims(temp_image, axis=2)
                temp_image = tf.expand_dims(temp_image, axis=0)
                glimpse.append(temp_image)
            image_glimpse = tf.concat(glimpse, axis= 0)


            mean_box = []
            sample_box = []
            for i in range(depth):
                # mean location bounding box
                mean_left_corner = mean_one_12 - (i + 1) * minRadius / (1.0 * img_size)
                # mean_left_corner = tf.clip_by_value(mean_left_corner, 0, 1)
                mean_right_corner = mean_one_12 + (i + 1) * minRadius / (1.0 * img_size)
                # mean_right_corner = tf.clip_by_value(mean_right_corner, 0, 1)
                mean_temp_box = tf.concat([mean_left_corner, mean_right_corner], axis=1)
                mean_temp_box = tf.expand_dims(mean_temp_box, axis=1)
                mean_box.append(mean_temp_box)

                # sample location bounding box
                sample_left_corner = sample_one_12 - (i + 1) * minRadius / (1.0 * img_size)
                # sample_left_corner = tf.clip_by_value(sample_left_corner, 0, 1)
                sample_right_corner = sample_one_12 + (i + 1) * minRadius / (1.0 * img_size)
                # sample_left_corner = tf.clip_by_value(sample_right_corner, 0, 1)
                sample_temp_box = tf.concat([sample_left_corner, sample_right_corner], axis=1)
                sample_temp_box = tf.expand_dims(sample_temp_box, axis=1)
                sample_box.append(sample_temp_box)
            mean_bounding_box = tf.concat(mean_box, axis=1)
            sample_bounding_box = tf.concat(sample_box, axis=1)

            mean_image = tf.image.draw_bounding_boxes(image_glimpse, mean_bounding_box)
            sample_image = tf.image.draw_bounding_boxes(image_glimpse, sample_bounding_box)
            tf.summary.image('mean_locs', mean_image, nGlimpses)
            tf.summary.image('sample_locs', sample_image, nGlimpses)



            summary_op = tf.summary.merge_all()


            ####################################### START RUNNING THE MODEL #######################################
            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess_config.gpu_options.allow_growth = True
            sess = tf.Session(config=sess_config)

            summary_writer_train = tf.summary.FileWriter(summaryFolderName + str(exp) + '/train', graph=sess.graph)
            summary_writer_test = tf.summary.FileWriter(summaryFolderName + str(exp) + '/test')

            saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            sess.run(init)

            # saver.restore(sess, save_dir + save_prefix + str(100000) + ".ckpt")

            var_loc = 1.0
            init_var_loc = 0.3
            # var_loc = 0.2

            ##Pretraining

            reconstruction_value = []

            if preTraining:
                print('#################This is pretraining###################')
                for epoch_r in range(1, preTraining_epoch):
                    start_time = time.time()

                    batch_samples = np.random.choice(train_index, batch_size)

                    nextX = images[batch_samples]

                    fetches_r = [reconstructionCost, reconstruction, train_op_r]

                    reconstructionCost_fetched, reconstruction_fetched, train_op_r_fetched = sess.run(fetches_r, feed_dict={
                        inputs_placeholder: nextX, tf_is_training: True, loc_sd : var_loc})

                    duration = time.time() - start_time

                    reconstruction_value.append(reconstructionCost_fetched)


                    if epoch_r % 20 == 0:
                        print(('Step %d: reconstructionCost = %.5f timeCost = %.3f'
                               % (epoch_r, reconstructionCost_fetched, duration)))
                    if epoch_r % 400 == 0:
                        m, s = divmod(duration * (preTraining_epoch - epoch_r), 60)
                        h, m = divmod(m, 60)
                        d, h = divmod(h, 24)
                        print('the remaining time is %02d:%02d:%02d:%02d' % (d, h, m, s))

                plt.plot(reconstruction_value)
                # plt.show(block = False)
                # plt.pause(3)
                # plt.show(block = False)
                plt.savefig('reconstruction value line.png', figsize=(1280, 1024), dpi=500)

            # training
            # var_loc = 0.2
            for epoch in range(start_step + 1, max_iters):
                start_time = time.time()

                batch_samples = np.random.choice(train_index, batch_size)

                nextX = images[batch_samples]
                # nextX = images_norm[batch_samples]
                nextY = no_hot_labels[batch_samples]

                # get the next batch of examples

                feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY,
                                    onehot_labels_placeholder: dense_to_one_hot(nextY),
                                    tf_is_training: True, loc_sd : var_loc, init_loc_sd: init_var_loc,
                             training_step: epoch}

                fetches = [train_op, J2, J21, J22, J_total, adjusted_locs, d_crops, outputs,
                           cost, reward, predicted_labels, correct_labels, glimpse_images,
                           avg_b, rminusb, mean_locs, sampled_locs, lr, mean_one_0, sample_one_0]
                # feed them to the model
                results = sess.run(fetches, feed_dict=feed_dict)

                _, J2_value, J21_value, J22_value, J_total_value, adjusted_locs_value, \
                d_crops_value,outputs_value , cost_fetched, reward_fetched, prediction_labels_fetched, \
                correct_labels_fetched, glimpse_images_fetched, avg_b_fetched, \
                rminusb_fetched, mean_locs_fetched, sampled_locs_fetched, lr_fetched, \
                mean_one_0_value, sample_one_0_value = results

                duration = time.time() - start_time

                label_zero = 1 - np.average(nextY)


                # ###########draw different glimpse image##############
                #
                # for i in range(nGlimpses):
                #     mean_num = np.floor(mean_one_0_value[i] * channels)
                #     # mean_num = tf.cast(mean_num, tf.string)
                #     tf.summary.image('mean_locs_' + str(mean_num), tf.expand_dims(mean_image[i,:,:,:], axis=0), max_outputs= 1)
                # for i in range(nGlimpses):
                #     sample_num = np.floor(sample_one_0_value[i] * channels)
                #     # sample_num = tf.cast(sample_num, tf.string)
                #     tf.summary.image('sample_locs_' + str(sample_num), tf.expand_dims(sample_image[i,:,:,:], axis=0), max_outputs=1)
                #
                #
                # summary_op = tf.summary.merge_all()
                #
                # summary_writer = tf.summary.FileWriter(summaryFolderName + str(exp), graph=sess.graph)


                if epoch % 20 == 0:
                    # print(base_values)
                    # print(JP1_value)
                    # print('########')
                    # print(JP2_value)
                    var_loc *= 0.9992
                    init_var_loc *= 0.996
                    # var_loc *= 0.9998
                    var_loc = np.maximum(var_loc, 0.03)
                    init_var_loc = np.maximum(init_var_loc, 0.001)
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer_train.add_summary(summary_str, epoch)
                    # print(J_total_value)
                    # print(mean_locs_fetched)
                    # print('##############')
                    # print(temp_value)
                    # print('##############')
                    # print(J3_value)
                    # print(adjusted_locs_value)
                    # print(d_crops_value)
                    # print('before softmax############')
                    # print(np.linalg.norm(outputs_value[-1][0,:] - outputs_value[-1][1,:])/np.linalg.norm(outputs_value[-1][0,:]))
                    # print('after softmax#############')
                    # print(p_y_value)
                    # J3_mean = np.mean(J3_value)
                    # print(np.floor(mean_one_0_value * channels))
                    print(('Exp: %d Step %d: cost = %.3f (J_total = [%.3f, %.3f, %.3f, %.3f]) reward = %.3f (%.3f sec) label_zero = %.3f b = %.5f, LR = %.5f, Loc_SD = %.5f, Init_locSD = %.5f '
                               % (exp, epoch, cost_fetched, J_total_value[0], J_total_value[1],J_total_value[2],J_total_value[3], reward_fetched, duration, label_zero, avg_b_fetched, lr_fetched, var_loc, init_var_loc)))



                    # print(('         Batch_samples = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d], Labels = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]'
                    #         %(batch_samples[0], batch_samples[1], batch_samples[2], batch_samples[3], batch_samples[4], batch_samples[5], batch_samples[6], batch_samples[7],batch_samples[8], batch_samples[9],\
                    #         nextY[0], nextY[1], nextY[2], nextY[3],nextY[4], nextY[5], nextY[6], nextY[7],nextY[8], nextY[9])))

                    # print(('Labels      = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]' \
                    #        % (nextY[0], nextY[1], nextY[2], nextY[3], nextY[4], \
                    #           nextY[5], nextY[6], nextY[7], nextY[8], nextY[9])))
                    # print(('Predicted   = [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]' \
                    #        % (p_y_value[0,nextY[0]], p_y_value[1,nextY[1]], p_y_value[2,nextY[2]], p_y_value[3,nextY[3]], p_y_value[4,nextY[4]], \
                    #           p_y_value[5,nextY[5]], p_y_value[6,nextY[6]], p_y_value[7,nextY[7]], p_y_value[8,nextY[8]], p_y_value[9,nextY[9]])))


                    if epoch % test_step == 0:

                        path_chck = chckPtsFolderName

                        if os.path.isdir(path_chck):
                            pass
                        else:
                            os.makedirs(path_chck)

                        save_path = saver.save(sess, chckPtsFolderName + "Original_Experiment_" + str(exp) + '_' + save_dir + save_prefix + str(epoch) + ".ckpt")
                        # print('network data has been saved to path: ' + str(save_path))
                        remaining_seconds = (max_iters - epoch) * duration + max_iters * (cross_validation_times - 1 - exp) * duration
                        m, s = divmod(remaining_seconds, 60)
                        h, m = divmod(m, 60)
                        d, h = divmod(h, 24)
                        this_remaining_seconds = (max_iters - epoch) * duration
                        this_m, this_s = divmod(this_remaining_seconds, 60)
                        this_h, this_m = divmod(this_m, 60)
                        this_d, this_h = divmod(this_h, 24)
                        print('the remaining time is %02d:%02d:%02d:%02d, the remaining time for this crossvalidation is %02d:%02d:%02d:%02d'
                              % (d, h, m, s, this_d, this_h, this_m, this_s))
                        acc_temp = []
                        for i in range(test_num):
                            summary_test, acc = evaluate()
                            acc_temp.append(acc)

                        acc_temp = np.asarray(acc_temp)
                        mean_acc_temp = np.mean(acc_temp)
                        std_acc_temp = np.std(acc_temp)

                        print('the mean test accuracy is %.5f' %(mean_acc_temp))

                        mean_acc.append(mean_acc_temp)
                        std_acc.append(std_acc_temp)

                        summary_writer_test.add_summary(summary_test, epoch)

            sess.close()
            path_cv = accuracyFolderName

            if os.path.isdir(path_cv):
                pass
            else:
                os.makedirs(path_cv)
            np.save(path_cv + 'test_acc_experiment_' + str(exp), mean_acc)
            np.save(path_cv + 'test_err_experiment_' + str(exp), std_acc)