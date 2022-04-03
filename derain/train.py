# -*- coding: utf-8 -*-

import time, os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from GuidedFilter import guided_filter

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################
tf.reset_default_graph()
##################### Network parameters ###################################
num_feature = 512  # number of feature maps
num_channels = 3  # number of input's channels
patch_size = 64  # patch size
############################################################################

# randomly select image patches
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    rainy = tf.cast(image_decoded, tf.float32) / 255.0

    image_string = tf.read_file(label)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    label = tf.cast(image_decoded, tf.float32) / 255.0

    t = time.time()
    rainy = tf.random_crop(rainy, [patch_size, patch_size, 3], seed=t)  # randomly select patch
    label = tf.random_crop(label, [patch_size, patch_size, 3], seed=t)
    return rainy, label


# DerainNet
def inference(images):
    with tf.variable_scope('DerainNet', reuse=tf.AUTO_REUSE):
        base = guided_filter(images, images, 15, 1, nhwc=True)  # using guided filter for obtaining base layer
        detail = images - base  # detail layer

        conv1 = tf.layers.conv2d(detail, num_feature, 16, padding="valid", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, num_feature, 1, padding="valid", activation=tf.nn.relu)
        output = tf.layers.conv2d_transpose(conv2, num_channels, 8, strides=1, padding="valid")

    return output, base

