# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

import time
from datetime import datetime
import os

import numpy
from six.moves import xrange
import tensorflow as tf

from tensorflow import gfile

import cv2
from PIL import Image

from real_nvp_utils import (
    batch_norm, batch_norm_log_diff, conv_layer,
    squeeze_2x2, squeeze_2x2_ordered, standard_normal_ll,
    standard_normal_sample, unsqueeze_2x2, variable_on_cpu)

if True:
    tf.flags.DEFINE_string("master", "local",
                        "BNS name of the TensorFlow master, or local.")

    tf.flags.DEFINE_integer("recursion_type", 2,
                            "Type of the recursion.")

    tf.flags.DEFINE_integer("image_size", 64,
                            "Size of the input image.")

    tf.flags.DEFINE_string(
        "hpconfig", "",
        "A comma separated list of hyperparameters for the model. Format is "
        "hp1=value1,hp2=value2,etc. If this FLAG is set, the model will be trained "
        "with the specified hyperparameters, filling in missing hyperparameters "
        "from the default_values in |hyper_params|.")

tf.flags.DEFINE_string("traindir", "./tmp/real_nvp_celeba/train",
                       "Directory to which stores ckpt.")
tf.flags.DEFINE_string('GPU_device', "0", "gpu devices")

tf.flags.DEFINE_string("logdir", "./tmp/real_nvp_celeba/train",
                       "Directory to which writes logs.")

tf.flags.DEFINE_string('img', "", "input image")

tf.flags.DEFINE_integer("train_steps", 150,
                        "Number of steps to train for.")  

tf.flags.DEFINE_integer("reparameter", 0, "reparameterization")

tf.flags.DEFINE_integer("mask", 0, "mask or not")
tf.flags.DEFINE_float("mask_scale", 0., "mask scale or not")
tf.flags.DEFINE_float("lam1", 0., "lam1")
tf.flags.DEFINE_float("lam2", 0., "lam2")

tf.flags.DEFINE_integer("mid", 0, "add noise at the middle layer")
tf.flags.DEFINE_integer("mid-type", 0, "mid-type: 0 - only first split; 1 - all split")

FLAGS = tf.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_device
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class HParams(object):
    """Dictionary of hyperparameters."""
    def __init__(self, **kwargs):
        self.dict_ = kwargs
        self.__dict__.update(self.dict_)

    def update_config(self, in_string):
        """Update the dictionary with a comma separated list."""
        pairs = in_string.split(",")
        pairs = [pair.split("=") for pair in pairs]
        for key, val in pairs:
            self.dict_[key] = type(self.dict_[key])(val)
        self.__dict__.update(self.dict_)
        return self

    def __getitem__(self, key):
        return self.dict_[key]

    def __setitem__(self, key, val):
        self.dict_[key] = val
        self.__dict__.update(self.dict_)


def get_default_hparams():
    """Get the default hyperparameters."""
    return HParams(
        batch_size=64,
        residual_blocks=2,
        n_couplings=2,
        n_scale=5,
        learning_rate=0.001,
        momentum=1e-1,
        decay=1e-3,
        l2_coeff=0.00005,
        clip_gradient=100.,
        optimizer="adam",
        dropout_mask=0,
        base_dim=32,
        bottleneck=0,
        use_batch_norm=1,
        alternate=1,
        use_aff=1,
        skip=1,
        data_constraint=.9,
        n_opt=0)


# RESNET UTILS
def residual_block(input_, dim, name, use_batch_norm=True,
                   train=True, weight_norm=True, bottleneck=False):
    """Residual convolutional block."""
    with tf.variable_scope(name):
        res = input_
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=dim, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
        res = tf.nn.relu(res)
        if bottleneck:
            res = conv_layer(
                input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                name="h_0", stddev=numpy.sqrt(2. / (dim)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=(not use_batch_norm),
                weight_norm=weight_norm, scale=False)
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim,
                    name="bn_0", scale=False, train=train,
                    epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim,
                dim_out=dim, name="h_1", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME", nonlinearity=None,
                bias=(not use_batch_norm),
                weight_norm=weight_norm, scale=False)
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim, name="bn_1", scale=False,
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME", nonlinearity=None,
                bias=True, weight_norm=weight_norm, scale=True)
        else:
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim, dim_out=dim,
                name="h_0", stddev=numpy.sqrt(2. / (dim)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=(not use_batch_norm),
                weight_norm=weight_norm, scale=False)
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim, name="bn_0", scale=False,
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim, dim_out=dim,
                name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME", nonlinearity=None,
                bias=True, weight_norm=weight_norm, scale=True)
        res += input_

    return res


def resnet(input_, dim_in, dim, dim_out, name, use_batch_norm=True,
           train=True, weight_norm=True, residual_blocks=5,
           bottleneck=False, skip=True):
    """Residual convolutional network."""
    with tf.variable_scope(name):
        res = input_
        if residual_blocks != 0:
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim_in, dim_out=dim,
                name="h_in", stddev=numpy.sqrt(2. / (dim_in)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=True,
                weight_norm=weight_norm, scale=False)
            if skip:
                out = conv_layer(
                    input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                    name="skip_in", stddev=numpy.sqrt(2. / (dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=True,
                    weight_norm=weight_norm, scale=True)

            # residual blocks
            for idx_block in xrange(residual_blocks):
                res = residual_block(res, dim, "block_%d" % idx_block,
                                     use_batch_norm=use_batch_norm, train=train,
                                     weight_norm=weight_norm,
                                     bottleneck=bottleneck)
                if skip:
                    out += conv_layer(
                        input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                        name="skip_%d" % idx_block, stddev=numpy.sqrt(2. / (dim)),
                        strides=[1, 1, 1, 1], padding="SAME",
                        nonlinearity=None, bias=True,
                        weight_norm=weight_norm, scale=True)
            # outputs
            if skip:
                res = out
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim, name="bn_pre_out", scale=False,
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[1, 1], dim_in=dim,
                dim_out=dim_out,
                name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=True,
                weight_norm=weight_norm, scale=True)
        else:
            if bottleneck:
                res = conv_layer(
                    input_=res, filter_size=[1, 1], dim_in=dim_in, dim_out=dim,
                    name="h_0", stddev=numpy.sqrt(2. / (dim_in)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=(not use_batch_norm),
                    weight_norm=weight_norm, scale=False)
                if use_batch_norm:
                    res = batch_norm(
                        input_=res, dim=dim, name="bn_0", scale=False,
                        train=train, epsilon=1e-4, axes=[0, 1, 2])
                res = tf.nn.relu(res)
                res = conv_layer(
                    input_=res, filter_size=[3, 3], dim_in=dim,
                    dim_out=dim, name="h_1", stddev=numpy.sqrt(2. / (1. * dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None,
                    bias=(not use_batch_norm),
                    weight_norm=weight_norm, scale=False)
                if use_batch_norm:
                    res = batch_norm(
                        input_=res, dim=dim, name="bn_1", scale=False,
                        train=train, epsilon=1e-4, axes=[0, 1, 2])
                res = tf.nn.relu(res)
                res = conv_layer(
                    input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim_out,
                    name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=True,
                    weight_norm=weight_norm, scale=True)
            else:
                res = conv_layer(
                    input_=res, filter_size=[3, 3], dim_in=dim_in, dim_out=dim,
                    name="h_0", stddev=numpy.sqrt(2. / (dim_in)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=(not use_batch_norm),
                    weight_norm=weight_norm, scale=False)
                if use_batch_norm:
                    res = batch_norm(
                        input_=res, dim=dim, name="bn_0", scale=False,
                        train=train, epsilon=1e-4, axes=[0, 1, 2])
                res = tf.nn.relu(res)
                res = conv_layer(
                    input_=res, filter_size=[3, 3], dim_in=dim, dim_out=dim_out,
                    name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=True,
                    weight_norm=weight_norm, scale=True)
        return res


# COUPLING LAYERS
# masked convolution implementations
def masked_conv_aff_coupling(input_, mask_in, dim, name,
                             use_batch_norm=True, train=True, weight_norm=True,
                             reverse=False, residual_blocks=5,
                             bottleneck=False, use_width=1., use_height=1.,
                             mask_channel=0., skip=True):
    """Affine coupling with masked convolution."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # build mask
        mask = use_width * numpy.arange(width)
        mask = use_height * numpy.arange(height).reshape((-1, 1)) + mask
        mask = mask.astype("float32")
        mask = tf.mod(mask_in + mask, 2)
        mask = tf.reshape(mask, [-1, height, width, 1])
        if mask.get_shape().as_list()[0] == 1:
            mask = tf.tile(mask, [batch_size, 1, 1, 1])
        res = input_ * tf.mod(mask_channel + mask, 2)

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
            res *= 2.
        res = tf.concat([res, -res], 3)
        res = tf.concat([res, mask], 3)
        dim_in = 2. * channels + 1
        res = tf.nn.relu(res)
        res = resnet(input_=res, dim_in=dim_in, dim=dim,
                     dim_out=2 * channels,
                     name="resnet", use_batch_norm=use_batch_norm,
                     train=train, weight_norm=weight_norm,
                     residual_blocks=residual_blocks,
                     bottleneck=bottleneck, skip=skip)
        mask = tf.mod(mask_channel + mask, 2)
        res = tf.split(axis=3, num_or_size_splits=2, value=res)
        shift, log_rescaling = res[-2], res[-1]
        scale = variable_on_cpu(
            "rescaling_scale", [],
            tf.constant_initializer(0.))
        shift = tf.reshape(
            shift, [batch_size, height, width, channels])
        log_rescaling = tf.reshape(
            log_rescaling, [batch_size, height, width, channels])
        log_rescaling = scale * tf.tanh(log_rescaling)
        if not use_batch_norm:
            scale_shift = variable_on_cpu(
                "scale_shift", [],
                tf.constant_initializer(0.))
            log_rescaling += scale_shift
        shift *= (1. - mask)
        log_rescaling *= (1. - mask)
        if reverse:
            res = input_
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask), dim=channels, name="bn_out",
                    train=False, epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var * (1. - mask))
                res += mean * (1. - mask)
            res *= tf.exp(-log_rescaling)
            res -= shift
            log_diff = -log_rescaling
            if use_batch_norm:
                log_diff += .5 * log_var * (1. - mask)
        else:
            res = input_
            res += shift
            res *= tf.exp(log_rescaling)
            log_diff = log_rescaling
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask), dim=channels, name="bn_out",
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean * (1. - mask)
                res *= tf.exp(-.5 * log_var * (1. - mask))
                log_diff -= .5 * log_var * (1. - mask)

    return res, log_diff


def masked_conv_add_coupling(input_, mask_in, dim, name,
                             use_batch_norm=True, train=True, weight_norm=True,
                             reverse=False, residual_blocks=5,
                             bottleneck=False, use_width=1., use_height=1.,
                             mask_channel=0., skip=True):
    """Additive coupling with masked convolution."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # build mask
        mask = use_width * numpy.arange(width)
        mask = use_height * numpy.arange(height).reshape((-1, 1)) + mask
        mask = mask.astype("float32")
        mask = tf.mod(mask_in + mask, 2)
        mask = tf.reshape(mask, [-1, height, width, 1])
        if mask.get_shape().as_list()[0] == 1:
            mask = tf.tile(mask, [batch_size, 1, 1, 1])
        res = input_ * tf.mod(mask_channel + mask, 2)

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
            res *= 2.
        res = tf.concat([res, -res], 3)
        res = tf.concat([res, mask], 3)
        dim_in = 2. * channels + 1
        res = tf.nn.relu(res)
        shift = resnet(input_=res, dim_in=dim_in, dim=dim, dim_out=channels,
                       name="resnet", use_batch_norm=use_batch_norm,
                       train=train, weight_norm=weight_norm,
                       residual_blocks=residual_blocks,
                       bottleneck=bottleneck, skip=skip)
        mask = tf.mod(mask_channel + mask, 2)
        shift *= (1. - mask)
        # use_batch_norm = False
        if reverse:
            res = input_
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask),
                    dim=channels, name="bn_out", train=False, epsilon=1e-4)
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var * (1. - mask))
                res += mean * (1. - mask)
            res -= shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                log_diff += .5 * log_var * (1. - mask)
        else:
            res = input_
            res += shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask), dim=channels,
                    name="bn_out", train=train, epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean * (1. - mask)
                res *= tf.exp(-.5 * log_var * (1. - mask))
                log_diff -= .5 * log_var * (1. - mask)

    return res, log_diff


def masked_conv_coupling(input_, mask_in, dim, name,
                         use_batch_norm=True, train=True, weight_norm=True,
                         reverse=False, residual_blocks=5,
                         bottleneck=False, use_aff=True,
                         use_width=1., use_height=1.,
                         mask_channel=0., skip=True):
    """Coupling with masked convolution."""
    if use_aff:
        return masked_conv_aff_coupling(
            input_=input_, mask_in=mask_in, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, use_width=use_width, use_height=use_height,
            mask_channel=mask_channel, skip=skip)
    else:
        return masked_conv_add_coupling(
            input_=input_, mask_in=mask_in, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, use_width=use_width, use_height=use_height,
            mask_channel=mask_channel, skip=skip)


# channel-axis splitting implementations
def conv_ch_aff_coupling(input_, dim, name,
                         use_batch_norm=True, train=True, weight_norm=True,
                         reverse=False, residual_blocks=5,
                         bottleneck=False, change_bottom=True, skip=True):
    """Affine coupling with channel-wise splitting."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()

        if change_bottom:
            input_, canvas = tf.split(axis=3, num_or_size_splits=2, value=input_)
        else:
            canvas, input_ = tf.split(axis=3, num_or_size_splits=2, value=input_)
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        res = input_

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
        res = tf.concat([res, -res], 3)
        dim_in = 2. * channels
        res = tf.nn.relu(res)
        res = resnet(input_=res, dim_in=dim_in, dim=dim, dim_out=2 * channels,
                     name="resnet", use_batch_norm=use_batch_norm,
                     train=train, weight_norm=weight_norm,
                     residual_blocks=residual_blocks,
                     bottleneck=bottleneck, skip=skip)
        shift, log_rescaling = tf.split(axis=3, num_or_size_splits=2, value=res)
        scale = variable_on_cpu(
            "scale", [],
            tf.constant_initializer(1.))
        shift = tf.reshape(
            shift, [batch_size, height, width, channels])
        log_rescaling = tf.reshape(
            log_rescaling, [batch_size, height, width, channels])
        log_rescaling = scale * tf.tanh(log_rescaling)
        if not use_batch_norm:
            scale_shift = variable_on_cpu(
                "scale_shift", [],
                tf.constant_initializer(0.))
            log_rescaling += scale_shift
        if reverse:
            res = canvas
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res, dim=channels, name="bn_out", train=False,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var)
                res += mean
            res *= tf.exp(-log_rescaling)
            res -= shift
            log_diff = -log_rescaling
            if use_batch_norm:
                log_diff += .5 * log_var
        else:
            res = canvas
            res += shift
            res *= tf.exp(log_rescaling)
            log_diff = log_rescaling
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res, dim=channels, name="bn_out", train=train,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean
                res *= tf.exp(-.5 * log_var)
                log_diff -= .5 * log_var
        if change_bottom:
            res = tf.concat([input_, res], 3)
            log_diff = tf.concat([tf.zeros_like(log_diff), log_diff], 3)
        else:
            res = tf.concat([res, input_], 3)
            log_diff = tf.concat([log_diff, tf.zeros_like(log_diff)], 3)

    return res, log_diff


def conv_ch_add_coupling(input_, dim, name,
                         use_batch_norm=True, train=True, weight_norm=True,
                         reverse=False, residual_blocks=5,
                         bottleneck=False, change_bottom=True, skip=True):
    """Additive coupling with channel-wise splitting."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()

        if change_bottom:
            input_, canvas = tf.split(axis=3, num_or_size_splits=2, value=input_)
        else:
            canvas, input_ = tf.split(axis=3, num_or_size_splits=2, value=input_)
        shape = input_.get_shape().as_list()
        channels = shape[3]
        res = input_

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
        res = tf.concat([res, -res], 3)
        dim_in = 2. * channels
        res = tf.nn.relu(res)
        shift = resnet(input_=res, dim_in=dim_in, dim=dim, dim_out=channels,
                       name="resnet", use_batch_norm=use_batch_norm,
                       train=train, weight_norm=weight_norm,
                       residual_blocks=residual_blocks,
                       bottleneck=bottleneck, skip=skip)
        if reverse:
            res = canvas
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res, dim=channels, name="bn_out", train=False,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var)
                res += mean
            res -= shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                log_diff += .5 * log_var
        else:
            res = canvas
            res += shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res, dim=channels, name="bn_out", train=train,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean
                res *= tf.exp(-.5 * log_var)
                log_diff -= .5 * log_var
        if change_bottom:
            res = tf.concat([input_, res], 3)
            log_diff = tf.concat([tf.zeros_like(log_diff), log_diff], 3)
        else:
            res = tf.concat([res, input_], 3)
            log_diff = tf.concat([log_diff, tf.zeros_like(log_diff)], 3)

    return res, log_diff


def conv_ch_coupling(input_, dim, name,
                     use_batch_norm=True, train=True, weight_norm=True,
                     reverse=False, residual_blocks=5,
                     bottleneck=False, use_aff=True, change_bottom=True,
                     skip=True):
    """Coupling with channel-wise splitting."""
    if use_aff:
        return conv_ch_aff_coupling(
            input_=input_, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, change_bottom=change_bottom, skip=skip)
    else:
        return conv_ch_add_coupling(
            input_=input_, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, change_bottom=change_bottom, skip=skip)


# RECURSIVE USE OF COUPLING LAYERS
def rec_masked_conv_coupling(input_, hps, scale_idx, n_scale,
                             use_batch_norm=True, weight_norm=True,
                             train=True, mid=False, b=None, mask_1=None):
    """Recursion on coupling layers."""
    shape = input_.get_shape().as_list()
    channels = shape[3]
    residual_blocks = hps.residual_blocks
    base_dim = hps.base_dim
    mask = 1.
    use_aff = hps.use_aff
    res = input_
    skip = hps.skip
    log_diff = tf.zeros_like(input_)
    dim = base_dim
    if FLAGS.recursion_type < 4:
        dim *= 2 ** scale_idx
    with tf.variable_scope("scale_%d" % scale_idx):
        # initial coupling layers
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=mask, dim=dim,
            name="coupling_0",
            use_batch_norm=use_batch_norm, train=train,
            weight_norm=weight_norm,
            reverse=False, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=1. - mask, dim=dim,
            name="coupling_1",
            use_batch_norm=use_batch_norm, train=train,
            weight_norm=weight_norm,
            reverse=False, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=mask, dim=dim,
            name="coupling_2",
            use_batch_norm=use_batch_norm, train=train,
            weight_norm=weight_norm,
            reverse=False, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=True,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
    if scale_idx < (n_scale - 1):
        with tf.variable_scope("scale_%d" % scale_idx):
            res = squeeze_2x2(res)
            log_diff = squeeze_2x2(log_diff)
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=True, dim=2 * dim,
                name="coupling_4",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=False, dim=2 * dim,
                name="coupling_5",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=True, dim=2 * dim,
                name="coupling_6",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True, skip=skip)
            log_diff += inc_log_diff
            res = unsqueeze_2x2(res)
            log_diff = unsqueeze_2x2(log_diff)
        if FLAGS.recursion_type > 1:



            if scale_idx == 0 and mid:
                print("Add here: ", train)
                # res += b * mask_1
                res += b
            # print(res.shape)
            print("Before Split")
            print(res.shape)


            res = squeeze_2x2_ordered(res)
            log_diff = squeeze_2x2_ordered(log_diff)
            if FLAGS.recursion_type > 2:
                res_1 = res[:, :, :, :channels]
                res_2 = res[:, :, :, channels:]
                log_diff_1 = log_diff[:, :, :, :channels]
                log_diff_2 = log_diff[:, :, :, channels:]
            else:
                res_1, res_2 = tf.split(axis=3, num_or_size_splits=2, value=res)
                log_diff_1, log_diff_2 = tf.split(axis=3, num_or_size_splits=2, value=log_diff)
            res_1, inc_log_diff = rec_masked_conv_coupling(
                input_=res_1, hps=hps, scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            res = tf.concat([res_1, res_2], 3)
            log_diff_1 += inc_log_diff
            log_diff = tf.concat([log_diff_1, log_diff_2], 3)
            res = squeeze_2x2_ordered(res, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
        else:
            res = squeeze_2x2_ordered(res)
            log_diff = squeeze_2x2_ordered(log_diff)
            res, inc_log_diff = rec_masked_conv_coupling(
                input_=res, hps=hps, scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            log_diff += inc_log_diff
            res = squeeze_2x2_ordered(res, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
    else:
        with tf.variable_scope("scale_%d" % scale_idx):
            res, inc_log_diff = masked_conv_coupling(
                input_=res,
                mask_in=1. - mask, dim=dim,
                name="coupling_3",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True,
                use_width=1., use_height=1., skip=skip)
            log_diff += inc_log_diff
    return res, log_diff


def rec_masked_deconv_coupling(input_, hps, scale_idx, n_scale,
                               use_batch_norm=True, weight_norm=True,
                               train=True):
    """Recursion on inverting coupling layers."""
    shape = input_.get_shape().as_list()
    channels = shape[3]
    residual_blocks = hps.residual_blocks
    base_dim = hps.base_dim
    mask = 1.
    use_aff = hps.use_aff
    res = input_
    log_diff = tf.zeros_like(input_)
    skip = hps.skip
    dim = base_dim
    if FLAGS.recursion_type < 4:
        dim *= 2 ** scale_idx
    if scale_idx < (n_scale - 1):
        if FLAGS.recursion_type > 1:
            res = squeeze_2x2_ordered(res)
            log_diff = squeeze_2x2_ordered(log_diff)
            if FLAGS.recursion_type > 2:
                res_1 = res[:, :, :, :channels]
                res_2 = res[:, :, :, channels:]
                log_diff_1 = log_diff[:, :, :, :channels]
                log_diff_2 = log_diff[:, :, :, channels:]
            else:
                res_1, res_2 = tf.split(axis=3, num_or_size_splits=2, value=res)
                log_diff_1, log_diff_2 = tf.split(axis=3, num_or_size_splits=2, value=log_diff)
            res_1, log_diff_1 = rec_masked_deconv_coupling(
                input_=res_1, hps=hps,
                scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            res = tf.concat([res_1, res_2], 3)
            log_diff = tf.concat([log_diff_1, log_diff_2], 3)
            res = squeeze_2x2_ordered(res, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
        else:
            res = squeeze_2x2_ordered(res)
            log_diff = squeeze_2x2_ordered(log_diff)
            res, log_diff = rec_masked_deconv_coupling(
                input_=res, hps=hps,
                scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            res = squeeze_2x2_ordered(res, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
        with tf.variable_scope("scale_%d" % scale_idx):
            res = squeeze_2x2(res)
            log_diff = squeeze_2x2(log_diff)
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=True, dim=2 * dim,
                name="coupling_6",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=False, dim=2 * dim,
                name="coupling_5",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=True, dim=2 * dim,
                name="coupling_4",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res = unsqueeze_2x2(res)
            log_diff = unsqueeze_2x2(log_diff)
    else:
        with tf.variable_scope("scale_%d" % scale_idx):
            res, inc_log_diff = masked_conv_coupling(
                input_=res,
                mask_in=1. - mask, dim=dim,
                name="coupling_3",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True,
                use_width=1., use_height=1., skip=skip)
            log_diff += inc_log_diff

    with tf.variable_scope("scale_%d" % scale_idx):
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=mask, dim=dim,
            name="coupling_2",
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=True, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=True,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=1. - mask, dim=dim,
            name="coupling_1",
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=True, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=mask, dim=dim,
            name="coupling_0",
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=True, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff

    return res, log_diff


# ENCODER AND DECODER IMPLEMENTATIONS
# start the recursions
def encoder(input_, hps, n_scale, use_batch_norm=True,
            weight_norm=True, train=True, mid=False, epsilon=None, mask=None):
    """Encoding/gaussianization function."""
    res = input_
    log_diff = tf.zeros_like(input_)
    res, inc_log_diff = rec_masked_conv_coupling(
        input_=res, hps=hps, scale_idx=0, n_scale=n_scale,
        use_batch_norm=use_batch_norm, weight_norm=weight_norm,
        train=train, mid=mid, b=epsilon, mask_1=mask)
    log_diff += inc_log_diff

    return res, log_diff


def decoder(input_, hps, n_scale, use_batch_norm=True,
            weight_norm=True, train=True):
    """Decoding/generator function."""
    res, log_diff = rec_masked_deconv_coupling(
        input_=input_, hps=hps, scale_idx=0, n_scale=n_scale,
        use_batch_norm=use_batch_norm, weight_norm=weight_norm,
        train=train)

    return res, log_diff


class noise_RealNVP(object):
    """Real NVP model."""

    def __init__(self, hps, sampling=False):
        # TODO: Modify Later
        input_x = Image.open(FLAGS.img)
        image_size = FLAGS.image_size
        input_x = numpy.array(input_x).reshape(hps.batch_size, image_size, image_size, 3)
        x_orig = tf.convert_to_tensor(input_x, dtype=tf.uint8)
        x_in = tf.reshape(x_orig, [hps.batch_size, image_size, image_size, 3])
        x_orig = (tf.cast(x_orig, tf.float32)) / 255. # [0, 1]
        x_in = (tf.cast(x_in, tf.float32)) / 255. # [0,1]

        # Add trainable Varibale b and Annealing:
        self.annealing = 1.
        self.learning_rate = hps.learning_rate
        with tf.variable_scope("new_epsilon", reuse=True):
            if not FLAGS.reparameter:
                trained_b = tf.Variable(tf.zeros([hps.batch_size, image_size, image_size, 3]))
                b = tf.math.tanh(trained_b)
                # b = tf.math.tanh(trained_b * self.annealing)f
                self.trained_b = trained_b
                self.b = b
            else: # Reparameterization
                trained_b = tf.Variable(0.5 * tf.math.log(x_in / (1 - x_in + 1e-8)) )
                self.b = b = 0.5 * (tf.math.tanh(trained_b) + 1) - x_in

            self.s = s = tf.Variable(FLAGS.mask_scale)

            trained_mask = tf.Variable(tf.ones([hps.mask_length, hps.mask_length,]))
            self.trained_mask = trained_mask
            mask_scalar = tf.reshape(trained_mask, [-1])
            mask_scalar = tf.nn.softmax(mask_scalar)
            # mask_scalar = mask_scalar / self.annealing

            if FLAGS.mask_scale > 0:
                mask_scalar = mask_scalar * s
            else:
                mask_scalar = mask_scalar / tf.reduce_max(mask_scalar)

            mask_scalar = tf.reshape(mask_scalar, [hps.mask_length, hps.mask_length])
            mask_list = []

            for row in range(8):
                row_list = []
                for col in range(8):
                    row_list.append(tf.ones([8, 8, 3]) * mask_scalar[row, col])
                    # row_list.append(tf.ones([8, 8, 3]) * mask_scalar[row, col, :])
                mask_list.append(tf.concat(row_list, axis=0))
            cur_mask = tf.concat(mask_list, axis=1)

            self.mask = cur_mask

        # Add dx to the input
        if FLAGS.mid:
            x_in = x_in
        elif FLAGS.mask:
            x_in += b * cur_mask
        else:
            x_in += b
        x_in = tf.clip_by_value(x_in, 0, 1)
        
        # restrict the data
        if True:
            data_constraint = hps.data_constraint
            pre_logit_scale = numpy.log(data_constraint)
            pre_logit_scale -= numpy.log(1. - data_constraint)
            pre_logit_scale = tf.cast(pre_logit_scale, tf.float32)

            logit_x_in = 2. * x_in  # [0, 2]
            logit_x_in -= 1.  # [-1, 1]
            logit_x_in *= data_constraint  # [-.9, .9]
            logit_x_in += 1.  # [.1, 1.9]
            logit_x_in /= 2.  # [.05, .95]
            # logit the data
            logit_x_in = tf.log(logit_x_in) - tf.log(1. - logit_x_in)
            transform_cost = tf.reduce_sum(
                tf.nn.softplus(logit_x_in) + tf.nn.softplus(-logit_x_in)
                - tf.nn.softplus(-pre_logit_scale),
                [1, 2, 3])
            
            logit_x_orig = 2. * x_orig  # [0, 2]
            logit_x_orig -= 1.  # [-1, 1]
            logit_x_orig *= data_constraint  # [-.9, .9]
            logit_x_orig += 1.  # [.1, 1.9]
            logit_x_orig /= 2.  # [.05, .95]
            # logit the data
            logit_x_orig = tf.log(logit_x_orig) - tf.log(1. - logit_x_orig)
            # orig_transform_cost = tf.reduce_sum(
            #     tf.nn.softplus(logit_x_orig) + tf.nn.softplus(-logit_x_orig)
            #     - tf.nn.softplus(-pre_logit_scale),
            #     [1, 2, 3])

        # INFERENCE AND COSTS, only one [Train=True]
        z_out, log_diff = encoder(
            input_=logit_x_in, hps=hps, n_scale=hps.n_scale,
            use_batch_norm=hps.use_batch_norm, weight_norm=True,
            train=True, mid=FLAGS.mid, epsilon=trained_b, mask=cur_mask)

        z_out, log_diff = encoder(
            input_=logit_x_in, hps=hps, n_scale=hps.n_scale,
            use_batch_norm=hps.use_batch_norm, weight_norm=True,
            train=False, mid=FLAGS.mid, epsilon=trained_b, mask=cur_mask)

        final_shape = [image_size, image_size, 3]
        prior_ll = standard_normal_ll(z_out)
        prior_ll = tf.reduce_sum(prior_ll, [1, 2, 3])
        log_diff = tf.reduce_sum(log_diff, [1, 2, 3])
        log_diff += transform_cost
        logpx = -(prior_ll + log_diff - numpy.log(256.) * image_size * image_size * 3)

        self.x_in = x_in
        self.z_out = z_out
        self.logpx = logpx = tf.reduce_mean(logpx)
        bit_per_dim = (logpx / (image_size * image_size * 3. * numpy.log(2.)))
        self.bit_per_dim = bit_per_dim

        # Prepare loss
        if FLAGS.mask and FLAGS.mask_scale == 0:
            b_norm = tf.norm(b * cur_mask, ord=1)
        elif FLAGS.mid:
            b_norm = tf.norm(trained_b, ord=1)
        else:
            b_norm = tf.norm(b, ord=1)
        self.b_norm = b_norm

        if FLAGS.mask_scale > 0:
            loss = logpx + FLAGS.lam1 * b_norm + FLAGS.lam2 * tf.math.abs(self.s)
        else:
            loss = logpx + FLAGS.lam1 * b_norm

        self.loss = loss

        orig_z_out, orig_log_diff = encoder(
            input_=logit_x_orig, hps=hps, n_scale=hps.n_scale,
            use_batch_norm=hps.use_batch_norm, weight_norm=True,
            train=False, mid=0, epsilon=b, mask=cur_mask)
        # orig_prior_ll = standard_normal_ll(orig_z_out)
        # orig_prior_ll = tf.reduce_sum(orig_prior_ll, [1, 2, 3])
        # orig_log_diff = tf.reduce_sum(orig_log_diff, [1, 2, 3])
        # orig_log_diff += orig_transform_cost
        # orig_cost = -(orig_prior_ll + orig_log_diff - numpy.log(256.) * image_size * image_size * 3)
        # self.orig_cost = tf.reduce_mean(orig_cost)

        # OPTIMIZATION
        momentum = 1. - hps.momentum
        decay = 1. - hps.decay
        if hps.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=momentum, beta2=decay, epsilon=1e-08,
                use_locking=False, name="Adam")
        elif hps.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate, decay=decay,
                momentum=momentum, epsilon=1e-04,
                use_locking=False, name="RMSProp")
        else:
            optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                   momentum=momentum)

        if FLAGS.mask_scale > 0:
            grads_and_vars = optimizer.compute_gradients(loss, [trained_b, trained_mask, s])
        elif FLAGS.mask:
            grads_and_vars = optimizer.compute_gradients(loss, [trained_b, trained_mask])
        else:
            grads_and_vars = optimizer.compute_gradients(loss, [trained_b])
            
        grads, vars_ = zip(*grads_and_vars)

        if FLAGS.mask and FLAGS.mask_scale == 0:
            grads = (grads[0], grads[1])
            # grads = (grads[0] * self.annealing, grads[1])
        elif FLAGS.mask and FLAGS.mask_scale > 0:
            grads = (grads[0], grads[1] * 10, grads[2] * 10)

        capped_grads, gradient_norm = tf.clip_by_global_norm(
            grads, clip_norm=hps.clip_gradient)
        gradient_norm = tf.check_numerics(gradient_norm,
                                            "Gradient norm is NaN or Inf.")

        capped_grads_and_vars = zip(capped_grads, vars_)

        step = tf.get_variable(
            "global_step", [], tf.int64,
            tf.zeros_initializer(),
            trainable=False)
        self.step = step
        self.train_step = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=step)
        
        # Update learning rate and annealing:
        # self.annealing.assign( 10 ** (- 1 *self.train_step/FLAGS.train_steps))

        rec_x, _ = decoder(
                input_=z_out, hps=hps, n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=False)
        self.rec_x = tf.sigmoid(rec_x)
        orig_rec_x, _ = decoder(
                input_=orig_z_out, hps=hps, n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=False)
        self.orig_rec_x = tf.sigmoid(orig_rec_x)


if "red" in FLAGS.img: # "??red.png"
    image_name = FLAGS.img[-9:-7]
else: # "??.png"
    image_name = FLAGS.img[-6:-4]

def parse(i, b, x):
    l2b = numpy.linalg.norm(b*x, axis=2)
    l2b = numpy.uint8((l2b - l2b.min()) / (l2b.max() - l2b.min() + 1e-8) * 255)
    Image.fromarray(l2b).save(FLAGS.logdir + '/'  + image_name + '/' + str(i)+ "deltaX.png")

    # if FLAGS.mask:
    x = x[:,:,0]
    x = (x - x.min()) / (x.max() - x.min())
    heatmap = cv2.applyColorMap(numpy.uint8(x * 255), cv2.COLORMAP_JET)
    cv2.imwrite(FLAGS.logdir + "/" + image_name+ "/{}mask.png".format(i), heatmap)

def parse_mid(i, orig, rec):
    l2b = numpy.linalg.norm(rec - orig, axis=2)
    l2b = numpy.uint8((l2b - l2b.min()) / (l2b.max() - l2b.min() + 1e-8) * 255)
    Image.fromarray(l2b).save(FLAGS.logdir + '/'  + image_name + '/' + str(i)+ "deltaX.png")

    # # if FLAGS.mask:
    # x = x[:,:,0]
    # x = (x - x.min()) / (x.max() - x.min())
    # heatmap = cv2.applyColorMap(numpy.uint8(x * 255), cv2.COLORMAP_JET)
    # cv2.imwrite(FLAGS.logdir + "/" + image_name+ "/{}mask.png".format(i), heatmap)
def train_b(hps, logdir, traindir, subset="valid", return_val=False):
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(0)):
            with tf.variable_scope("model") as var_scope:
                model = noise_RealNVP(hps)

            # Load ckpt
            all_params = tf.contrib.framework.get_variables_to_restore()
            var_to_restore = [v for v in all_params if 'new_epsilon' not in v.name]
            saver = tf.train.Saver(var_to_restore)
            init = tf.global_variables_initializer()
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True)
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)
            sess.run(init)
            with sess.as_default():
                ckpt_state = tf.train.get_checkpoint_state(traindir)
                saver.restore(sess, ckpt_state.model_checkpoint_path)
                print("Loading file %s" % ckpt_state.model_checkpoint_path)
            tf.train.start_queue_runners(sess)

            # Could Train the model:
            local_step = 0
            bpd = []
            logpx = []
            loss = []
            b_norm = []
            s = []
            
            orig_rec_x = sess.run([model.orig_rec_x])[0]
            # print("usefule info", orig_rec_x.max(), orig_rec_x.min(), orig_rec_x.shape)
            
            while True:
                steps = [model.step, model.train_step]
                _ = sess.run(steps)[0]

                prob_stats = sess.run([model.logpx, model.bit_per_dim, model.loss, model.b_norm, model.s])
                logpx.append(prob_stats[0])
                bpd.append(prob_stats[1])
                loss.append(prob_stats[2])
                b_norm.append(prob_stats[3])
                s.append(prob_stats[4])

                # orig_info = [model.orig_cost]
                # pics = sess.run(orig_info)
                # orig_info = pics[0]

                if local_step % 50 == 0:
                    add_on = sess.run([model.b, model.mask, model.x_in, model.z_out, model.rec_x])
                    cur_b = add_on[0]
                    mask = add_on[1]
                    new_x = add_on[2]
                    z_out = add_on[3]
                    rec_x = add_on[4]
                    train_var = sess.run([model.trained_b, model.trained_mask])
                    trained_b = train_var[0]
                    trained_mask = train_var[1]

                    # format_str = ('step %d, logpx = %.3f, b_norm = %.3f, loss = %.3f, orig = %.3f')
                    # print(format_str % (train_step, logpx, b_norm, loss, orig_info))
                    # format_str = ('step %d, logpx = %.3f, b_norm = %.3f, loss = %.3f')
                    # print(format_str % (train_step, logpx, b_norm, loss))

                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'b.npy', cur_b[0]) # After tanh()
                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'z.npy', z_out) 
                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'after.npy', new_x)
                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'rec.npy', rec_x)
                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'trained_b.npy', trained_b)

                    if FLAGS.mask:
                        numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'mask.npy', mask) # After softmax
                        numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'trained_mask.npy', trained_mask)
                    if not FLAGS.mid:
                        parse(local_step, cur_b[0], mask)
                    else:
                        parse_mid(local_step, orig_rec_x[0], rec_x[0])

                    cur_x = numpy.uint8(numpy.clip(new_x * 255, 0, 255))
                    cur_x = Image.fromarray(cur_x[0])
                    cur_x.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step) + 'after.png')
                    rec_x = numpy.uint8(numpy.clip(rec_x * 255, 0, 255))
                    rec_x = Image.fromarray(rec_x[0])
                    rec_x.save(FLAGS.logdir + '/' + image_name + '/' + str(local_step) + 'rec.png')
                    print("s: ", prob_stats[4])

                if local_step == FLAGS.train_steps:
                    logpx = numpy.array(logpx)
                    bpd = numpy.array(bpd)
                    loss = numpy.array(loss)
                    b_norm = numpy.array(b_norm)
                    s = numpy.array(s)

                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'logpX.npy', logpx)
                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'bpd.npy', bpd)
                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'loss.npy', loss)
                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 'b_norm.npy', b_norm)
                    numpy.save(FLAGS.logdir + '/'  + image_name + '/' + str(local_step)+ 's.npy', s)
                    break
                
                local_step += 1
                # model.annealing = 10 ** (- 1 *local_step/FLAGS.train_steps)
                model.learning_rate = hps.learning_rate * (10 ** (- 1 *local_step/FLAGS.train_steps))

def main(unused_argv):
    hps = get_default_hparams().update_config(FLAGS.hpconfig)
    hps.mask_length = 8
    hps.batch_size = 1

    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)
    path = FLAGS.logdir + '/'  + image_name + '/'
    if not os.path.exists(path):
        os.mkdir(path)

    train_b(hps=hps, logdir=FLAGS.logdir,
                traindir=FLAGS.traindir, subset="eval")

if __name__ == "__main__":
    tf.app.run()

