import os
import sys
sys.path.append('../')
sys.path.append('../utils')
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, cls_labels_pl

NUM_CATEGORIES = 16

def get_model(point_cloud, cls_label, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.1,0.2,0.4], [16,32,64], [[32,32,64], [32,32,64], [32,32,64]], is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 256, [0.4,0.8], [64,96], [[64,64,128],[64,64,128]], is_training, bn_decay, scope='layer2')
    xyz_centers, feat_centers, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    tf.add_to_collection('xyz_centers', xyz_centers[0])

    # Feature propagation layers
    interp_l2_points = pointnet_fp_module(l2_xyz, xyz_centers, l2_points, feat_centers, [256,256], is_training, bn_decay, scope='fp_layer1')
    interp_l2_points = skip_attention(l2_points, interp_l2_points, scope='layer1_attention', bn_decay=bn_decay, weight_decay=0., is_training=is_training)
    interp_l2_points = self_attention(interp_l2_points, scope='layer1_self_attention', bn_decay=bn_decay, weight_decay=0., is_training=is_training)

    interp_l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, interp_l2_points, [256,128], is_training, bn_decay, scope='fp_layer2')
    interp_l1_points = skip_attention(l1_points, interp_l1_points, scope='layer2_attention', bn_decay=bn_decay, weight_decay=0., is_training=is_training)
    interp_l1_points = self_attention(interp_l1_points, scope='layer2_self_attention', bn_decay=bn_decay, weight_decay=0., is_training=is_training)

    cls_label_one_hot = tf.one_hot(cls_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    cls_label_one_hot = tf.reshape(cls_label_one_hot, [batch_size, 1, NUM_CATEGORIES])
    cls_label_one_hot = tf.tile(cls_label_one_hot, [1,num_point,1])
    
    interp_l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([cls_label_one_hot, l0_xyz, l0_points],axis=-1), interp_l1_points, [128,128], is_training, bn_decay, scope='fp_layer3')
    interp_l0_points = self_attention(interp_l0_points, scope='layer3_self_attention', bn_decay=bn_decay, weight_decay=0., is_training=is_training)
    net = tf_util.conv1d(interp_l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    # FC layers
    net = tf_util.conv1d(interp_l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc_out')

    return net, end_points


def gate(x, y, scope='gate'):
    with tf.variable_scope(scope):
        g = tf.layers.dense(inputs=x, units=128, activation=None, use_bias=False, name='g')
        g = tf.nn.sigmoid(g)
        return tf.add(tf.multiply(x, g), tf.multiply(tf.subtract(tf.ones_like(g), g), y))#net_sem)

def skip_attention(x, y, scope, bn_decay, weight_decay, is_training):
    """
    Args:
        x: (batch_size, npoint1, ch1)
        y: (batch_size, npoint2, ch2)
    Returns:
        o: (batch_size, npoint2, ch2)
    """
    with tf.variable_scope(scope) as sc:
        batch_size = x.get_shape()[0].value
        npoint1 = x.get_shape()[1].value
        ch1 = x.get_shape()[2].value
        npoint2 = y.get_shape()[1].value
        ch2 = y.get_shape()[2].value
        dims = int(min(ch1, ch2) / 8)
        o_list = []
        d_k = tf.sqrt(float(dims))
        for i in range(8):
            f = tf_util.conv1d(x, dims, kernel_size=1, stride=1, scope='f_conv%d'%(i), bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
            g = tf_util.conv1d(y, dims, kernel_size=1, stride=1, scope='g_conv%d'%(i), bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
            s = tf.matmul(g, f, transpose_b=True)  # (batch_size, npoint2, npoint1)
            attention_map = tf.nn.softmax(s/d_k)

            x = tf_util.conv1d(x, dims, kernel_size=1, stride=1, scope='x_conv%d'%(i), bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
            o = tf.matmul(attention_map, x)  # (batch_size, npoint2, ch2)
            o_list.append(o)
        o_list = tf.concat(o_list, axis=-1)
        o = tf_util.conv1d(o, ch2, kernel_size=1, stride=1, scope='o_conv0', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)

        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = o_list * gamma + y
        o = tf_util.conv1d(o, ch2, kernel_size=1, stride=1, scope='o_conv1', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        return o


def self_attention(x, scope, bn_decay, weight_decay, is_training):
    with tf.variable_scope(scope) as sc:
        batch_size = x.get_shape()[0].value
        npoint = x.get_shape()[1].value
        ch = x.get_shape()[2].value

        f = tf_util.conv1d(x, ch//8, kernel_size=1, stride=1, scope='f_conv', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        g = tf_util.conv1d(x, ch//8, kernel_size=1, stride=1, scope='g_conv', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        h = tf_util.conv1d(x, ch, kernel_size=1, stride=1, scope='h_conv', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)

        s = tf.matmul(g, f, transpose_b=True)  # (batch_size, npoint, npoint)
        attention_map = tf.nn.softmax(s)
        o = tf.matmul(attention_map, h) # (batch_size, npoint, ch)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = o * gamma + x
        #o = gate(o, x)
        return o

LABEL_WEIGHT = np.load('label_weight.npy')
LABEL_WEIGHT = LABEL_WEIGHT.reshape(1,-1)

def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    cls_label_one_hot = tf.one_hot(label, depth=50, on_value=1.0, off_value=0.0)
    sample_weights = tf.reduce_sum(tf.multiply(cls_label_one_hot, LABEL_WEIGHT), -1)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=label, loss_collection=None)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss



if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        cls_labels = tf.zeros((32),dtype=tf.int32)
        output, ep = get_model(inputs, cls_labels, tf.constant(True))
        print(output)
