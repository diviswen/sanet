import os
import sys
import numpy as np
import tensorflow as tf
from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from utils import loupe as lp
from utils import tf_util
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate
from tf_ops.emd import tf_auctionmatch
from externals.structural_losses.tf_approxmatch import approx_match, match_cost
from externals.structural_losses.tf_nndistance import nn_distance


def create_2d_grid(size):
    col_idx = np.tile(np.arange(size), [size]).reshape([size, size])
    row_idx = col_idx.transpose()
    col_idx = np.expand_dims(col_idx, axis=-1)
    row_idx = np.expand_dims(row_idx, axis=-1)
    grid_2d = np.concatenate([row_idx, col_idx], axis=-1)
    #grid_2d = tf.constant(grid_2d, dtype=tf.float32)  # (16, 16, 2)
    return grid_2d


def add_2d_grid(x, grid_2d, h, w, scope):
    """(batch_size, npoint, channel) -> (batch_size, npoint, channel+2)"""
    grid = np.array([[i,j] for i in range(0,32,32/h) for j in range(0,64,64/w)]).reshape(h,w,2)
    grid = tf.constant(grid, dtype=tf.float32)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        batch_size = x.get_shape()[0].value
        grids = tf.tile(tf.reshape(grid, [1, -1, 2]), [batch_size, 1, 1])
        result = tf.concat([x, grids], axis=-1)
        return result

def add_2d_grid_tile(x, grid_2d, h, w, scope):
    """(batch_size, npoint, channel) -> (batch_size, npoint, channel+2)"""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        batch_size = x.get_shape()[0].value
        grids = tf.tile(tf.reshape(grid_2d, [1, h*w, 2]), [batch_size, 1, 1])
        result = tf.concat([x, grids], axis=-1)
        return result

def down_sample(x, ratio, scope, bn_decay, weight_decay, is_training):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        batch_size = x.get_shape()[0].value
        npoint = x.get_shape()[1].value
        ch = x.get_shape()[2].value
        x = tf.reshape(x, [batch_size, int(npoint / ratio), -1])
        x = tf_util.conv1d(x, ch, kernel_size=1, stride=1, scope='downsample_conv', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
    return x

def up_sample_folding(x, ratio, grid_2d, h, w, scope, bn_decay, weight_decay, is_training):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        batch_size = x.get_shape()[0].value
        npoint = x.get_shape()[1].value
        ch = x.get_shape()[2].value

        x = tf.tile(x, [1, ratio, 1])
        x = add_2d_grid(x, grid_2d, h, w, scope=('add_grid_%sx%s' % (h, w)))
        x_3 = tf_util.conv1d(x, ch, kernel_size=1, stride=1, scope='upsample_conv3', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        x_3 = tf_util.conv1d(x, ch/2, kernel_size=1, stride=1, scope='upsample_conv2', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        x_3 = tf_util.conv1d(x, 3, kernel_size=1, stride=1, scope='upsample_conv0', bn=False, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay, activation_fn=None)
        tf.add_to_collection('foldings', x_3)
        x = tf.concat([x,x_3], -1)
        x = self_attention(x, scope='upsample_self_attention', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
        x = tf_util.conv1d(x, ch, kernel_size=1, stride=1, scope='upsample_conv1', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        return x

def up_sample_folding_tile(x, ratio, grid_2d, h, w, scope, bn_decay, weight_decay, is_training):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        batch_size = x.get_shape()[0].value
        npoint = x.get_shape()[1].value
        ch = x.get_shape()[2].value

        x = tf.tile(tf.reshape(x, [batch_size,npoint,1,ch]), [1,1, ratio, 1])
        x = tf.reshape(x, [batch_size,npoint*ratio,ch])
        x = add_2d_grid_tile(x, grid_2d, h, w, scope=('add_grid_%sx%s' % (h, w)))
        x_3 = tf_util.conv1d(x, ch, kernel_size=1, stride=1, scope='upsample_conv3', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        x_3 = tf_util.conv1d(x, ch/2, kernel_size=1, stride=1, scope='upsample_conv2', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        x_3 = tf_util.conv1d(x, 3, kernel_size=1, stride=1, scope='upsample_conv0', bn=False, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay, activation_fn=None)
        tf.add_to_collection('foldings', x_3)
        x = tf.concat([x,x_3], -1)
        x = self_attention(x, scope='upsample_self_attention', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
        x = tf_util.conv1d(x, ch, kernel_size=1, stride=1, scope='upsample_conv', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        return x

def up_down_up_folding(x, ratio, grid_2d, h, w, scope, bn_decay, weight_decay, is_training):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        x_up = up_sample_folding(x, ratio, grid_2d, h, w, scope='up_sample_x', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
        x_up_down = down_sample(x_up, ratio, scope='down_sample', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
        delta = x_up_down - x
        delta_up = up_sample_folding(delta, ratio, grid_2d, h, w, scope='up_sample_delta', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        x_up = x_up + delta_up * gamma
        return x_up

def up_down_up_folding_tile(x, ratio, grid_2d, h, w, scope, bn_decay, weight_decay, is_training):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        x_up = up_sample_folding_tile(x, ratio, grid_2d, h, w, scope='up_sample_x', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
        x_up_down = down_sample(x_up, ratio, scope='down_sample', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
        delta = x_up_down - x
        delta_up = up_sample_folding_tile(delta, ratio, grid_2d, h, w, scope='up_sample_delta', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        x_up = x_up + delta_up * gamma
        return x_up

def self_attention(x, scope, bn_decay, weight_decay, is_training):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        batch_size = x.get_shape()[0].value
        npoint = x.get_shape()[1].value
        ch = x.get_shape()[2].value

        f = tf_util.conv1d(x, ch//8, kernel_size=1, stride=1, scope='f_conv', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay, activation_fn=None)
        g = tf_util.conv1d(x, ch//8, kernel_size=1, stride=1, scope='g_conv', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay, activation_fn=None)
        h = tf_util.conv1d(x, ch, kernel_size=1, stride=1, scope='h_conv', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)

        s = tf.matmul(g, f, transpose_b=True)  # (batch_size, npoint, npoint)
        attention_map = tf.nn.softmax(s)
        o = tf.matmul(attention_map, h) # (batch_size, npoint, ch)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = o * gamma + x
        return o

def get_emd_loss(output, truth):
    match = approx_match(output, truth)
    emd_loss = tf.reduce_mean(match_cost(output, truth, match))
    return emd_loss

def get_chamfer_loss(output, truth):
    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(output, truth)
    chamfer_loss = tf.reduce_mean(tf.reduce_sum(cost_p1_p2, -1) + tf.reduce_sum(cost_p2_p1, -1))
    return chamfer_loss

def get_eval_chamfer_loss(output, truth):
    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(output, truth)
    chamfer_loss = tf.reduce_sum(cost_p1_p2, -1) + tf.reduce_sum(cost_p2_p1, -1)
    return chamfer_loss

def get_chamfer_loss_pcn(output, truth):
    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(output, truth)
    cost_p1_p2 = tf.reduce_mean(tf.sqrt(cost_p1_p2))
    cost_p2_p1 = tf.reduce_mean(tf.sqrt(cost_p2_p1))
    return (cost_p1_p2 + cost_p2_p1) / 2


def get_repulsion_loss(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss


def create_loss(output, truth):
    # return tf.reduce_mean(tf.square(output-truth))
    match = approx_match(output, truth)
    repulsion_loss = get_repulsion_loss(output)
    emd_loss = tf.reduce_mean(match_cost(output, truth, match))

    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(output, truth)
    chamfer_loss = tf.reduce_mean(tf.reduce_sum(cost_p1_p2, -1) + tf.reduce_sum(cost_p2_p1, -1))
    return emd_loss, chamfer_loss, repulsion_loss


def create_loss_local(output, truth, loss_type='emd'):
    # return tf.reduce_mean(tf.square(output-truth))
    output = tf.reshape(output,[-1,output.get_shape()[2],output.get_shape()[3]])
    truth = tf.reshape(truth,[-1,truth.get_shape()[2],truth.get_shape()[3]])
    if loss_type == 'emd':
        match = approx_match(output, truth)
        return tf.reduce_mean(match_cost(output, truth, match))
    else:
        cost_p1_p2, _, cost_p2_p1, _ = nn_distance(output, truth)
        return tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)


def pre_load_checkpoint(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        return epoch_step,ckpt.model_checkpoint_path
    else:
        return 0,None


def get_emd_completion_loss(pred, gt, radius=1):
    """ pred: BxNxC,
        label: BxN, """
    npoint = gt.get_shape()[1].value
    pred = tf.reshape(pred,[-1,gt.get_shape()[2],gt.get_shape()[3]])
    gt = tf.reshape(gt,[-1,gt.get_shape()[2],gt.get_shape()[3]])
    batch_size = gt.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = gather_point(gt, matchl_out)
    dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    emd_loss = tf.reduce_mean(dist)
    return emd_loss


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointnet_sa_module_decay(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training,
                       bn_decay, weight_decay, scope, bn=True, ibn=False, pooling='max', tnet_spec=None, knn=False, use_xyz=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            batch_radius: the size of each object
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn=knn)
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.layer_conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, ibn=ibn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay, weight_decay=weight_decay)
        if pooling=='avg':
            new_points = tf.layers.average_pooling2d(new_points, [1,nsample], [1,1], padding='VALID', name='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1', reuse=tf.AUTO_REUSE):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf.layers.max_pooling2d(-1 * new_points, [1, nsample], [1, 1], padding='VALID',name='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf.layers.max_pooling2d(new_points, [1,nsample], [1,1], padding='VALID', name='maxpool1')
            max_points = tf.layers.average_pooling2d(new_points, [1,nsample],[1,1], padding='VALID', name='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)
            
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.layer_conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, ibn=ibn,is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay, weight_decay=weight_decay)
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1], data_format=data_format,
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

def attention(x, y, scope, bn_decay, weight_decay, is_training):
    """
    Args:
        x: (batch_size, npoint1, ch1)
        y: (batch_size, npoint2, ch2)
    Returns:
        o: (batch_size, npoint2, ch2)
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        batch_size = x.get_shape()[0].value
        npoint1 = x.get_shape()[1].value
        ch1 = x.get_shape()[2].value
        npoint2 = y.get_shape()[1].value
        ch2 = y.get_shape()[2].value
        dims = int(min(ch1, ch2)//4)

        bi_ins = tf.matmul(y, tf.transpose(x, perm=[0, 2, 1]))
        cos_y = tf.multiply(y, y)
        cos_y = tf.reduce_sum(cos_y, 2, keep_dims=True)
        cos_y = tf.sqrt(cos_y)
        cos_y = tf.add(cos_y, 1e-7)

        cos_x = tf.multiply(x, x)
        cos_x = tf.reduce_sum(cos_x, 2, keep_dims=True)
        cos_x = tf.sqrt(cos_x)
        cos_x = tf.add(cos_x, 1e-7)
        dis = tf.matmul(cos_y, tf.transpose(cos_x, perm=[0, 2, 1]))

        bi_ins = tf.div(bi_ins, dis)

        o = tf.matmul(bi_ins, x)
        o = tf_util.conv1d(o, ch2, kernel_size=1, stride=1, scope='x_conv', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = o * gamma + y

        return o