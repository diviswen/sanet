import numpy as np
import tensorflow as tf
import pointnet_util as pu
from utils import tf_util


def placeholder_inputs(batch_size, num_point, num_point_gt):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, num_point_gt, 3))
    is_training = tf.placeholder(tf.bool,shape=[])
    return pointclouds_pl, pointclouds_gt, is_training


grid_44_16 = np.array([[i,j] for i in range(0,32,32/4) for j in range(0,64,64/4)], dtype=np.float32)
grid_88_64 = (np.tile(grid_44_16.reshape(16,1,2), [1,4,1])+np.array([[i,j] for i in range(0,8,8/2) for j in range(0,16,16/2)], dtype=np.float32).reshape(1,4,2)).reshape(64,2)
grid_1616_256 = (np.tile(grid_88_64.reshape(64,1,2), [1,4,1])+np.array([[i,j] for i in range(0,4,4/2) for j in range(0,8,8/2)], dtype=np.float32).reshape(1,4,2)).reshape(256,2)
grid_final = (np.tile(grid_1616_256.reshape(256,1,2), [1,8,1])+np.array([[i,j] for i in range(0,2) for j in range(0,4)], dtype=np.float32).reshape(1,8,2)).reshape(2048,2)



def get_model(point_clouds, is_training, bn_decay=None, weight_decay=None):
    """
    Args:
        point_clouds: (batch_size, num_point, 3)
    Returns:
        pointclouds_pred: (batch_size, num_point, 3)
    """
    batch_size = point_clouds.get_shape()[0].value
    num_point = point_clouds.get_shape()[1].value
    grid_2d = pu.create_2d_grid(128)

    # layer0: (batch_size, 2048, 3)
    l0_xyz = point_clouds
    l0_points = None
    # layer1: (batch_size, 256, 128)
    l1_xyz, l1_points, _ = pu.pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=None, nsample=16, mlp=[64, 128], mlp2=None,
                                                 group_all=False, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay,
                                                 scope='layer1', knn=True)
    # layer2: (batch_size, 64, 256)
    l2_xyz, l2_points, _ = pu.pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=None, nsample=8, mlp=[128, 256], mlp2=None,
                                                 group_all=False, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay,
                                                 scope='layer2', knn=True)
    # layer3: (batch_size, 1, 512)
    l3_xyz, l3_points, _ = pu.pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512], mlp2=None,
                                                 group_all=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay,
                                                 scope='layer3', knn=True)

    # layer4: (batch_size, 16, 514)
    l4_points = tf.tile(l3_points, [1, 16, 1])
    x = pu.add_2d_grid_tile(l4_points, grid_44_16, 4, 4, scope='add_grid_4x4')
    x = tf_util.conv1d(x, 256, kernel_size=1, stride=1, scope='upsample_conv3', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
    x = tf_util.conv1d(x, 128, kernel_size=1, stride=1, scope='upsample_conv2', bn=True, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)
    x = tf_util.conv1d(x, 3, kernel_size=1, stride=1, scope='upsample_conv0', bn=False, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay, activation_fn=None)
    l4_points = tf.concat([l4_points,x], -1)
    lk_points = tf_util.conv1d(l4_points, 512, 1, stride=1, scope='layer5_conv', bn=True, bn_decay=bn_decay,
                               weight_decay=weight_decay, is_training=is_training)
    # Note: if using up_down_up_folding, hand-crafted grid_88_64 will be ignored. If you want to use hand-crafted grid, please call up_down_up_folding_tile.
    lk_points = pu.up_down_up_folding(lk_points, 2, grid_88_64, 4, 8, scope='layer5_up_down_up', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_training=is_training)


    # layer5: (batch_size, 128, 256)
    l5_points = tf_util.conv1d(lk_points, 256, 1, stride=1, scope='layerk_conv', bn=True, bn_decay=bn_decay,
                               weight_decay=weight_decay, is_training=is_training)
    l5_points = pu.attention(l2_points, l5_points, scope='layerk_attention', bn_decay=bn_decay,
                             weight_decay=weight_decay, is_training=is_training)
    l5_points = pu.up_down_up_folding(l5_points, 4, grid_88_64, 8, 16, scope='layerk_up_down_up', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_training=is_training)

    # layer6: (batch_size, 512, 128)
    l6_points = tf_util.conv1d(l5_points, 128, 1, stride=1, scope='layer6_conv', bn=True, bn_decay=bn_decay,
                               weight_decay=weight_decay, is_training=is_training)
    l6_points = pu.attention(l1_points, l6_points, scope='layer6_attention', bn_decay=bn_decay,
                             weight_decay=weight_decay, is_training=is_training)
    l6_points = pu.up_down_up_folding(l6_points, 4, grid_1616_256, 16, 32, scope='layer6_up_down_up', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_training=is_training)

    # layer7: (batch_size, 2048, 128)
    l7_points = tf_util.conv1d(l6_points, 128, 1, stride=1, scope='layer7_conv', bn=True, bn_decay=bn_decay,
                               weight_decay=weight_decay, is_training=is_training)
    l7_points = pu.up_down_up_folding(l7_points, 4, grid_final, 32, 64, scope='layer7_up_down_up', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_training=is_training)

    # pointclouds_pred: (batch_size, 2048, 3)
    pointclouds_pred = tf_util.conv1d(l7_points, 128, 1, stride=1, scope='pred_conv1', bn=True,
                                      bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
    pointclouds_pred = tf_util.conv1d(pointclouds_pred, 16, 1, stride=1, scope='pred_conv2', bn=True,
                                      bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
    pointclouds_pred = tf_util.conv1d(pointclouds_pred, 3, 1, stride=1, scope='pred_conv3', bn=False,
                                      weight_decay=weight_decay, is_training=is_training, activation_fn=tf.tanh)
    return pointclouds_pred


def get_loss(pointclouds_pred, pointclouds_gt):
    emd_loss = pu.get_emd_loss(pointclouds_pred, pointclouds_gt)
    repulsion_loss = pu.get_repulsion_loss(pointclouds_pred)
    chamfer_loss = pu.get_chamfer_loss(pointclouds_pred, pointclouds_gt)
    return emd_loss, repulsion_loss, chamfer_loss