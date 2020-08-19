import sys
sys.path.append('../')
sys.path.append('../utils')
import numpy as np
import h5py
import time
import Queue
import threading
import cv2
from utils import show3d
import os
import glob



def load_data(path, batch_size, categories, npoint=2048):
    encode = {
        "chair": "03001627",
        "table": "04379243",
        "sofa": "04256520",
        "cabinet": "02933112",
        "lamp": "03636649",
        "car": "02958343",
        "plane": "02691156",
        "watercraft": "04530566"
    }
    
    train_dataset = []
    train_dataset_gt = []
    test_dataset = []
    test_dataset_gt = []
    test_dataset_category = []
    for ct in categories:
        ct_path = os.path.join(path, ct)
        TRAIN_DATASET, TRAIN_DATASET_GT, TEST_DATASET, TEST_DATASET_GT = load_completion_data(ct_path, batch_size, encode[ct], npoint=npoint)
        train_dataset.append(TRAIN_DATASET)
        train_dataset_gt.append(TRAIN_DATASET_GT)
        test_dataset.append(TEST_DATASET)
        test_dataset_gt.append(TEST_DATASET_GT)
        for i in range(TEST_DATASET.shape[0]):
            test_dataset_category.append(ct)
    train_dataset = np.concatenate(train_dataset, axis=0)
    train_dataset_gt = np.concatenate(train_dataset_gt, axis=0)
    test_dataset = np.concatenate(test_dataset, axis=0)
    test_dataset_gt = np.concatenate(test_dataset_gt, axis=0)
    return train_dataset, train_dataset_gt, test_dataset, test_dataset_gt, test_dataset_category

def load_completion_data(path, batch_size, encode, npoint=2048):
    save_path_train = os.path.join(path,"train")
    save_path_gt = os.path.join(path,"gt_pro")
    f_lidar = glob.glob(os.path.join(save_path_train, '*.npy'))
    new_dataset = []
    gt_dataset = []
    test_dataset = []
    test_dataset_gt = []
    a = np.loadtxt('test_models.txt',str)
    b = []
    for i in a:
        if int(i[:8]) == int(encode):
            i = i[9:]
            b.append(i)
    for i in f_lidar:
        raw_lidar = np.load(i)
        file = i.split('/')[-1].split('.')[0][:-5]+".ply.npy"
        gt_lidar = np.load(os.path.join(save_path_gt,file))
        if file[:-8] in b:
            test_dataset.append(raw_lidar)
            test_dataset_gt.append(gt_lidar)
        else:
            new_dataset.append(raw_lidar)
            gt_dataset.append(gt_lidar)
    new_dataset = np.array(new_dataset)
    gt_dataset = np.array(gt_dataset)
    test_dataset = np.array(test_dataset)
    test_dataset_gt = np.array(test_dataset_gt)
    batch_dataset = []
    batch_dataset_gt = []
    test_batch_dataset = []
    test_batch_dataset_gt = []
    i=0
    while i+batch_size<=new_dataset.shape[0]:
        batch_dataset.append(new_dataset[i:i+batch_size])
        batch_dataset_gt.append(gt_dataset[i:i+batch_size])
        i = i + batch_size
    i=0
    while i+batch_size<=test_dataset.shape[0]:
        test_batch_dataset.append(test_dataset[i:i+batch_size])
        test_batch_dataset_gt.append(test_dataset_gt[i:i+batch_size])
        i = i + batch_size
    batch_dataset = np.array(batch_dataset)
    batch_dataset_gt = np.array(batch_dataset_gt)
    test_batch_dataset = np.array(test_batch_dataset)
    test_batch_dataset_gt = np.array(test_batch_dataset_gt)
    return batch_dataset, batch_dataset_gt, test_batch_dataset, test_batch_dataset_gt

def normalize_point_cloud(input):
    if len(input.shape)==2:
        axis = 0
    elif len(input.shape)==3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
    input = input / furthest_distance
    return input, centroid,furthest_distance

def rotate_point_cloud_and_gt(batch_data,batch_gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        # rotation_angle = np.random.uniform(size=(3)) * 2 * np.pi
        # cosval = np.cos(rotation_angle)
        # sinval = np.sin(rotation_angle)
        # rotation_matrix = np.array([[cosval, 0, sinval],
        #                             [0, 1, 0],
        #                             [-sinval, 0, cosval]])

        batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
        if batch_data.shape[-1]>3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

        if batch_gt is not None:
            batch_gt[k, ..., 0:3]   = np.dot(batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_gt.shape[-1] > 3:
                batch_gt[k, ..., 3:] = np.dot(batch_gt[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

    return batch_data,batch_gt


def shift_point_cloud_and_gt(batch_data, batch_gt = None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] += shifts[batch_index,0:3]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] += shifts[batch_index, 0:3]

    return batch_data,batch_gt


def random_scale_point_cloud_and_gt(batch_data, batch_gt = None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] *= scales[batch_index]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] *= scales[batch_index]

    return batch_data,batch_gt,scales


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in xrange(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        batch_data[k, ...,0:3] = np.dot(batch_data[k, ...,0:3].reshape((-1, 3)), R)
        if batch_data.shape[-1]>3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), R)

    return batch_data


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data[:,:,3:] = 0
    jittered_data += batch_data
    return jittered_data


def save_pl(path, pl):
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    myfile = file(path, "w")
    point_num = pl.shape[0]
    for j in range(point_num):
        if len(pl[j])==3:
            print >> myfile, "%f %f %f" % (pl[j,0],pl[j,1],pl[j,2])
        elif len(pl[j])==6:
            print >> myfile, "%f %f %f %f %f %f" % (pl[j, 0], pl[j, 1], pl[j, 2],pl[j, 3],pl[j, 4],pl[j, 5])
            # print >> myfile, "%f %f %f %f %f %f %f" % (
            # pl[j, 0], pl[j, 1], pl[j, 2], pl[j, 3], pl[j, 4], pl[j, 5], pl[j, 2])
        elif len(pl[j])==7:
            print >> myfile, "%f %f %f %f %f %f %f" % (pl[j, 0], pl[j, 1], pl[j, 2],pl[j, 3],pl[j, 4],pl[j, 5],pl[j, 2])
    myfile.close()
    if np.random.rand()>1.9:
        show3d.showpoints(pl[:, 0:3])


def nonuniform_sampling(num, sample_num = 8000):
    sample = set()
    loc = np.random.rand()*0.8+0.1
    while(len(sample)<sample_num):
        a = int(np.random.normal(loc=loc,scale=0.3)*num)
        if a<0 or a>=num:
            continue
        sample.add(a)
    return list(sample)


if __name__ == '__main__':
    from datetime import datetime
    DATA_PATH = '../data/shapenet_completion'

    for ct in ['chair', 'table', 'sofa', 'cabinet', 'lamp', 'car', 'plane', 'watercraft']:
        print '\n\n', str(datetime.now())
        TRAIN_DATASET, TRAIN_DATASET_GT, TEST_DATASET, TEST_DATASET_GT, TEST_DATASET_CATEGORY = load_data(DATA_PATH, 16, ct, npoint=2048)
        print 'category: ', ct
        print 'train_dataset: ', TRAIN_DATASET.shape
        print 'train_dataset_gt: ', TRAIN_DATASET_GT.shape
        print 'test_dataset: ', TEST_DATASET.shape
        print 'test_dataset_gt: ', TEST_DATASET_GT.shape
        print 'test_dataset_category: ', len(TEST_DATASET_CATEGORY), TEST_DATASET_CATEGORY[0], TEST_DATASET_CATEGORY[-1]
        print str(datetime.now()), '\n\n'
