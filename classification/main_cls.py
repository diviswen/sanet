import os
import sys
import time
import math
from datetime import datetime
import argparse
import importlib
import random
import numpy as np
import tensorflow as tf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../')
sys.path.append('../utils')
import modelnet_dataset
import modelnet_h5_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sanet_cls', help='Model name [default: model_l2h]')
parser.add_argument('--log_dir', default='logs', help='Log dir [default: logs]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 400]')
parser.add_argument('--min_epoch', type=int, default=0, help='Epoch from which training starts [default: 0]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay [default: 0.007]')
parser.add_argument('--warmup_step', type=int, default=1000, help='Warm up step for lr [default: 200000]')
parser.add_argument('--gamma_cd', type=float, default=10.0, help='Gamma for chamfer loss [default: 10.0]')
parser.add_argument('--restore', default='/data/wenxin/pointnet2/logs/model_l2h_cls_1110-002741-modelnet40-tilegrid/checkpoints', help='Restore path [default: None]')
parser.add_argument('--augment_scale', type=float, default=0.0, help='Random scale, range 1.0/scale ~ scale [default: 0.0]')
parser.add_argument('--augment_rotate', default=False, action='store_true')
parser.add_argument('--augment_mirror', default=False, action='store_true')
parser.add_argument('--augment_jitter', default=False, action='store_true')
parser.add_argument('--modelnet10', action='store_true', default=False, help='Whether to use normal information')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
MIN_EPOCH = FLAGS.min_epoch
NUM_POINT = FLAGS.num_point
NUM_POINT_GT = 1024
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
WEIGHT_DECAY = FLAGS.weight_decay
if WEIGHT_DECAY <= 0.:
    WEIGHT_DECAY = None
WARMUP_STEP = float(FLAGS.warmup_step)
GAMMA_CD = FLAGS.gamma_cd
MODEL = importlib.import_module(FLAGS.model)
TIME = time.strftime("%m%d-%H%M%S", time.localtime())
MODEL_FILE = FLAGS.model
MODEL_NAME = '%s_%s' % (FLAGS.model, TIME)
LOG_DIR = os.path.join(FLAGS.log_dir, MODEL_NAME)
RESTORE_PATH = FLAGS.restore
AUGMENT_SCALE = FLAGS.augment_scale
AUGMENT_ROTATE = FLAGS.augment_rotate
AUGMENT_MIRROR = FLAGS.augment_mirror
AUGMENT_JITTER = FLAGS.augment_jitter
AUGMENT = (AUGMENT_SCALE>1.0) or AUGMENT_ROTATE or AUGMENT_MIRROR or AUGMENT_JITTER

BN_INIT_DECAY = 0.1
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

if FLAGS.modelnet10:
    assert(NUM_POINT<=10000)
    DATA_PATH = '../data/modelnet40_normal_resampled'
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=False, modelnet10=FLAGS.modelnet10, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=False, modelnet10=FLAGS.modelnet10, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset('../data/modelnet40_ply_hdf5_2048/train_files.txt', batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset('../data/modelnet40_ply_hdf5_2048/test_files.txt', batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_RESULT_FOUT = open(os.path.join(LOG_DIR, 'log_result.csv'), 'w')
LOG_RESULT_FOUT.write('total_loss,emd_loss,repulsion_loss,chamfer_loss,l2_reg_loss,lr\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


log_string(str(FLAGS))


def get_learning_rate(batch):
    lr_wu = batch * BATCH_SIZE / WARMUP_STEP * BASE_LEARNING_RATE
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE / DECAY_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.minimum(learning_rate, lr_wu)
    learning_rate = tf.maximum(learning_rate, 0.000001) # CLIP THE LEARNING RATE!
    return learning_rate
    #return tf.constant(0.001)

def get_learning_rate_wo_warmup(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.000001) # CLIP THE LEARNING RATE!
    return learning_rate        
    #return tf.constant(0.001)



def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            pointclouds_pl, pointclouds_gt, is_training = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_POINT_GT)
            batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            pointclouds_pred, global_feats = MODEL.get_model(pointclouds_pl, is_training, bn_decay, WEIGHT_DECAY)
            emd_loss, repulsion_loss, chamfer_loss = MODEL.get_loss(pointclouds_pred, pointclouds_gt)
            chamfer_loss = chamfer_loss * GAMMA_CD
            tf.summary.scalar('emd_loss', emd_loss)
            tf.summary.scalar('repulsion_loss', repulsion_loss)
            tf.summary.scalar('chamfer_loss', chamfer_loss)
            tf.add_to_collection('losses', emd_loss)
            tf.add_to_collection('losses', repulsion_loss)
            tf.add_to_collection('losses', chamfer_loss)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            l2_reg_loss = total_loss - emd_loss - repulsion_loss - chamfer_loss
            tf.summary.scalar('l2_reg_loss', l2_reg_loss)

            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
            updata_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updata_ops):
                train_op = optimizer.minimize(total_loss, global_step=batch)

            saver = tf.train.Saver(max_to_keep=300)
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        ckpt_state = tf.train.get_checkpoint_state(RESTORE_PATH)
        if ckpt_state is not None:
            LOAD_MODEL_FILE = os.path.join(RESTORE_PATH, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, LOAD_MODEL_FILE)
            log_string('Model loaded in file: %s' % LOAD_MODEL_FILE)
        else:
            log_string('Failed to load model file: %s' % RESTORE_PATH)
            init = tf.global_variables_initializer()
            sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_gt': pointclouds_gt,
               'is_training': is_training,
               'pointclouds_pred': pointclouds_pred,
               'emd_loss': emd_loss,
               'repulsion_loss': repulsion_loss,
               'chamfer_loss': chamfer_loss,
               'l2_reg_loss': l2_reg_loss,
               'total_loss': total_loss,
               'learning_rate': learning_rate,
               'global_feats': global_feats,
               'train_op': train_op,
               'merged': merged,
               'lr': learning_rate,
               'step': batch}
        min_emd = 999999.9
        min_cd = 999999.9
        min_emd_epoch = 0
        min_cd_epoch = 0
        #test(sess, ops)
        for epoch in range(MIN_EPOCH, MAX_EPOCH):
            log_string('**** EPOCH %03d ****  %s' % (epoch, FLAGS.model))
            train_one_epoch(sess, ops, train_writer)
            emd_loss_i, cd_loss_i = eval_one_epoch(sess, ops, test_writer)
            if emd_loss_i < min_emd:
                min_emd = emd_loss_i
                min_emd_epoch = epoch
                save_path = saver.save(sess, os.path.join(LOG_DIR, "checkpoints", "min_emd.ckpt"))
                log_string("Model saved in file: %s" % save_path)
            if cd_loss_i < min_cd:
                min_cd = cd_loss_i
                min_cd_epoch = epoch
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'checkpoints', 'min_cd.ckpt'))
                log_string('Model saved in file: %s' % save_path)
            test(sess, ops, epoch)
            log_string('min emd epoch: %d, emd = %f, min cd epoch: %d, cd = %f\n' % (min_emd_epoch, min_emd, min_cd_epoch, min_cd))

def train_one_epoch(sess, ops, train_writer):
    is_training = True
    log_string(str(datetime.now()))

    batch_input_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    batch_data_gt = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    emd_loss_total_sum = 0.
    total_loss_total_sum = 0.
    repulsion_loss_total_sum = 0.
    chamfer_loss_total_sum = 0.
    l2_reg_loss_total_sum = 0.
    idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=False)
        #if AUGMENT:
        #    batch_data, _ = augment_batch_data(batch_data, batch_data)

        bsize = batch_data.shape[0]
        batch_input_data[0:bsize,...] = batch_data
        batch_data_gt[0:bsize,...] = batch_data



        feed_dict = {
            ops['pointclouds_pl']: batch_input_data[:, :, 0:3],
            ops['pointclouds_gt']: batch_data_gt[:, :, 0:3],
            ops['is_training']: is_training
        }
        lr, summary, step, l2_reg_loss, emd_loss, total_loss, pred_val, repulsion_loss, chamfer_loss, _ = sess.run([ops['learning_rate'], ops['merged'], ops['step'], ops['l2_reg_loss'],
            ops['emd_loss'], ops['total_loss'], ops['pointclouds_pred'], ops['repulsion_loss'], ops['chamfer_loss'], ops['train_op']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        emd_loss_total_sum += emd_loss
        total_loss_total_sum += total_loss
        repulsion_loss_total_sum += repulsion_loss
        chamfer_loss_total_sum += chamfer_loss
        l2_reg_loss_total_sum += l2_reg_loss
        if idx == 0:
            log_string('learning_rate: %.7f' % lr)
        print('progress: %4d\r' % (idx)),
        idx += 1
        sys.stdout.flush()

    TRAIN_DATASET.reset()

    mean_emd_loss = emd_loss_total_sum / idx
    mean_total_loss = total_loss_total_sum / idx
    mean_repulsion_loss = repulsion_loss_total_sum / idx
    mean_chamfer_loss = chamfer_loss_total_sum / idx
    mean_l2_reg_loss = l2_reg_loss_total_sum / idx
    log_string('train total loss: %.2f, train emd loss: %.2f, train repulsion loss: %.4f, train chamfer loss: %.3f, train l2 reg loss: %.3f' % \
               (mean_total_loss, mean_emd_loss, mean_repulsion_loss, mean_chamfer_loss, mean_l2_reg_loss))


def eval_one_epoch(sess, ops, test_writer):
    is_training = False
    batch_input_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    batch_data_gt = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))

    emd_loss_sum = 0.
    total_loss_sum = 0.
    repulsion_loss_sum = 0.
    chamfer_loss_sum = 0.
    l2_reg_loss_sum = 0.
    idx = 0
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        batch_input_data[0:bsize,...] = batch_data
        batch_data_gt[0:bsize,...] = batch_data


        feed_dict = {
            ops['pointclouds_pl']: batch_input_data[:, :, 0:3],
            ops['pointclouds_gt']: batch_data_gt[:, :, 0:3],
            ops['is_training']: is_training
        }
        summary, lr, step, l2_reg_loss, emd_loss, total_loss, repulsion_loss, chamfer_loss, pred_val = sess.run([ops['merged'], ops['lr'], ops['step'], ops['l2_reg_loss'],
             ops['emd_loss'], ops['total_loss'], ops['repulsion_loss'], ops['chamfer_loss'], ops['pointclouds_pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        emd_loss_sum += emd_loss
        total_loss_sum += total_loss
        repulsion_loss_sum += repulsion_loss
        chamfer_loss_sum += chamfer_loss
        l2_reg_loss_sum += l2_reg_loss
        idx += 1
    
    mean_emd_loss = emd_loss_sum / idx
    mean_total_loss = total_loss_sum / idx
    mean_repulsion_loss = repulsion_loss_sum / idx
    mean_chamfer_loss = chamfer_loss_sum / idx
    mean_l2_reg_loss = l2_reg_loss_sum / idx
    log_string('eval  total loss: %.2f, eval  emd loss: %.2f, eval  repulsion loss: %.4f, eval  chamfer loss: %.3f, eval  l2 reg loss: %.3f' % \
               (mean_total_loss, mean_emd_loss, mean_repulsion_loss, mean_chamfer_loss, mean_l2_reg_loss))
    LOG_RESULT_FOUT.write('%.2f,%.2f,%.4f,%.3f,%.3f,%8f\n' % (mean_total_loss, mean_emd_loss, mean_repulsion_loss, mean_chamfer_loss, mean_l2_reg_loss, lr))
    LOG_RESULT_FOUT.flush()

    TEST_DATASET.reset()

    return mean_emd_loss, mean_chamfer_loss


def test(sess, ops, epoch):
    Train_Set = []
    Train_label = []
    Test_Set = []
    Test_label = []
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    is_training = False
    index = 0
    while TRAIN_DATASET.has_next_batch():
        print('Train: %d\r'%(index)),
        sys.stdout.flush()
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['is_training']: is_training}
        feature = sess.run(ops['global_feats'], feed_dict=feed_dict)
        feature = np.array(feature, dtype=np.float32)
        label = np.eye(40)[batch_label]#[1 if k == batch_label else 0 for k in range(40)]
        Train_Set.append(feature[0:bsize, ...])
        Train_label.append(label)
        index = index + 1
    TRAIN_DATASET.reset()
    Train_Set = np.concatenate(Train_Set, 0)
    Train_label = np.concatenate(Train_label, 0)
    print(Train_Set.shape)
    np.save('./Npy_Features/modelnet40_train_feats_%d.npy'%(epoch), Train_Set)
    np.save('./Npy_Features/modelnet40_train_label_%d.npy'%(epoch), Train_label)
    
    index = 0
    while TEST_DATASET.has_next_batch():
        print('Test: %d\r'%(index)),
        sys.stdout.flush()
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['is_training']: is_training}

        feature = sess.run(ops['global_feats'], feed_dict=feed_dict)
        label = np.eye(40)[batch_label]#[1 if k == batch_label else 0 for k in range(40)]
        Test_Set.append(feature[0:bsize, ...])
        Test_label.append(label)
        index = index + 1
    TEST_DATASET.reset()
    Test_Set = np.concatenate(Test_Set, 0)
    Test_label = np.concatenate(Test_label, 0)
    print(Test_label.shape)
    np.save('./Npy_Features/modelnet40_test_feats_%d.npy'%(epoch), Test_Set)
    np.save('./Npy_Features/modelnet40_test_label_%d.npy'%(epoch), Test_label)

def SVM():
    train_data = np.load('./Npy_Features/modelnet10_train_feats.npy')
    train_label = np.load('./Npy_Features/modelnet10_train_label.npy')
    test_data = np.load('./Npy_Features/modelnet10_test_feats.npy')
    test_label = np.load('./Npy_Features/modelnet10_test_label.npy')

    train_label = np.argmax(train_label, axis=1)
    test_label = np.argmax(test_label, axis=1)
    clf = OneVsRestClassifier(LinearSVC(C = 0.01, random_state = 0)).fit(train_data, train_label)
    train_pred = clf.predict(train_data)
    test_pred = clf.predict(test_data)
    print(np.mean(train_label == train_pred))
    print(np.mean(test_label == test_pred))


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    tf.set_random_seed(int(time.time()))
    train()
    LOG_FOUT.close()
    LOG_RESULT_FOUT.close()
