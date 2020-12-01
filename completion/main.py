import os
import sys
import time
import math
from datetime import datetime
import argparse
import importlib
import random
import transforms3d
import numpy as np
import tensorflow as tf
import data_provider as dp
sys.path.append('../')
sys.path.append('../utils')
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sanet', help='Model name [default: model_l2h]')
parser.add_argument('--log_dir', default='logs', help='Log dir [default: logs]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 400]')
parser.add_argument('--min_epoch', type=int, default=0, help='Epoch from which training starts [default: 0]')
parser.add_argument('--batch_size', type=int, default=18, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=238158*8, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--weight_decay', type=float, default=0.0007, help='Weight decay [default: 0.007]')
parser.add_argument('--warmup_step', type=int, default=1000, help='Warm up step for lr [default: 200000]')
parser.add_argument('--gamma_cd', type=float, default=10.0, help='Gamma for chamfer loss [default: 10.0]')
parser.add_argument('--restore', default='None', help='Restore path [default: None]')
parser.add_argument('--augment_scale', type=float, default=0.0, help='Random scale, range 1.0/scale ~ scale [default: 0.0]')
parser.add_argument('--augment_rotate', action='store_true')
parser.add_argument('--augment_mirror', action='store_true')
parser.add_argument('--augment_jitter', action='store_true')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
MIN_EPOCH = FLAGS.min_epoch
NUM_POINT = FLAGS.num_point
NUM_POINT_GT = 2048
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


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

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

cat_list = ['lamp']#['plane','car','lamp','chair','table','cabinet','watercraft','sofa']

TRAIN_DATASET = []
TRAIN_DATASET_GT = []
TEST_DATASET = []
TEST_DATASET_GT = []
TEST_DATASET_LABEL = []

for idx, cat in enumerate(cat_list):
    DATA_PATH = os.path.join('../data/shapenet_completion', cat)
    TRAIN_DATASET_, TRAIN_DATASET_GT_, TEST_DATASET_, TEST_DATASET_GT_ = dp.load_completion_data(DATA_PATH, BATCH_SIZE, encode[cat], npoint=NUM_POINT)
    TRAIN_DATASET.append(TRAIN_DATASET_)
    TRAIN_DATASET_GT.append(TRAIN_DATASET_GT_)
    TEST_DATASET.append(TEST_DATASET_)
    TEST_DATASET_GT.append(TEST_DATASET_GT_)
    TEST_DATASET_LABEL+=[idx for _ in range(TEST_DATASET_GT_.shape[0])]

TRAIN_DATASET=np.concatenate(TRAIN_DATASET)
TRAIN_DATASET_GT=np.concatenate(TRAIN_DATASET_GT)
TEST_DATASET=np.concatenate(TEST_DATASET)
TEST_DATASET_GT=np.concatenate(TEST_DATASET_GT)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_RESULT_FOUT = open(os.path.join(LOG_DIR, 'log_result.csv'), 'w')
LOG_RESULT_FOUT.write('total_loss,emd_loss,repulsion_loss,chamfer_loss,l2_reg_loss,lr\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


log_string(str(FLAGS))
log_string('TRAIN_DATASET: ' + str(TRAIN_DATASET.shape))
log_string('TEST_DATASET: ' + str(TEST_DATASET.shape))


def shuffle_dataset():
    data = np.reshape(TRAIN_DATASET, [-1, NUM_POINT, 3])
    gt = np.reshape(TRAIN_DATASET_GT, [-1, NUM_POINT_GT, 3])
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx, ...]
    gt = gt[idx, ...]
    return np.reshape(data, (-1, BATCH_SIZE, NUM_POINT, 3)), np.reshape(gt, (-1, BATCH_SIZE, NUM_POINT_GT, 3))


def augment_batch_data(batch_data, point_gt):
    new_batch_data = np.zeros_like(batch_data)
    new_point_gt = np.zeros_like(point_gt)
    for i in range(batch_data.shape[0]):
        M = transforms3d.zooms.zfdir2mat(1)
        if AUGMENT_SCALE > 1.0:
            s = random.uniform(1.0/AUGMENT_SCALE, AUGMENT_SCALE)
            M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
        if AUGMENT_ROTATE:
            angle = random.uniform(0, 2*math.pi)
            M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M)
        if AUGMENT_MIRROR:
            if random.random() < 0.3:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
            if random.random() < 0.3:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)
        new_batch_data[i] = np.dot(batch_data[i], M.T)
        new_point_gt[i] = np.dot(point_gt[i], M.T)
        if AUGMENT_JITTER:
            sigma, clip = 0.01, 0.05
            new_batch_data[i] += np.clip(sigma * np.random.randn(*new_batch_data[i].shape), -1*clip, clip).astype(np.float64)
            new_point_gt[i] += np.clip(sigma * np.random.randn(*new_point_gt[i].shape), -1*clip, clip).astype(np.float64)
    return new_batch_data, new_point_gt


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

            pointclouds_pred = MODEL.get_model(pointclouds_pl, is_training, bn_decay, WEIGHT_DECAY)
            emd_loss, repulsion_loss, chamfer_loss = MODEL.get_loss(pointclouds_pred, pointclouds_gt)
            chamfer_loss = chamfer_loss * GAMMA_CD
            tf.summary.scalar('emd_loss', emd_loss)
            tf.summary.scalar('repulsion_loss', repulsion_loss)
            tf.summary.scalar('chamfer_loss', chamfer_loss)
            #tf.add_to_collection('losses', emd_loss)
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
               'train_op': train_op,
               'merged': merged,
               'lr': learning_rate,
               'step': batch}
        min_emd = 999999.9
        min_cd = 999999.9
        min_emd_epoch = 0
        min_cd_epoch = 0

        #eval_one_epoch(sess, ops, test_writer)
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
            log_string('min emd epoch: %d, emd = %f, min cd epoch: %d, cd = %f\n' % (min_emd_epoch, min_emd, min_cd_epoch, min_cd))

def train_one_epoch(sess, ops, train_writer):
    is_training = True
    log_string(str(datetime.now()))

    TRAIN_DATASET, TRAIN_DATASET_GT = shuffle_dataset()
    total_batch = TRAIN_DATASET.shape[0]
    emd_loss_total_sum = 0.
    total_loss_total_sum = 0.
    repulsion_loss_total_sum = 0.
    chamfer_loss_total_sum = 0.
    l2_reg_loss_total_sum = 0.

    for i in range(total_batch):
        batch_input_data = TRAIN_DATASET[i]
        batch_data_gt = TRAIN_DATASET_GT[i]
        if AUGMENT:
            batch_input_data, batch_data_gt = augment_batch_data(batch_input_data, batch_data_gt)

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
        if i == 0:
            log_string('learning_rate: %.7f' % lr)
        print('progress: %4d / %4d | chamfer_loss: %.3f | %s \n' % (i, total_batch, chamfer_loss/2.048, LOG_DIR)),
        sys.stdout.flush()

    mean_emd_loss = emd_loss_total_sum / total_batch
    mean_total_loss = total_loss_total_sum / total_batch
    mean_repulsion_loss = repulsion_loss_total_sum / total_batch
    mean_chamfer_loss = chamfer_loss_total_sum / total_batch
    mean_l2_reg_loss = l2_reg_loss_total_sum / total_batch
    log_string('train total loss: %.2f, train emd loss: %.2f, train repulsion loss: %.4f, train chamfer loss: %.3f, train l2 reg loss: %.3f' % \
               (mean_total_loss, mean_emd_loss, mean_repulsion_loss, mean_chamfer_loss, mean_l2_reg_loss))


def eval_one_epoch(sess, ops, test_writer):
    is_training = False
    total_batch = TEST_DATASET.shape[0]
    emd_loss_sum = 0.
    total_loss_sum = 0.
    repulsion_loss_sum = 0.
    chamfer_loss_sum = 0.
    l2_reg_loss_sum = 0.

    per_class_cd = [0. for i in range(8)]
    per_class_batch = [0. for i in range(8)]

    for i in range(total_batch):
        batch_input_data = TEST_DATASET[i]
        batch_data_gt = TEST_DATASET_GT[i]
        class_id = TEST_DATASET_LABEL[i]

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

        per_class_cd[class_id]+=chamfer_loss
        per_class_batch[class_id]+=1.

    per_class_str = ''
    for i in range(len(cat_list)):
        #per_class_cd[i] = per_class_cd[i]/per_class_batch[i]
        per_class_str += '%s: %.2f | '%(cat_list[i], per_class_cd[i]/per_class_batch[i]/2.048)


    mean_emd_loss = emd_loss_sum / total_batch
    mean_total_loss = total_loss_sum / total_batch
    mean_repulsion_loss = repulsion_loss_sum / total_batch
    mean_chamfer_loss = chamfer_loss_sum / total_batch
    mean_l2_reg_loss = l2_reg_loss_sum / total_batch
    log_string('eval  total loss: %.2f, eval  emd loss: %.2f, eval  repulsion loss: %.4f, eval  chamfer loss: %.3f, eval  l2 reg loss: %.3f' % \
               (mean_total_loss, mean_emd_loss, mean_repulsion_loss, mean_chamfer_loss, mean_l2_reg_loss))
    log_string(per_class_str)
    LOG_RESULT_FOUT.write('%.2f,%.2f,%.4f,%.3f,%.3f,%8f\n' % (mean_total_loss, mean_emd_loss, mean_repulsion_loss, mean_chamfer_loss, mean_l2_reg_loss, lr))
    LOG_RESULT_FOUT.flush()
    return mean_emd_loss, mean_chamfer_loss


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    tf.set_random_seed(int(time.time()))
    train()
    LOG_FOUT.close()
    LOG_RESULT_FOUT.close()
