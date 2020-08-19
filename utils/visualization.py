import os
import numpy as np
import io_util
import sys
sys.path.append('./')

FILE_PATH = '/data/kuaishou/lugenhe/data/logs/Cos-Area2-bilinear-0905_224147/44_test/test_results'
OUT_PATH = '/data/kuaishou/lugenhe/data/logs/Cos-Area2-bilinear-0905_224147/44_test/vis/'
if not os.path.exists(OUT_PATH): os.mkdir(OUT_PATH)

for _, _, fl in os.walk(FILE_PATH):
	FILE_LIST = fl
	break
PRED = [x for x in FILE_LIST if '_gt.txt' not in x]
GT = [x.strip('pred.txt')+'gt.txt' for x in PRED ]

def main():
	for i in range(len(PRED)):
		name = PRED[i].strip('pred.txt')
		pred_data = np.loadtxt(os.path.join(FILE_PATH, PRED[i]))
		xyz = pred_data[:, 0:3]
		pred_sem = pred_data[:, -2].reshape(-1)
		pred_ins = pred_data[:, -1].reshape(-1)

		gt_data = np.loadtxt(os.path.join(FILE_PATH, GT[i]))
		gt_sem = gt_data[:, -2].reshape(-1)
		gt_ins = gt_data[:, -1].reshape(-1)

		io_util.write_label_ply(xyz, pred_ins.astype(int), OUT_PATH+name+'insPred.ply')
		io_util.write_label_ply(xyz, pred_sem.astype(int), OUT_PATH+name+'semPred.ply')
		io_util.write_label_ply(xyz, gt_ins.astype(int), OUT_PATH+name+'insGt.ply')
		io_util.write_label_ply(xyz, gt_sem.astype(int), OUT_PATH+name+'semGt.ply')
		io_util.write_color_ply(pred_data[:,0:6], OUT_PATH+name+'rgb.ply')

def main_single():
	file_name = 'Area_6_office_24_pred.txt'
	name = file_name.strip('pred.txt')
	pred_data = np.loadtxt(os.path.join(FILE_PATH, file_name))
	xyz = pred_data[:, 0:3]
	pred_sem = pred_data[:, -2].reshape(-1)
	pred_ins = pred_data[:, -1].reshape(-1)

	gt_data = np.loadtxt(os.path.join(FILE_PATH, GT[PRED.index(file_name)]))
	gt_sem = gt_data[:, -2].reshape(-1)
	gt_ins = gt_data[:, -1].reshape(-1)

	io_util.write_label_ply(xyz, pred_ins.astype(int), OUT_PATH+name+'insPred.ply')
	io_util.write_label_ply(xyz, pred_sem.astype(int), OUT_PATH+name+'semPred.ply')
	io_util.write_label_ply(xyz, gt_ins.astype(int), OUT_PATH+name+'insGt.ply')
	io_util.write_label_ply(xyz, gt_sem.astype(int), OUT_PATH+name+'semGt.ply')
	io_util.write_color_ply(pred_data[:,0:6], OUT_PATH+name+'rgb.ply')

if __name__ == '__main__':
	main()