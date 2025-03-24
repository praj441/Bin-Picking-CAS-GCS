import numpy as np
import cv2
import time
import sys, os
from datetime import datetime
from math import *

sys.path.append('commons')
from baseline_grasp_algo import run_grasp_algo
from utils_gs import Parameters

data_path = '../ours_data'
out_path = data_path
w = 640 #320
h = 480 #240
param = Parameters(w,h)
param.gripper_height = int(param.mh*70)  

inputs = {'param':param}
inputs['slanted_pose_detection'] = False #False
inputs['cone_detection'] = False #False
# self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')

folder_path = data_path + '/depth_maps'
folder_path_2 = data_path + '/color_ims' 

scan_names = sorted(list(set([os.path.basename(x)[0:3] \
	for x in os.listdir(folder_path)])))

score_list = []
median_depth_map = np.loadtxt(data_path+'/median_depth_map.txt')
median_depth_map = cv2.resize(median_depth_map,(param.w,param.h))

avg_time = 0.0
# scan_names = ['000006','000008']
scenes = 1
for idx in scan_names:
	# if scenes == 51:
	# 	break
	# if scenes < 51:
	# 	scenes += 1
	# 	continue
	dmap = np.loadtxt(os.path.join(folder_path, idx)+'_depth_array.txt')
	img = cv2.imread(os.path.join(folder_path_2, idx)+'_ref_image.png')
	
	# dt = datetime.now().strftime("%d%m%Y")
	# h,w,_ = img.shape
	# param = Parameters(w,h)

	inputs['image']= img
	inputs['darray'] = dmap
	inputs['depth_image'] = None
	inputs['final_attempt'] = True
	inputs['dump_dir'] = None #out_path+'/baseline/'+ idx
	inputs['median_depth_map'] = median_depth_map
	inputs['adaptive_clusters'] = True
	inputs['num_dirs'] = 6
	inputs['gqs_score'] = None

	st = time.time()
	result = run_grasp_algo(inputs)
	if scenes > 1:
		avg_time += (time.time()-st)
		print('inference time',(time.time()-st))
		print('avg_time',avg_time/(scenes-1))

	grasp = result[8]
	grasp_score = result[0][5]
	cluster_image = result[6]
	final_image = result[7]
	score_list.append(grasp_score)
	# bmap_vis = (bmap / bmap.max())*255
	# bmap_vis_denoised = (bmap_denoised / bmap_denoised.max())*255
	# cv2.imwrite(out_path+'/bmaps/bmap{0}.jpg'.format(idx),bmap_vis)#.astype(np.uint8))
	# cv2.imwrite(out_path+'/bmaps/bmap{0}_denoised.jpg'.format(idx),bmap_vis_denoised)#.astype(np.uint8))
	if inputs['adaptive_clusters']:
		method = '/baseline_adaptive'
	else:
		method = '/baseline'
	cv2.imwrite(out_path+method+'/final_image_{0}.png'.format(idx),final_image)
	np.savetxt(out_path+method+'/grasp_{0}.txt'.format(idx),grasp)
	cv2.imwrite(out_path+method+'/cluster_{0}.png'.format(idx),cluster_image)
	np.savetxt(out_path+method+'/score_list.txt',score_list,fmt='%3d')
	# print('avg_time',avg_time/scenes)
	print('acc',np.count_nonzero(score_list)/scenes)
	scenes += 1
avg_time = avg_time/len(scan_names)
	# pc_cloud = np.loadtxt(os.path.join(data_path, idx)+'_pc.npy')
	# c = input('ruko. analyze karo.')

