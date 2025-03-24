import numpy as np
import cv2
import time
import sys, os
from datetime import datetime
from math import *
import matplotlib.pyplot as plt

root = '../'

sys.path.append('commons')
from cas_grasp_algo import run_grasp_algo
from utils_gs_cas import Parameters, create_directory, get_seg_mask



#import file paths 

data_path = root + 'ours_tase_mid_clutter_data'
# data_path = root + '/data'
folder_path = data_path + '/depth_maps_estimated'
folder_path_2 = data_path + '/color_ims' 
out_final_path = data_path + '/out_final'

# out_path = data_path
w = 640 #320
h = 480 #240
param = Parameters(w,h)
param.gripper_height = int(param.mh*70)




#************ Hyperparameter: depth-based-collision-check (DBCC) **********
DBCC_enable = True
param.DBCC_enable = DBCC_enable
if not DBCC_enable:   #IRC23 method
	param.cut_length = -1
#**************************************************************************



inputs = {'param':param}
inputs['slanted_pose_detection'] = False #False
inputs['cone_detection'] = False #False
inputs['run_with_dbcc'] = False

inputs['num_dirs'] = 4
param.calculate_asc_indices(inputs['num_dirs'])


# self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
scan_names = sorted(list(set([os.path.basename(x)[0:6] 
	for x in os.listdir(folder_path)])))
#to pick only a selected range of files to run in code 
l = 1
m = 100
# scan_names = [str(i).zfill(6) for i in range(l, m)]

score_list = []
# file_list = os.listdir(folder_path)
# for file_name in file_list:
#     if file_name.endswith('.txt'):
#         file_path = os.path.join(folder_path, file_name)
#         median_depth_map = np.loadtxt(file_path)
#         median_depth_map = cv2.resize(median_depth_map, (param.w, param.h))


method = '/da_method_testrun_noLdvd'
out_path = data_path+method
create_directory(out_path)

avg_time = 0.0
# scan_names = ['000012']
scenes = 1

for idx in scan_names:
	print('Processing the sample ',idx)
	if scenes == 101:
		break
	# if scenes < 71:
	# 	scenes += 1
	# 	continue
	dmap = np.loadtxt(os.path.join(folder_path, idx)+'_depth_array.txt')
	img = cv2.imread(os.path.join(folder_path_2, idx)+'_ref_image.png')
	mask_img = cv2.imread(out_final_path+'/seg_{0}.jpg'.format(int(idx)-1))
	
	mask_path = f"{out_final_path}/masks_{l-1}.npy"
	score_path = f"{out_final_path}/scores_{l-1}.npy"


	l=l+1 #index of masks starts from zero 
	mask = np.load(mask_path)
	scores = 100*np.load(score_path)
	gcs_scores = np.zeros(scores.shape)

	# selecting top m masks
	m = 5
	if m > scores.shape[0]:
		m = scores.shape[0]
	topm = np.argpartition(scores, -m)[-m:]
	# mask = create_a_single_mask(mask)
	#cv2.imwrite(out_path+method+'/final_image_{0}.png'.format(idx),final_image)
	
	# dt = datetime.now().strftime("%d%m%Y")
	# h,w,_ = img.shape
	# param = Parameters(w,h)
	
	inputs['seg_mask'] = get_seg_mask(mask)

	seg_mask = inputs['seg_mask']
	cv2.imwrite('seg_mask.png',255*(seg_mask/seg_mask.max()))


	inputs['image']= img
	inputs['mask_image']= mask_img
	inputs['darray'] = dmap
	inputs['depth_image'] = None
	inputs['final_attempt'] = True
	inputs['dump_dir'] = out_path + '/' + idx
	# inputs['dump_dir'] = None
	# inputs['median_depth_map'] = median_depth_map
	inputs['adaptive_clusters'] = False
	
	inputs['gcs_scores'] = gcs_scores[topm]
	inputs['mask'] = mask[topm]
	inputs['scores'] = scores[topm]
	inputs['topm'] = topm
	# inputs['divide_masks'] = True 

	st = time.time()

	#************** main function calling here ********************
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
	# if inputs['adaptive_clusters']:
	# 	method = '/baseline_adaptive'
	# else:
	# 	method = '/baseline'



	# path_final_pose = out_path + '/final_pose'
	# path_all_pose = out_path + '/final_pose'
	# path_segments = out_path + '/final_pose'
	# path_ = out_path + '/final_pose'
	# create_directory(path_final_pose)
	# create_directory(path_final_pose)
	# create_directory(path_final_pose)
	# create_directory(path_final_pose)

	cv2.imwrite(out_path+'/final_image_{0}.png'.format(idx),final_image)
	np.savetxt(out_path+'/grasp_{0}.txt'.format(idx),grasp)
	# cv2.imwrite(out_path+'/cluster_{0}.png'.format(idx),cluster_image)
	np.savetxt(out_path+'/score_list.txt',score_list,fmt='%s')
	# print('avg_time',avg_time/scenes)
	# print('acc',np.count_nonzero(score_list)/scenes)
	scenes += 1
	# c = input('ruko. analyze karo.')

avg_time = avg_time/len(scan_names)
	# pc_cloud = np.loadtxt(os.path.join(data_path, idx)+'_pc.npy')
