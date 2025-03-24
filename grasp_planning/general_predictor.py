import torch
import numpy as np
import copy
import cv2
import time
import sys
sys.path.append('commons/')
from custom_grasp_planning_algorithm_dense_cas import select_a_best_grasp_pose
# from utils_cnn import draw_final_pose
from cas_grasp_algo import run_grasp_algo
from utils_gs_cas import create_directory,get_seg_mask
from utils_gs_cas import Parameters, get_pivot_heat_value

sys.path.append('../../src/DA_maskrcnn')
from inference_single import get_prediction

import matplotlib.pyplot as plt
import random

def random_colour_masks(image):
	colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
	r = np.zeros_like(image).astype(np.uint8)
	g = np.zeros_like(image).astype(np.uint8)
	b = np.zeros_like(image).astype(np.uint8)
	r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
	coloured_mask = np.stack([r, g, b], axis=2)
	return coloured_mask

def draw_and_save_seg_masks(inputs,img,masks,boxes):
	mask_img = np.zeros_like(img).astype(np.uint8)
	for i in range(len(masks)):
		rgb_mask = random_colour_masks(masks[i])
		# cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
		mask_img += rgb_mask
	img = cv2.addWeighted(img, 0.25, mask_img, 0.75, 0)
			# cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	
	out_path = inputs['path']
	cv2.imwrite(out_path+'/seg_mask_{0}.png'.format(inputs['exp_num']),img)
	# plt.savefig(out_path+'/seg_mask_{0}.png'.format(inputs['exp_num']))


class Predictor:
	def __init__(self):
		self.threshold = 0.5

	def process_depth_data(self,depth_raw):
		# depth_raw = depth_raw[:,:,0]
		depth_raw = depth_raw/255

		max_depth = 0.615

		depth_raw = depth_raw/depth_raw.max()
		depth_map_est = max_depth*depth_raw

		return depth_map_est

	def predict(self, inputs):

		img_path = inputs['image_path']
		depth_path = inputs['darray_path']

		pred = get_prediction(depth_path, img_path, threshold=self.threshold)
		masks,boxes,scores,gcs_scores = pred[0],pred[1],100*np.array(pred[2]),100*np.array(pred[4])
		dmap_est = self.process_depth_data(pred[5])

		img = cv2.imread(img_path)
		dmap = np.loadtxt(depth_path)

		draw_and_save_seg_masks(inputs,img,masks,boxes)

		scores = scores.astype(int)
		gcs_scores =  gcs_scores.astype(int)

		# selecting top m masks
		m = 5
		if m > gcs_scores.shape[0]:
			m = gcs_scores.shape[0]
		if gcs_scores is not None:
			topm = np.argpartition(gcs_scores, -m)[-m:]
		else:
			topm = np.argpartition(scores, -m)[-m:]

		inputs['seg_mask'] = get_seg_mask(masks)

		seg_mask = inputs['seg_mask']
		

		inputs['image']= img
		inputs['darray'] = dmap #dmap
		inputs['depth_image'] = None
		inputs['final_attempt'] = True
		
		inputs['adaptive_clusters'] = False
		inputs['num_dirs'] = 4
		inputs['param'].calculate_asc_indices(inputs['num_dirs'])
		inputs['slanted_pose_detection'] = False #False
		inputs['cone_detection'] = False #False

		# inputs['mask'] = masks
		inputs['divide_masks'] = True 
		# inputs['scores'] = scores

		inputs['gcs_scores'] = gcs_scores[topm]

		inputs['mask'] = masks[topm]
		inputs['scores'] = scores[topm]
		inputs['topm'] = topm
		st = time.time()

		out_path = inputs['path']

		mask_img = np.zeros_like(img).astype(np.uint8)
		
		mn = gcs_scores.min()
		mx = gcs_scores.max()
		for i in range(masks.shape[0]):
			score = gcs_scores[i]
			r,g,b = get_pivot_heat_value(score,minimum=0,maximum=mx)
			mask = masks[i]
			R = np.zeros_like(mask).astype(np.uint8)
			G = np.zeros_like(mask).astype(np.uint8)
			B = np.zeros_like(mask).astype(np.uint8)
			R[mask == 1], G[mask == 1], B[mask == 1] = r,g,b
			rgb_mask = np.stack([R, G, B], axis=2)
			mask_img += rgb_mask
		colored_gcs_map = cv2.addWeighted(img, 0.25, mask_img, 0.75, 0)
		cv2.imwrite(out_path+'/{0}_gcs_mask_colored.png'.format(inputs['exp_num']),colored_gcs_map)
		inputs['colored_gcs_map'] = colored_gcs_map


		result = run_grasp_algo(inputs)
		# if scenes > 1:
		# 	avg_time += (time.time()-st)

		valid_flag = result[3]
		grasp = result[8]
		grasp.append(valid_flag)

		
		# grasp_score = result[0][5]
		# cluster_image = result[6]
		final_image = result[7]
		# score_list.append(grasp_score)
		
		
		cv2.imwrite(out_path+'/final_image_{0}.png'.format(inputs['exp_num']),final_image)
		cv2.imwrite(out_path+'/seg_mask_{0}.png'.format(inputs['exp_num']),255*(seg_mask/seg_mask.max()))


		darr = 255*(pred[5]/pred[5].max())
		darr = darr - darr.min() + 50
		darr = 255*(darr/darr.max())
		cv2.imwrite(out_path+'/depth_est_{0}.png'.format(inputs['exp_num']),darr)

		darr = 255*(pred[5]/pred[5].max())
		darr = darr - darr.min() + 25
		darr = 255*(darr/darr.max())
		cv2.imwrite(out_path+'/depth_est_{0}.1.png'.format(inputs['exp_num']),darr)

		mask_img = np.zeros_like(img).astype(np.uint8)
		for i in range(len(masks)):
		  rgb_mask = inputs['param'].random_colour_masks(masks[i])
		  # cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
		  mask_img += rgb_mask
		colored_segmask = cv2.addWeighted(img, 0.25, mask_img, 0.75, 0)
		cv2.imwrite(out_path+'/seg_mask_colored_{0}.png'.format(inputs['exp_num']),colored_segmask)

		cv2.imwrite(out_path+'/{0}_pivot_gcs_image.png'.format(inputs['exp_num']),result[9])

		# writing cnn prediction to files

		np.save(out_path+'/{0}/masks.npy'.format(inputs['exp_num']),masks)
		np.save(out_path+'/{0}/boxes.npy'.format(inputs['exp_num']),boxes)
		np.save(out_path+'/{0}/scores.npy'.format(inputs['exp_num']),scores)
		np.save(out_path+'/{0}/gcs.npy'.format(inputs['exp_num']),gcs_scores)


		return grasp



if __name__ == '__main__':


	w = 640 #320
	h = 480 #240
	param = Parameters(w,h)
	param.THRESHOLD2 = 0.02
	#************ Hyperparameter: depth-based-collision-check (DBCC) **********
	DBCC_enable = True
	param.DBCC_enable = DBCC_enable
	#**************************************************************************

	inputs = {'param':param}
	inputs['slanted_pose_detection'] = False #False
	inputs['cone_detection'] = False #False
	inputs['run_with_dbcc'] = True

	path = '../results_rerun'

	# try:
	# exp_num = 0
	for exp_num in [0,1,11,34]:
		# except:
		# 	print('please provide exp_num as first argument!')
		# 	sys.exit(0)

		predictor = Predictor()

		img_path = path+'/{0}_ref_image.png'.format(exp_num)
		dmap_path = path+'/{0}_depth_array.txt'.format(exp_num)

		# inputs = {}
		inputs['image_path'] = img_path
		inputs['darray_path'] = dmap_path
		inputs['depth_image'] = None
		inputs['dump_dir'] = path+'/{0}'.format(exp_num)
		inputs['pc_cloud'] = None
		# inputs_np['param'] = param
		inputs['pc_arr'] = None

		inputs['path'] = path
		inputs['exp_num'] = exp_num
		inputs['mask_image'] = None
		#*************** Calling deep network here ***
		result = predictor.predict(inputs)
		#*********************************************
