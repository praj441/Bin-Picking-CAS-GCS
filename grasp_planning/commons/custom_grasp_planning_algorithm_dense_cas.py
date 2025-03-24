#! /usr/bin/env python 
from math import *
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
import copy
from scipy.signal import medfilt2d
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore")

manualSeed = np.random.randint(1, 10000)  # fix seed
np.random.seed(manualSeed)
cv2.setRNGSeed(manualSeed)
from sklearn.cluster import KMeans
import sys, os

# sys.path.append('../commons/')
from grasp_evaluation_cas import calculate_GDI2_Lite
from utils_gs_cas import select_best_rectangles_gdi_old_way, query_point_cloud_client, final_axis_angle
from utils_gs_cas import keep_angle_bounds, height_difference_consideration, select_best_rectangles, draw_rectified_rect
from utils_gs_cas import draw_rectified_rect_plain, normalize_gdi_score
from utils_gs_cas import Parameters, create_directory, draw_pivots_n_gcs_scores
import matplotlib.pyplot as plt
from functions_for_length_wise_cuts import cut_long_masks_along_major_axis
# parallel_module = False
# try:
# 	from joblib import Parallel, delayed
# 	parallel_module = True
# except:
# 	print('joblib not loaded.')

from joblib import Parallel, delayed
parallel_module = False

# parallel_module = False
def process_a_single_graps_pose(i,inputs,rectangle_pixels,angle):
	# print(i)
	dump_dir = inputs['dump_dir']
	if dump_dir is not None:
		np.savetxt(dump_dir+'/grasp_pose_info'+'/rectangle_{0}.txt'.format(i),rectangle_pixels)
		np.savetxt(dump_dir+'/grasp_pose_info'+'/angle_{0}.txt'.format(i),[angle-radians(180)])
	darray = inputs['darray']

	if darray[int(rectangle_pixels[0][1]),int(rectangle_pixels[0][0])]:
		# **************main function calling here****************
		bmap,gdi,gdi_plus,gdi2, bmap_denoised ,cx,cy = calculate_GDI2_Lite(inputs,rectangle_pixels,angle-radians(180))
		
		# print('**',i,gdi,gdi_plus,gdi2.surface_normal_score,gdi2.invalid_reason,'**')
		return [bmap,gdi,gdi_plus,gdi2, bmap_denoised ,cx,cy,rectangle_pixels]
	else:
		print('zero depth')
		return []

def evaluate_selected_grasp_poses_parallel(inputs):

	top_grasp_points = inputs['top_grasp_points']
	dump_dir = inputs['dump_dir']
	initial_img = inputs['image']
	# angles = inputs['angles']
	darray = inputs['darray']
	final_attempt=inputs['final_attempt']
	param = inputs['param']


	# GQS = []
	GDI = []
	GDI_plus = []
	GDI_calculator = []
	rectangle_list = []
	angle_list = []
	GDI_calculator_all = []
	rectangle_list_all = []
	angle_list_all = []
	centroid_list = []
	original_idx = []
	# directions = 4

	st = time.time()
	for k,each_point in enumerate(top_grasp_points):
		
		# rectangle_pixels_list, angle_list, centroid = param.draw_rect(centroid=each_point, angle=angles[k] + radians(90), directions=directions)
		rectangle_pixels_list, angle_list = param.draw_rect_generic_fix_angles(centroid=each_point, directions=inputs['num_dirs'])
		rectangle_list_all.extend(rectangle_pixels_list)
		angle_list_all.extend(angle_list)
		# centroid_list.extend([centroid,centroid,centroid,centroid])
	print('sampling',time.time()-st,len(rectangle_list_all))	

	st = time.time()
	results = Parallel(n_jobs=10)(delayed(process_a_single_graps_pose)(i,inputs,rectangle_pixels,angle_list_all[i]) for i,rectangle_pixels in enumerate(rectangle_list_all))
	print('processing',time.time()-st)
		# start_time = time.time()
	
	st = time.time()	
	for i,result in enumerate(results):
		if len(result):			
			bmap_vis,gdi,gdi_plus,gdi2, bmap_vis_denoised ,cx,cy,rectangle_pixels = result
							
			if final_attempt:
				GDI_calculator_all.append(gdi2)

			if gdi is not None and gdi_plus is not None: # valid grasp pose
				original_idx.append(i)
				GDI_calculator.append(gdi2)
				GDI.append(gdi)
				GDI_plus.append(gdi_plus)
				rectangle_list.append(rectangle_pixels)
				angle_list.append(angle_list_all[i])
				
			else:
				if gdi is not None and gdi_plus is None:
					gdi2.invalid_reason = 'insufficient contact region'
					gdi2.invalid_id = 5
				
			if dump_dir is not None:
				cv2.imwrite(dump_dir+'/bmaps'+'/bmap{0}_ws.jpg'.format(i),gdi2.bmap_ws)#.astype(np.uint8))
				cv2.imwrite(dump_dir+'/bmaps'+'/bmap{0}_denoised.jpg'.format(i),bmap_vis_denoised)#.astype(np.uint8))
				img_copy = copy.deepcopy(initial_img)
				if gdi is not None and gdi_plus is not None:
					gdi2.draw_refined_pose(img_copy)
				else:
					draw_rectified_rect(img=img_copy, pixel_points=rectangle_pixels)
				cv2.imwrite(dump_dir+'/directions'+'/gpose{0}.jpg'.format(i), img_copy)
				np.save(dump_dir+'/directions'+'/pmap{0}.npy'.format(i), gdi2.pmap)
				gdi2.dmap = gdi2.dmap - gdi2.dmap.min()
				cv2.imwrite(dump_dir+'/directions'+'/dmap{0}.jpg'.format(i), (gdi2.dmap / gdi2.dmap.max())*255)
				gdi2.final_image = img_copy
			# print(i,'gdi:',gdi,'gdi_plus:',gdi_plus)
		# print('time:',time.time()-st)#,'gqs:',gqs[k],'mps:',mps[k])
	print('arranging',time.time()-st)
	return GDI,GDI_plus,GDI_calculator,rectangle_list,GDI_calculator_all,rectangle_list_all,original_idx,angle_list




def select_a_best_grasp_pose(inputs):

	st = time.time()
	img = inputs['image']
	depth_image_copy = inputs['depth_image']
	darray = inputs['darray']
	final_attempt=inputs['final_attempt']
	dump_dir = inputs['dump_dir']
	param = inputs['param']

	# print(darray)
	if dump_dir is not None:
		create_directory(dump_dir+'/grasp_pose_info')
		create_directory(dump_dir+'/directions')
		create_directory(dump_dir+'/bmaps')
	
	if parallel_module:
		result_list = evaluate_selected_grasp_poses_parallel(inputs)
		max_score_per_point = None
	else:
		result_list = evaluate_selected_grasp_poses(inputs)
		max_score_per_point = result_list[8]
	
		
	GDI,GDI_plus, GDI_calculator, rectangle_list = result_list[0:4]
	GDI_calculator_all, rectangle_list_all, original_idx = result_list[4:7]



	angle_list = np.array(result_list[7])
	max_idx_per_point = result_list[9]
	GCS = result_list[10]

	original_idx = np.array(original_idx)
	

	gdi_old_way = False
	c = 0
	# print(rectangle_list_all)
	if len(GDI)==0 :
		if not final_attempt:
			return None,True,None,False,False, None
		else:
			best_rectangles, their_idx, GDI, GDI_plus  = select_best_rectangles_gdi_old_way(rectangle_list_all,GDI_calculator_all)
			GDI_calculator = GDI_calculator_all
			print('gdi old way')
			gdi_old_way = True
			# print('original_idx',original_idx,their_idx)
			# c = input('choose a pose?')
			# c = int(c)
	else:
		# GDI = normalize_gdi_score(GDI)
		# try:
		# 	GQS = inputs['gqs_score'][original_idx[:,0]] # It needs to modify according to the parallel
		# except:
		# 	GQS = None
		best_rectangles, their_idx = select_best_rectangles(rectangle_list,GDI,GDI_plus,GCS=GCS,top_rectangles_needed=10,final_attempt=final_attempt,inputs=inputs,angle_list=angle_list)
		# print('original_idx',original_idx[their_idx])
		# c = input('choose a pose?')
		# c = int(c)

	# print('org idx',original_idx)
	# print('GDI',GDI)
	# print('GDI_plus',GDI_plus)
	# # print('GQS',GQS)
	# print('Total',np.array(GDI)+np.array(GDI_plus))
	# print(best_rectangles)
	best_rectangles, their_idx = height_difference_consideration(best_rectangles, their_idx,darray) 
	
	best_rectangles = np.array(best_rectangles)

	
	# print(best_rectangles)
	final_rect_pixel_array = np.array(best_rectangles[c])


	outputs = {'final_pose_rectangle':final_rect_pixel_array}
	outputs['gdi_calculator'] = GDI_calculator[their_idx[c]]
	outputs['gdi_old_way'] = gdi_old_way
	outputs['best_rectangles'] = best_rectangles
	outputs['bmap'] = GDI_calculator[their_idx[c]].bmap_vis_denoised
	outputs['bmap_ws'] = GDI_calculator[their_idx[c]].bmap_ws
	outputs['invalid_reason'] = GDI_calculator[their_idx[c]].invalid_reason
	outputs['max_score_per_point'] = max_score_per_point
	outputs['rectangle_all'] = rectangle_list_all
	outputs['selected_gdi_calculators'] = GDI_calculator
	outputs['max_idx_per_point'] = max_idx_per_point.astype(int)

	outputs['pivots_gcs_img'] = result_list[11]

	if gdi_old_way:
		outputs['selected_idx'] = their_idx[c]
	else:
		outputs['selected_idx'] = original_idx[their_idx[c]]

	return outputs

	


def evaluate_selected_grasp_poses(inputs):

	masks = inputs['mask']
	top_grasp_points = inputs['top_grasp_points']
	dump_dir = inputs['dump_dir']
	initial_img = inputs['image']
	angles = inputs['angles']
	darray = inputs['darray']
	final_attempt=inputs['final_attempt']
	param = inputs['param']

	if inputs['gcs_scores'] is not None:
		gcs_scores = inputs['gcs_scores']
	else:
		gcs_scores = np.zeros(top_grasp_points.shape[0])


	GCS = [] # Grasp Confidence score
	GDI = [] # Collision Free score -> CFS
	GDI_plus = [] # Grasp Stability Score -> GSS
	GDI_calculator = []
	rectangle_list = []
	angle_list_valids = []
	GDI_calculator_all = []
	rectangle_list_all = []
	original_idx = []
	# directions = 4


	max_idx_per_point = np.zeros(top_grasp_points.shape[0])
	max_score_per_point = np.zeros(top_grasp_points.shape[0])
	# scores = inputs['scores']
	topm = inputs['topm']

	all_pivots = []
	pivot_gcs_scores = []
	for k,mask in enumerate(masks):
		# print(mask.shape)
		param.target = topm[k]+1
		# cut_length = 60
		# pivot_points = [top_grasp_points[k]]#
		pivot_points = cut_long_masks_along_major_axis(mask.transpose(),param.cut_length)

		# to save pivot points and their gcs scores
		

		for l,each_point in enumerate(pivot_points):
			
			all_pivots.append(each_point)
			pivot_gcs_scores.append(gcs_scores[k])

			centroid = each_point
			# rectangle_pixels_list, angle_list, asc_list = param.draw_rect_generic_fix_angles(centroid=each_point, directions=inputs['num_dirs'])
			rectangle_pixels_list, angle_list, asc_list = param.draw_rect_generic(each_point,angles[k], directions=inputs['num_dirs'])

			# start_time = time.time()
			# print(darray[int(centroid[1]),int(centroid[0])])
			# if darray[int(centroid[1]),int(centroid[0])]:
			for index,rectangle_pixels in enumerate(rectangle_pixels_list):
				print('	grasp point ',k,l,index)
				bmap_vis,gdi,gdi_plus,gdi2, bmap_vis_denoised ,cx,cy = calculate_GDI2_Lite(inputs,rectangle_pixels,angle_list[index]-radians(180))
				
				# gdi_plus = scores[k]
				gdi2.gcs_score = gcs_scores[k]

				if final_attempt:
					GDI_calculator_all.append(gdi2)
					rectangle_list_all.append(rectangle_pixels)
				if gdi is not None and gdi_plus is not None: # valid grasp pose

					# multiplying asc index to stability score (i.e. gdi_plus)
					gdi_plus = asc_list[index]*gdi_plus

					original_idx.append([k,index])
					GDI_calculator.append(gdi2)
					GDI.append(gdi)
					GDI_plus.append(gdi_plus)
					GCS.append(gdi2.gcs_score)
					rectangle_list.append(rectangle_pixels)
					angle_list_valids.append(angle_list[index])
					grasp_score = (gdi+gdi_plus+gdi2.gcs_score)/3
					if max_score_per_point[k] < grasp_score:
						max_score_per_point[k] = grasp_score
						max_idx_per_point[k] = len(GDI)-1

				else:
					if gdi is not None and gdi_plus is None:
						gdi2.invalid_reason = 'insufficient contact region'
						gdi2.invalid_id = 5
				if dump_dir is not None:
					np.savetxt(dump_dir+'/grasp_pose_info'+'/rectangle_{0}_{1}.txt'.format(k,index),rectangle_pixels)
					np.savetxt(dump_dir+'/grasp_pose_info'+'/angle_{0}_{1}.txt'.format(k,index),[angle_list[index]+radians(180)])
					# cv2.imwrite(dump_dir+'/bmaps'+'/bmap{0}_{1}.jpg'.format(k,index),bmap_vis)#.astype(np.uint8))
					# cv2.imwrite(dump_dir+'/bmaps'+'/bmap{0}_{1}_denoised.jpg'.format(k,index),bmap_vis_denoised)#.astype(np.uint8))
					
					img_copy = copy.deepcopy(initial_img)
					if gdi is not None and gdi_plus is not None:
						gdi2.draw_refined_pose(img_copy)
					# else:
					draw_rectified_rect(img=img_copy, pixel_points=rectangle_pixels)
					cv2.imwrite(dump_dir+'/directions'+'/gpose{0}_{1}_{2}.jpg'.format(k,l,index), img_copy)
				# print(k,index,'gdi:',gdi,'gdi_plus:',gdi_plus,gdi2.invalid_id)
			# print('time:',time.time()-st)#,'gqs:',gqs[k],'mps:',mps[k])
	
	pivots_gcs_img = draw_pivots_n_gcs_scores(all_pivots,pivot_gcs_scores,inputs['colored_gcs_map'])

	return GDI,GDI_plus,GDI_calculator,rectangle_list,GDI_calculator_all,rectangle_list_all,original_idx,angle_list_valids,max_score_per_point, max_idx_per_point, GCS, pivots_gcs_img


