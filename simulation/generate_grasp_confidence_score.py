import numpy as np
import os 
from tqdm import tqdm
from PIL import Image
import sys
from math import radians
import cv2
import copy

sys.path.append('../../grasp_planning/src_cas_final/commons')
from grasp_evaluation_cas import calculate_GDI2_Lite
from utils_gs_cas import Parameters, create_directory, draw_rectified_rect
from cas_grasp_algo import calculate_center

data_path = '../../data/cas_sim_dm7' 
out_folder = "grasp_confidense_scores" 
mask_folder = "modal_segmasks"
darray_folder = "depth_maps"

out_path = os.path.join(data_path, out_folder)
mask_path = os.path.join(data_path, mask_folder)
darray_path = os.path.join(data_path, darray_folder)
dump_dir = os.path.join(data_path, "dump")
create_directory(out_path)
create_directory(dump_dir)
create_directory(dump_dir+'/directions')

N = 30000
w = 640 #320
h = 480 #240
param = Parameters(w,h)
param.gripper_height = int(param.mh*70)

#************ Hyperparameter: depth-based-collision-check (DBCC) **********
DBCC_enable = False
param.DBCC_enable = DBCC_enable
#**************************************************************************




inputs = {'param':param}
inputs['slanted_pose_detection'] = False
inputs['cone_detection'] = False 
inputs['num_dirs'] = 6
param.calculate_asc_indices(inputs['num_dirs'])

for i in tqdm(range(N)):
	if 'wisdom' in data_path:
		mask_file = os.path.join(mask_path, 'image_{0:06d}.png'.format(i))
	else:
		mask_file = os.path.join(mask_path, 'mask_{0:06d}.png'.format(i))
	whole_mask = Image.open(mask_file)
	whole_mask = np.array(whole_mask)


	if 'wisdom' in data_path:
		darray_file = os.path.join(darray_path, '{0:03d}_depth_array.txt'.format(i))
		darray = np.loadtxt(darray_file)
	else:
		darray_file = os.path.join(darray_path, 'dmap_{0:06d}.npy'.format(i))
		darray = np.load(darray_file)

	inputs['seg_mask'] = whole_mask
	inputs['darray'] = darray

	img = cv2.imread(os.path.join(data_path,"color_ims" ,'image_{0:06d}.png'.format(i)))

	# label issue: marking backgound objects as one class (0)
	if 'sim' in data_path:
		whole_mask = np.where(whole_mask <4 , 0, whole_mask)

	mask_ids = np.unique(whole_mask)[1:]
	center_list = []
	angles = []
	grasp_score_list = []
	for idx in mask_ids:
		# print(i,idx)
		mask = np.transpose(np.where(whole_mask==idx,idx,0))
		# cv2.imwrite('temp{0}.jpg'.format(idx), 255*(mask/mask.max()))
		center = calculate_center(mask)
		center_list.append(center)

		indices = np.argwhere(mask)
		dict = param.axis_angle(points=indices)
		minor_points = dict["minor_axis_points"]
		# imgk_with_minor_axis = param.draw_minor_axis(minor_points,mask[k],scores[k],copy.deepcopy(img))
		# cv2.imwrite(path+'/imgk_with_minor_axis_{0}.jpg'.format(k), imgk_with_minor_axis)
		angles.append(dict["angle"])

	top_grasp_points = np.array(center_list)
	angles = np.array(angles)

	for k,each_point in enumerate(top_grasp_points):
		# print('	grasp point ',k)
		centroid = each_point
		rectangle_pixels_list, angle_list, asc_list = param.draw_rect_generic(each_point,angles[k], directions=inputs['num_dirs'])
		avg_grasp_score = 0.0
		total_valids = 0
		for index,rectangle_pixels in enumerate(rectangle_pixels_list):
			bmap_vis,gdi,gdi_plus,gdi2, bmap_vis_denoised ,cx,cy = calculate_GDI2_Lite(inputs,rectangle_pixels,angle_list[index]-radians(180))
			# img_copy = copy.deepcopy(img)
			if gdi is not None and gdi_plus is not None: # valid grasp pose
				avg_grasp_score += (gdi+gdi_plus)/2
				# total_valids += 1
			# 	gdi2.draw_refined_pose(img_copy)
			# draw_rectified_rect(img=img_copy, pixel_points=rectangle_pixels)
			# cv2.imwrite(dump_dir+'/directions'+'/gpose{0}_{1}.jpg'.format(k,index), img_copy)
		# if total_valids > 1:
		avg_grasp_score = avg_grasp_score/inputs['num_dirs']
		grasp_score_list.append(avg_grasp_score)

	grasp_scores = np.array(grasp_score_list)
	out_file = os.path.join(out_path, '{0:06d}.txt'.format(i))
	np.savetxt(out_file,grasp_scores)