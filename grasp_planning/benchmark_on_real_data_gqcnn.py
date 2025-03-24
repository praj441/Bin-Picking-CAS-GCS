import numpy as np
import cv2
import time
import sys, os
from datetime import datetime
from math import *


sys.path.append('commons')
from utils_gs import  Parameters
from grasp_evaluation import calculate_GDI2
from utils_gs import create_directory

#preparaing dexnet pipeline
gqcnn_path = '/home/prem/prem_workspace/pybullet_learning/grasping'
sys.path.append(gqcnn_path+"/gqcnn")
from grasp_predictor import GQCNN_Predictor
model_name = 'GQCNN-4.0-PJ'
camera_intr_file = gqcnn_path+ '/gqcnn/data/calib/primesense/primesense.intr'
# camera_intr_file = '../gqcnn/data/calib/pybullet/pybullet_camera.intr'
gqcnn_predictor = GQCNN_Predictor(model_name,camera_intr_file)


data_path = '../wisdom-real'
out_path = data_path + '/dexnet'
w = 640 #320
h = 480 #240
param = Parameters(w,h)
param.gripper_height = int(param.mh*70) 

folder_path = data_path + '/depth_maps'
folder_path_2 = data_path + '/color_ims'

scan_names = sorted(list(set([os.path.basename(x)[0:3] \
	for x in os.listdir(folder_path)])))

inputs = {'param':param}
inputs['slanted_pose_detection'] = False #False
inputs['cone_detection'] = False #False
avg_time = 0.0
gqcnn_score_list = []



# scan_names = ['000007','000012']
scenes = 1
for idx in scan_names:
	# if scenes == 51:
	# 	break
	print(folder_path,idx)
	dmap = np.loadtxt(os.path.join(folder_path, idx)+'_depth_array.txt')
	img = cv2.imread(os.path.join(folder_path_2, idx)+'_ref_image.png')
	dmap = dmap[100:580,160:800]
	img = img[100:580,160:800]
	# dt = datetime.now().strftime("%d%m%Y")
	img = cv2.resize(img,(w,h))
	st = time.time()
	px,py,angle,d,grasp_width_px,flag = gqcnn_predictor.predict(cv2.resize(dmap,(w,h)),None,None)
	if scenes > 1:
		avg_time += (time.time()-st)
		print('inference time',(time.time()-st))
		print('avg_time',avg_time/(scenes-1))
	inputs['darray'] = cv2.resize(dmap,(w,h))
	# dump_dir = out_path + '/' + idx
	# create_directory(dump_dir)
	dump_dir = None
	inputs['dump_dir'] = dump_dir
	# inputs['param'] = param
	if angle is not None:
		img1,rectangle_pixels = param.draw_rect_gqcnn(img.copy(),np.array([int(px),int(py)]), angle, color = (0,0,255),scale=1)
		bmap,gdi,gdi_plus,gdi2, bmap_denoised ,cx,cy = calculate_GDI2(inputs,rectangle_pixels,angle-radians(180))
		if gdi is not None and gdi_plus is not None:
			gqs_score = (gdi+gdi_plus)/2
			# img = 0.8*img
			gdi2.draw_refined_pose(img,scale=1, thickness=4)
			grasp_width_px = gdi2.gripper_opening
		else:
			if gdi_plus is not None:
				gdi2.invalid_reason = 'small contact region'
			img = img1
			gqs_score = 0.0
			grasp_width_px = param.gripper_height
		print('gqs score',gqs_score,gdi2.FLS_score,gdi2.CRS_score)
		gqcnn_score_list.append(gqs_score)
	grasp = [px,py,angle,grasp_width_px]
	# bmap_vis = (bmap / bmap.max())*255
	# bmap_vis_denoised = (bmap_denoised / bmap_denoised.max())*255
	# cv2.imwrite(dump_dir+'/bmap.jpg',bmap_denoised)#.astype(np.uint8))
	# cv2.imwrite(dump_dir+'/bmap_ws.jpg',gdi2.bmap_ws)#.astype(np.uint8))
	# with open(dump_dir+'/invalid_reason.txt', 'w') as f:
	# 		f.write(gdi2.invalid_reason)

	cv2.imwrite(out_path+'/final_image_{0}.png'.format(idx),img)
	np.savetxt(out_path+'/grasp_{0}.txt'.format(idx),grasp)
	np.savetxt(out_path+'/score_list.txt',gqcnn_score_list,fmt='%3d')
	
	print('acc',np.count_nonzero(gqcnn_score_list)/scenes)
	scenes += 1
avg_time = avg_time/scenes-1

	# pc_cloud = np.loadtxt(os.path.join(data_path, idx)+'_pc.npy')
	
	# c = input('ruko. analyze karo.')

