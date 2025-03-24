import pybullet as p
import time
import pybullet_data
import random
import sys
from PIL import Image
import numpy as np
from camera import get_image,cam_pose_to_world_pose
from utils import load_random_urdf
from utils import load_shapenet
from utils import load_shapenet_natural_scale
from utils import sample_camera_displacement
from utils import relative_ee_pose_to_ee_world_pose
from utils import objects_picked_succesfully
from utils import load_random_object_texture, load_random_table_n_floor_texture
# from cam_ik import move_eye_camera,accurateIK
# from kuka_vipul import Kuka
from math import *
import cv2
# from roboaction import SetAction
from tqdm import tqdm
from os import path
from datetime import datetime
# from utils import HiddenPrints
import os
import logging

import open3d as o3d
import sys
sys.path.append('../')
# from grasp_planning_algorithm import run_grasp_algo
import random
import pickle



w = 640 #512 #320 #320
h = 480 #512 #240 #240
generate_bare_binimum_training_data = False # Setting it False will generate full data
generate_graspability_scores = False

median_depth_map = np.loadtxt('median_depth_map_hd.txt')



manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)



# def add_real_world_noise(D):
# 	pool_num = random.randint(0,225)
# 	pool_noise = np.loadtxt('../../data/noise_samples/{0:06d}.txt'.format(pool_num)).astype(np.float32)
# 	noise_downsample = pool_noise[::2,::2]
# 	dmap_noised = D+noise_downsample
# 	return dmap_noised

def generate_modal_n_binary_mask(S,uids):
	h,w = S.shape
	Sb = np.zeros((h,w))
	Sm = np.zeros((h,w))
	for i in range(h):
		for j in range(w):
			if S[i,j] in uids:
				Sb[i,j] = 255.0
				Sm[i,j] = S[i,j]
	return Sb,Sm

def create_directory(dname):
	if not os.path.exists(dname):
	    print('creating directory:',dname)
	    os.makedirs(dname)

if __name__ == "__main__":

	logging.basicConfig(filename='example.log', level=logging.DEBUG)
	path = 'temp' # to store temp files
	data_path = '../../data/cas_sim_dm7'
	obj_path = '/home/prem/ur_grasping_test/src/case_extension_work/urdfs/google-objects-kubric'
	urdf_path = '/home/prem/ur_grasping_test/src/case_extension_work/urdfs'
	depth_dir = data_path+'/depth_ims'
	depth_map_dir = data_path+'/depth_maps'
	img_dir = data_path+'/color_ims'
	seg_mask_dir = data_path+'/modal_segmasks'
	binary_mask_dir = data_path+'/segmasks_filled'

	create_directory(depth_dir)
	create_directory(depth_map_dir)
	create_directory(img_dir)
	create_directory(seg_mask_dir)
	create_directory(binary_mask_dir)


	num_scene = 30000
	offset = 0
	num_itr_per_scene = 5
	num_sample_per_itr = 5
	PI = pi
	index = 0
	physicsClient = p.connect(p.DIRECT) #or p.DIRECT for non-graphical version
	p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

	np.savetxt(data_path+'/median_depth_map.txt',median_depth_map)
	

	max_tries = 10
	# data = np.zeros((number_of_runs,5))
	data = []
	sample_dirs = 1
	fix_cluster = False
	FSL_only = False
	CRS_only = False
	pose_refine = True
	center_refine = False

	st = time.time()
	max_num_objects = 30000
	num_obj = max_num_objects
	objects_picked = 0
	gdi_old_way_counts = 0
	total_algo_time = 0
	algo_run_count = 0

	if len(sys.argv) > 2:
		part = int(sys.argv[1])
		total_parts = int(sys.argv[2])
		part_scenes = int(num_scene/total_parts)
		start = offset + part*part_scenes
		end = offset + (part+1)*part_scenes
		print(part,total_parts,start,end)
	else:
		start = 0
		end = num_scene

	with open(obj_path+'/short_list.txt') as f:
			object_list = f.readlines()

	for scene in range(start,end):
		if generate_bare_binimum_training_data:
			check_path = binary_mask_dir+'/image_{0:06d}.png'.format(scene)
		else:
			check_path = depth_map_dir+'/dmap_{0:06d}.npy'.format(scene)
		if os.path.exists(check_path):
			continue
		print('\n****',scene,'\n****')
		p.setGravity(0,0,-10)
		planeId = p.loadURDF(urdf_path+"/my-plane.urdf")
		cubeStartPos = [0,0,0]
		cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
		tabId = p.loadURDF(urdf_path+"/table.urdf",cubeStartPos, cubeStartOrientation)
		floorId = p.loadURDF(urdf_path+"/carpet.urdf",cubeStartPos, cubeStartOrientation)
		
		bin_mid_part_Id = p.loadURDF(urdf_path+"/bin_mid_part.urdf",[0.145,0.0,0.65],globalScaling=1.0)
		binId = p.loadURDF(urdf_path+"/bin.urdf",[0.145,0.0,0.7],globalScaling=1.0,useFixedBase=1)
		
		# if np.random.random() > 0.5:
		load_random_table_n_floor_texture(tabId)
		# if np.random.random() > 0.5:
		# load_random_table_n_floor_texture(floorId)
		# if np.random.random() > 0.5:	
		load_random_table_n_floor_texture(binId)
		load_random_table_n_floor_texture(bin_mid_part_Id)

		p.resetDebugVisualizerCamera( cameraDistance=2.2, cameraYaw=140, cameraPitch=-60, cameraTargetPosition=[0,0,0])
		
		for _ in range(100):
			p.stepSimulation()

		tableObjectUids = []
		object_list_ids = []
		d = 0.9
		fix_flag = 0
		
		for _ in range(1000):
			p.stepSimulation()
		
		
		num_obj = random.randint(5,45)
		obj_ids = {}
		random_order = np.random.permutation(len(object_list))
		object_nice_flag = np.loadtxt(obj_path+'/nice_flags.txt')
		scale_array = np.loadtxt(obj_path+'/scale.txt')
		regular_flag = np.loadtxt(obj_path+'/regular_flag.txt')
		obj_count = 0
		# random_order = range(125,144)
		for i in random_order:
			if obj_count >= num_obj:
				break
			if object_nice_flag[i] == -1 or object_nice_flag[i] == 3 or regular_flag[i] == 2:
				continue
			obj_idx = i #random.randint(0,len(object_list)-1) #
			object_list_ids.append(obj_idx)
			obj_name = object_list[obj_idx].replace('\n','')
			x = random.uniform(-0.05,0.20)
			y = random.uniform(-0.15,0.15)
			position = [x,y,d]
			scaling = scale_array[obj_idx]
			scale_factor = random.uniform(0.7,1.0)
			cubeStartOrientation1 = p.getQuaternionFromEuler([random.randint(0,1)*pi/2,random.randint(0,1)*pi/2,random.uniform(-pi,pi)])
			cubeStartOrientation2 = p.getQuaternionFromEuler([random.randint(0,0)*pi/2,random.randint(0,0)*pi/2,random.uniform(-pi,pi)])
			fp = obj_path + '/' + obj_name + '/object.urdf'
			print(obj_name)
			uid = p.loadURDF(fp,position, cubeStartOrientation,globalScaling=scale_factor*scaling,useFixedBase=fix_flag)
			# uid = load_obj_custom("../urdfs/other/{0}/".format(obj_name),position, cubeStartOrientation1,globalScaling=scaling,useFixedBase=fix_flag)
			tableObjectUids.append(uid)
			obj_ids[uid] = obj_idx #mapping between my obj ids and pybullet obj ids
			load_random_object_texture(tableObjectUids[-1])
	
			for _ in range(100):
				p.stepSimulation()
			obj_count += 1
			# print('******',i+1)
			# c = input('scale it')

		print(tableObjectUids)
		
		for _ in range(1000):
			p.stepSimulation()
		
		

		I,D,S = get_image(w,h)


		Sb,Sm = generate_modal_n_binary_mask(S,tableObjectUids)

		depth_image = 255*D #(D/D.max())
		
		# cv2.imwrite(depth_dir+'/image_{0:06d}.png'.format(scene),depth_image)
		cv2.imwrite(depth_dir+'/image_{0:06d}.png'.format(scene),depth_image)
		# np.save(seg_mask_dir+'/mask_{0:06d}.npy'.format(scene),S)
		cv2.imwrite(seg_mask_dir+'/mask_{0:06d}.png'.format(scene),Sm)
		cv2.imwrite(img_dir+'/image_{0:06d}.png'.format(scene),I)
		cv2.imwrite(binary_mask_dir+'/image_{0:06d}.png'.format(scene),Sb)

		if not generate_bare_binimum_training_data:
			# np.savetxt(binary_mask_dir+'/{0:06d}_binary_mask.txt'.format(scene),Sb)
			np.save(depth_map_dir+'/dmap_{0:06d}.npy'.format(scene),D)
			# cv2.imwrite(depth_dir+'/dmap_{0:06d}.png'.format(scene),D)
		
		p.resetSimulation()
	p.disconnect()