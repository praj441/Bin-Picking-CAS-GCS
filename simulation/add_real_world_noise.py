import numpy as np 
import random
import cv2
from tqdm import tqdm

def add_real_world_noise(D):
	pool_num = random.randint(0,225)
	pool_noise = np.loadtxt('../../data/noise_samples/{0:06d}.txt'.format(pool_num)).astype(np.float32)
	noise_downsample = pool_noise[::2,::2]
	dmap_noised = D+noise_downsample
	return dmap_noised

data_path = '.'
depth_dir = data_path+'/depth_ims_noised'
depth_map_dir = data_path+'/depth_maps'

for scene in tqdm(range(50000)):
	D = np.load(depth_map_dir+'/dmap_{0:06d}.npy'.format(scene))
	D = add_real_world_noise(D)

	depth_img = 255*D

	cv2.imwrite(depth_dir+'/image_{0:06d}.jpg'.format(scene),depth_img)