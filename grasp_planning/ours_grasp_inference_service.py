#!/usr/bin/env python3

import rospy, sys, numpy as np
from copy import deepcopy
from time import sleep
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from math import *
import copy

processing = False
new_msg = False
msg = None
cur_depth = None
cur_image_bgr = None
import time


class Camera:   
	def __init__(self):
		self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
		self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
		print('camera init_done')

	def image_callback(self,data):
		global cur_image_bgr
		global processing
		if not processing:
			try:
				cur_image_bgr = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
			except cv_bridge.CvBridgeError as e:
				print(e)

	def depth_callback(self,data):
		global cur_depth
		global processing
		if not processing:
			try:
				cur_depth = cv_bridge.CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
			except cv_bridge.CvBridgeError as e:
				print(e)


import sys
# sys.path.append("..")
sys.path.append("commons")
from general_predictor import Predictor
# from utils_gs import final_axis_angle, Parameters, draw_top_N_points, draw_grasp_map, draw_rectified_rect
from utils_gs_cas import Parameters
from termcolor import colored
from multiprocessing.connection import Listener

grasp_pose_pub = rospy.Publisher('grasp_pose',Float64MultiArray,queue_size=1,latch=False)

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



sampler = 'fcn_depth'
data_type = 'mid'
predictor = Predictor()
exp_num = 0
path = '../results_ros_policy'
def handle_grasp_pose_request():

	processing = True
	date_string = time.strftime("%Y-%m-%d-%H:%M:%S")
	exp_num = int(np.loadtxt(path+'/exp_num.txt'))
	print('exp_num',exp_num)

	img_path = path+'/{0}_ref_image.png'.format(exp_num)
	dmap_path = path+'/{0}_depth_array.txt'.format(exp_num)

	# inputs_np = {}
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
	

	print('info to be published')
	msg = Float64MultiArray()
	msg.data = result
	grasp_pose_pub.publish(msg)
	processing = False

	exp_num += 1

import logging
logging._srcfile = None
rospy.init_node('wcas_ros_policy_publisher')

# mp=Camera()
time.sleep(1)

print(colored("Ready for the grasp pose service.",'green'))
address = ('localhost', 6004) 
listener = Listener(address, authkey=b'secret password')
conn = listener.accept()

while not rospy.is_shutdown():
	try:
		msg = conn.recv()
	except KeyboardInterrupt:
		print("W: interrupt received, proceeding…")
	if msg == 'close':
		conn.close()
		break
	if int(msg):
		print(colored('received a request for grasp pose','green'))
		handle_grasp_pose_request()
		print(colored('grasp pose published','green'))
