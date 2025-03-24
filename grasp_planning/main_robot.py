#!/usr/bin/env python
import rospy, sys, numpy as np
import moveit_commander
from copy import deepcopy
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3,PoseStamped, WrenchStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool,Header, ColorRGBA
import moveit_msgs.msg
from sensor_msgs.msg import Image, CameraInfo,PointCloud2, PointField
from time import sleep
import time
from tf.transformations import quaternion_from_euler
import tf
import math
import cv2, cv_bridge
import os
import threading

from point_cloud.srv import point_cloud_service
from gripper_wsg import Gripper

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gripper_on',type=int,default=1, help='To tell whether gripper is on or off')
parser.add_argument('--exp_num',type=int,default=0, help='Current exp. number')
parser.add_argument('--total_attempt_current_run',type=int,default=0, help='Current trial number')
parser.add_argument('--method',type=str,default='ours', help='Current exp. number')
parser.add_argument('--start_fresh',type=int,default=1, help='start fresh or in-between')
FLAGS = parser.parse_args()

gripper_on = FLAGS.gripper_on
exp_num = FLAGS.exp_num
total_attempt_current_run = FLAGS.total_attempt_current_run
method = FLAGS.method

# if method == 'baseline':
# 	from grasp_planning_algorithm_new import grasp_planning
# elif method == 'ours':
from grasp_planning_ours import grasp_planning

DUMMY_FIELD_PREFIX = '__'
# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}


class ur10_grasp:
	def __init__(self):
		if FLAGS.start_fresh:
			self.start_fresh = True
			self.image_start = False
			self.image_end = False
			self.depth_start = False
			self.depth_end = False
			self.pc_start = False
			self.pc_end = False
		else:
			self.start_fresh = False
			self.image_start = True
			self.image_end = True
			self.depth_start = True
			self.depth_end = True
			self.pc_start = True
			self.pc_end = True
		self.cur_image = None
		self.cur_dmap = None
		self.cur_pc = None
		self.wrench_start = False
		self.wrench_end = False
		self.force_z = None
		self.brust_click_mode = False
		self.brust_count = 0


		#gripper parameters
		self.gripper = Gripper(gripper_on)
		self.gripper_length = 0.18
		self.gripper_homing_value = 100
		self.gripper_grasp_value = 20
		self.gripper_opening = 100

		self.exp_num = exp_num
		self.total_attempt_current_run = total_attempt_current_run

		rospy.init_node("ur10_mp", anonymous=False)
		rospy.loginfo("Starting node moveit_cartesian_path")
		rospy.on_shutdown(self.cleanup)
		# Initialize the move_group API
		moveit_commander.roscpp_initialize(sys.argv)
		# Initialize the move group for the ur5_arm
		self.arm = moveit_commander.MoveGroupCommander('manipulator')
		# Get the name of the end-effector link
		self.end_effector_link = self.arm.get_end_effector_link()
		# Set the reference frame for pose targets
		reference_frame = "/world"
		# Set the ur5_arm reference frame accordingly
		self.arm.set_pose_reference_frame(reference_frame)
		# Allow replanning to increase the odds of a solution
		self.arm.allow_replanning(True)
		# Allow some leeway in position (meters) and orientation (radians)
		self.arm.set_goal_position_tolerance(0.005)
		self.arm.set_goal_orientation_tolerance(0.01)
		self.arm.set_planning_time(0.1)
		self.arm.set_max_acceleration_scaling_factor(.15)
		self.arm.set_max_velocity_scaling_factor(.10)

		# rostopics
		self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
		self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
		self.marker_pub = rospy.Publisher('visualization_marker',Marker,queue_size = 100)
		self.marker_pub_other = rospy.Publisher('visualization_marker_other',Marker,queue_size = 100)
		self.msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
		self.normal_pose_publisher = rospy.Publisher('/normal_pose', PoseStamped, queue_size=1)
		self.world_pose_publisher = rospy.Publisher('/world_pose', PoseStamped, queue_size=1)
		self.wrench = rospy.Subscriber('/wrench', WrenchStamped, self.wrench_callback)
		self.point_sub = rospy.Subscriber('/camera/depth_registered/points',PointCloud2,self.point_cloud_callback)
		

		if method == 'baseline':
			self.path = '../images_ce/{0}/{1}'.format(10,self.exp_num)
		elif method == 'ours':
			self.path = '../results_ros_policy'
		if not os.path.exists(self.path):
			os.makedirs(self.path)
		self.action = None # to be filled in the grasp planning process
		self.fov_points = None # to be filled in the grasp planning process
		self.wTc = None # to be filled in the main loop , camera_pose_in_world
		self.main()
		print('init_end')


	def main(self):
		self.ee_pose = self.arm.get_current_pose(self.end_effector_link).pose
		self.save_camera_world_pose() # saved in self.wTc variable
		print('starting. Homing the gripper')
		self.gripper.run(self.gripper_homing_value)
		print('gripper moved.')
		grasp_planning(self)
		while True:

			self.show_point_in_rviz(self.action[0:3])
			# self.show_point_in_rviz(self.fov_points,other=True)
			input('If visualized in rviz enter any number:')
			# grasp manipulation action by the robot
			#**************************************************************************
			# move above the target               
			pos_error = self.move_gripper_xyz(self.action)
			time.sleep(0.5)
			self.gripper.run(self.gripper_opening)
			# orient the gripper
			self.orient_gripper(self.action[3])
			c = input('\ngripper opening check: enter 1 to continue, other value for alternate:')
			c = int(c)
			if c >= 5:
				self.gripper.run(self.gripper_opening+c)
			time.sleep(1)
			# move down
			print('moving down by:',-(0.05+self.finger_depth_down)+pos_error, pos_error)
			input('\nsafety check: enter 1 to continue:')
			downward_distance = -(0.05+self.finger_depth_down)+pos_error
			# self.move_gripper_updown(downward_distance) #down
			self.move_gripper_for_grasp(downward_distance)
			# close the fingers
			input('if sure press 1 and enter')
			self.gripper.run(self.gripper_closing)
			self.gripper.run(self.gripper_closing-20)
			time.sleep(1)


			self.move_gripper_updown(-2*downward_distance)
			time.sleep(1.0)

			self.move_to_predefined('bin_pose.txt')
			self.gripper.run(self.gripper_homing_value)
			time.sleep(0.5)

			self.move_to_predefined('start_pose_ur10.txt')
			print('robot moved to start_pose')
			sleep(1.0)
			self.ee_pose = self.arm.get_current_pose(self.end_effector_link).pose
			self.save_camera_world_pose() # saved in self.wTc variable
			
			# print('2',self.wTc)

			self.image_end = False
			self.depth_end = False
			self.image_start = False
			self.depth_start = False
			self.pc_start = False
			self.pv_end = False
			self.exp_num += 1
			self.total_attempt_current_run += 1
			if method == 'baseline':
				self.path = '../images_ce/{0}/{1}'.format(10,self.exp_num)
				if not os.path.exists(self.path):
					os.makedirs(self.path)
		
			grasp_planning(self)
			#**************************************************************************			

	def move_gripper_for_grasp(self,d):
		div = 4
		

		# for i in range(steps):
		while True:
			step = d/div
			self.move_gripper_updown(step)
			self.wrench_start = False
			self.wrench_end = False
			while not self.wrench_end:
				time.sleep(0.05)
			print('self.force_z',self.force_z)
			if self.force_z > 35 and div==8:
				self.move_gripper_updown(-1.25*step)
				break
			if div < 7:
				div = div+2


	def move_gripper_xyz(self,a):
		cpose = Pose()
		cpose.position.x = a[0]
		cpose.position.y = a[1]
		cpose.position.z = a[2]
		normal_pose = PoseStamped()
		normal_pose.header = Header(frame_id='camera_color_optical_frame')
		normal_pose.pose = cpose
		self.normal_pose_publisher.publish(normal_pose)

		# wpose = self.transform_pose_to_world(cpose)
		wpose = self.transform_pose_to_world(normal_pose)

		wpose.position.x += 0.0
		wpose.position.y += 0.0
		wpose.position.z = wpose.position.z + (self.gripper_length+0.05) # gripper length consideration and some extra margin
		wpose.orientation = self.ee_pose.orientation

		world_pose = PoseStamped()
		world_pose.header = Header(frame_id='world')
		world_pose.pose = wpose
		self.world_pose_publisher.publish(world_pose)

		return self.move_the_robot_new_way(wpose)

	
	def show_point_in_rviz(self,a,other=False):
		pose_temp = np.array(a).copy()
		if other:
			color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
		else:
			color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
		marker = Marker(
			type=Marker.SPHERE,
			id=0,
			lifetime=rospy.Duration(),
			pose=Pose(Point(a[0], a[1], a[2]), Quaternion(0, 0, 0, 1)),
			scale=Vector3(0.03, 0.03, 0.03),
			header=Header(frame_id='camera_color_optical_frame'),
			color=color)
		if other:
			self.marker_pub_other.publish(marker)
		else:
			self.marker_pub.publish(marker)

	def transform_pose_to_world(self,camera_pose):
		pose = camera_pose.pose
		transformer = tf.TransformerROS()
		wpose = deepcopy(pose)
		cTp = transformer.fromTranslationRotation((pose.position.x,pose.position.y,pose.position.z),(pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w))
		wTp = np.matmul(self.wTc,cTp)
		final_p = wTp[0:3,3]
		final_q = tf.transformations.quaternion_from_matrix(wTp)

		# Inserting the pose values
		wpose.position.x = final_p[0]
		wpose.position.y = final_p[1]
		wpose.position.z = final_p[2]
		wpose.orientation.x = final_q[1]
		wpose.orientation.y = final_q[2]
		wpose.orientation.z = final_q[3]
		wpose.orientation.w = final_q[0]

		return wpose
		

	def save_camera_world_pose(self):
		transformer = tf.TransformerROS()
		tfl = tf.TransformListener()
		from_link = '/camera_color_optical_frame'
		to_link = '/world'
		t = rospy.Time(0)
		tfl.waitForTransform(to_link,from_link,t,rospy.Duration(5))
		if tfl.canTransform(to_link,from_link,t):
			position, quaternion = tfl.lookupTransform(to_link,from_link,t)     
			self.wTc = transformer.fromTranslationRotation(position,quaternion)
		else:
		  rospy.logerr('Transformation is not possible!')

	
	def orient_gripper(self,angle_algo):
		
		if angle_algo > 0:
			angle = angle_algo - 1.57
		else:
			angle = angle_algo + 1.57
		self.arm.set_start_state_to_current_state()
		default_joint_states = self.arm.get_current_joint_values()
		default_joint_states[5] += angle
		self.arm.set_joint_value_target(default_joint_states)
		self.arm.set_start_state_to_current_state()
		plan = self.arm.plan()
		self.arm.execute(plan)


   
		
	def move_gripper_updown(self,value):
		ee_pose = self.arm.get_current_pose(self.end_effector_link).pose
		wpose = deepcopy(ee_pose)
		wpose.position.z += value

		self.waypoints = []
		self.waypoints.append(deepcopy(wpose))
		self.arm.set_start_state_to_current_state()
		plan, fraction = self.arm.compute_cartesian_path(self.waypoints, 0.02, 0.0, True)
		# input('enter 1 to continue')
		self.arm.execute(plan)
		print('robot moved')
		_,error_pos,error_orn = self.calculate_error_ee_pose(wpose)
		print('errors',error_pos,error_orn)

	def move_to_predefined(self,joint_states_txt):
		joint_states = self.arm.get_current_joint_values()
		joint_states = np.loadtxt(joint_states_txt, delimiter=',')
		self.arm.set_joint_value_target(joint_states)
		self.arm.set_start_state_to_current_state()
		plan = self.arm.plan()
		self.arm.execute(plan)
		

	def move_the_robot_new_way(self,wpose,loop=True):
		ee_pose = self.arm.get_current_pose(self.end_effector_link).pose
		wpose.position.x -= 0.005 # observed error correction
		# wpose.position.y += 0.005 # observed error correction
		dx = (wpose.position.x - ee_pose.position.x)
		dy = (wpose.position.y - ee_pose.position.y)
		dz = (wpose.position.z - ee_pose.position.z)
		self.waypoints = []
		for i in range(1,15):
			pose = deepcopy(ee_pose)
			pose.position.x = wpose.position.x - dx/i
			pose.position.y = wpose.position.y - dy/i
			# pose.position.z = wpose.position.z - dz/i
		for i in range(16,31):
			pose = deepcopy(ee_pose)
			pose.position.x = wpose.position.x - dx/i
			pose.position.y = wpose.position.y - dy/i
			pose.position.z = wpose.position.z - dz/i
			self.waypoints.append(deepcopy(pose))
		self.waypoints.append(deepcopy(wpose))

		self.waypoints = []
		self.waypoints.append(deepcopy(wpose))
		wait_for_safety_input = True
		
		while True:
			self.arm.set_start_state_to_current_state()
			plan, fraction = self.arm.compute_cartesian_path(self.waypoints, 0.01, 0.0, True)
			if wait_for_safety_input:
				c = input('Enter some number to continue:')
				wait_for_safety_input = False
				self.arm.execute(plan)
				# save_camera_images
				print('robot moved')
				time.sleep(2)
			else:
				self.arm.execute(plan)
				print('robot moved')
			error_depth,error_pos,error_orn = self.calculate_error_ee_pose(wpose)
			if (error_pos < 0.001 and error_orn < 0.01) or not loop:
				print('errors',error_pos,error_orn)
				break
			break
		return error_depth

	def calculate_error_ee_pose(self,wpose):
		ee_pose = self.arm.get_current_pose(self.end_effector_link).pose
		position_error = math.sqrt((wpose.position.x - ee_pose.position.x)**2 + (wpose.position.y - ee_pose.position.y)**2 + (wpose.position.z - ee_pose.position.z)**2)
		orn_error = math.sqrt((wpose.orientation.x - ee_pose.orientation.x)**2 + (wpose.orientation.y - ee_pose.orientation.y)**2 + (wpose.orientation.z - ee_pose.orientation.z)**2 + (wpose.orientation.w - ee_pose.orientation.w)**2)
		print('depth positions',wpose.position.x,ee_pose.position.x)
		return (wpose.position.x - ee_pose.position.x),position_error,orn_error

	def image_callback(self,data):
		if not self.image_start:
			try:
				self.image_start = True
				self.cur_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
				date_string = time.strftime("%Y-%m-%d-%H:%M")
				print('image callback')
				self.image_end = True
			except cv_bridge.CvBridgeError as e:
				print(e)

		if self.brust_click_mode:
			cur_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
			cv2.imwrite(self.path+'/brust/{0}.png'.format(self.brust_count),cur_image)
			self.brust_count += 1

	def depth_callback(self,data):
		if not self.depth_start:
			try:
				self.depth_start = True
				self.cur_dmap = cv_bridge.CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
				date_string = time.strftime("%Y-%m-%d-%H:%M")
				print('depth callback')
				self.depth_end = True
			except cv_bridge.CvBridgeError as e:
				print(e)

	def point_cloud_callback(self,data):
		h = data.height
		w = data.width
		if not self.pc_start:
			self.pc_start = True
			np_arr = self.pointcloud2_to_array(data)
			print('shape',np_arr.shape)
			point_arr = self.get_xyz_points(np_arr, remove_nans=True)
			# print('sizes',point_arr.shape, color_arr.shape)
			self.cur_pc = point_arr
			# np.save('result_dir/point_cloud_array.npy',point_arr)
			# np.save('result_dir/color_cloud_array.npy',color_arr)
			self.pc_end = True

	def wrench_callback(self,data):
		if not self.wrench_start:
			self.wrench_start = True
			self.force_z = data.wrench.force.z
			print('force_z',self.force_z)
			self.wrench_end = True

	def pointcloud2_to_dtype(self,cloud_msg):
		"""Convert a list of PointFields to a numpy record datatype.
		"""
		offset = 0
		np_dtype_list = []
		for f in cloud_msg.fields:
			while offset < f.offset:
				# might be extra padding between fields
				np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
				offset += 1
			np_dtype_list.append((f.name, pftype_to_nptype[f.datatype]))
			offset += pftype_sizes[f.datatype]

		# might be extra padding between points
		while offset < cloud_msg.point_step:
			np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
			offset += 1
			
		return np_dtype_list

	def pointcloud2_to_array(self,cloud_msg, split_rgb=False):
		"""
		Converts a rospy PointCloud2 message to a numpy recordarray
		
		Reshapes the returned array to have shape (height, width), even if the height is 1.

		The reason for using np.fromstring rather than struct.unpack is speed... especially
		for large point clouds, this will be <much> faster.
		"""
		# construct a numpy record type equivalent to the point type of this cloud
		dtype_list = self.pointcloud2_to_dtype(cloud_msg)
		# print(dtype_list)

		# parse the cloud into an array
		cloud_arr = np.fromstring(cloud_msg.data, dtype_list)

		# remove the dummy fields that were added
		cloud_arr = cloud_arr[
			[fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

		
		return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

	def get_xyz_points(self,cloud_array, remove_nans=True, dtype=np.float32):
		# global cur_image
		"""Pulls out x, y, and z columns from the cloud recordarray, and returns
		a 3xN matrix.
		"""
		# remove crap points
		print('before',cloud_array.shape, self.cur_image.shape)
		points = np.zeros(list(cloud_array.shape) + [3], dtype=dtype)
		h,w = cloud_array.shape
		points = np.zeros((h,w,3))
		points[:,:, 0] = cloud_array['x']
		points[:,:, 1] = cloud_array['y']
		points[:,:, 2] = cloud_array['z']

		return points#, color_masked

	def cleanup(self):
		rospy.loginfo("Stopping the robot")
		# Stop any current arm movement
		self.arm.stop()
		#Shut down MoveIt! cleanly
		rospy.loginfo("Shutting down Moveit!")
		moveit_commander.roscpp_shutdown()
		moveit_commander.os._exit(0)

mp=ur10_grasp()
rospy.spin()


