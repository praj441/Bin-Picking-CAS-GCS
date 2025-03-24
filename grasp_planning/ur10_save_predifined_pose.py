#!/usr/bin/env python


import rospy, sys, numpy as np
import moveit_commander
from copy import deepcopy
import moveit_msgs.msg


from time import sleep
from tf.transformations import quaternion_from_euler
import time
from std_msgs.msg import Float64MultiArray  

from urdf_parser_py.urdf import URDF
# from pykdl_utils.kdl_kinematics import KDLKinematics

import sys

#from get_jacobian import Jacobian

class ur5_mp:
	def __init__(self):
		rospy.init_node("ur5_mp", anonymous=False)
		rospy.loginfo("hi, is this the init ?: yes")
		
				
		self.vel_pub = rospy.Publisher('/joint_group_vel_controller/command',Float64MultiArray,queue_size=1,latch=True)
		self.state_change_time = rospy.Time.now()

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
		self.arm.set_goal_position_tolerance(0.001)
		self.arm.set_goal_orientation_tolerance(0.01)
		self.arm.set_planning_time(0.1)
		self.arm.set_max_acceleration_scaling_factor(.5)
		self.arm.set_max_velocity_scaling_factor(.5)
		


		# Set the internal state to the current state

		self.arm.set_start_state_to_current_state()
		self.default_joint_states = self.arm.get_current_joint_values()
		if len(sys.argv) < 2:
			print('please provide pose file')
			sys.exit()
		pose_file = sys.argv[1]
		np.savetxt(pose_file,self.default_joint_states,fmt='%1.3f')

		print(self.default_joint_states)
		print('init_end')


	def cleanup(self):
		rospy.loginfo("Stopping the robot")

		# Stop any current arm movement
		self.arm.stop()

		#Shut down MoveIt! cleanly
		rospy.loginfo("Shutting down Moveit!")
		moveit_commander.roscpp_shutdown()
		moveit_commander.os._exit(0)


mp=ur5_mp()
rospy.loginfo("hi, is this the start")

# rospy.spin()
