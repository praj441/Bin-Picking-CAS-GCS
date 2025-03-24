#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import UInt16
import sys
import rosservice

class Gripper:
	def __init__(self,gripper_on):
		# rospy.init_node('gripper')
		self.pub = rospy.Publisher('/data', UInt16, queue_size=10)

		self.gripper_on = gripper_on
	def homing(self):
		rospy.wait_for_service('/wsg_50/homing')
		try:
		    get_gripper_homing_position = rospy.ServiceProxy('/wsg_50/homing', rosservice.get_service_class_by_name('/wsg_50/homing'))
		    resp = get_gripper_homing_position()
		    return
		except rospy.ServiceException as e:
		    print("Service call failed: %s"%e)

	def move_by_value(self,value):
		rospy.wait_for_service('/wsg_50/move')
		try:
		    get_gripper_homing_position = rospy.ServiceProxy('/wsg_50/move', rosservice.get_service_class_by_name('/wsg_50/move'))
		    resp = get_gripper_homing_position(width=value ,speed=50.0)
		    return
		except rospy.ServiceException as e:
		    print("Service call failed: %s"%e)
	def run(self,value):
		if self.gripper_on:
			if value == 100:
				self.homing()
			elif value > 100 and value < 10:
				print('Gripper width should be less than 100 and greater than 10.')
				return
			else:
				self.move_by_value(value)
		else:
			print('attention: gripper is off')

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print('please provide gripper opening value')
		sys.exit()
	value = float(sys.argv[1])
	print(value)
	rospy.init_node('gripper')
	gripper = Gripper(True)
	gripper.run(value)