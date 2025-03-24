# import rospy
# from point_cloud.srv import point_cloud_service
from math import *
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point, Polygon
import numpy as np
import cv2
import copy
import sys
import random
# from focus_net_predictor import check_for_validity_by_cnn # should not be placed here


#import necessary functions to calculate different functions of masks 
from functions_for_length_wise_cuts import major_axis_length
from functions_for_length_wise_cuts import axis_angle
from functions_for_length_wise_cuts import calculate_center
from functions_for_length_wise_cuts import rotate_boolean_array
from functions_for_length_wise_cuts import crop_binary_mask

def divide_object_masks_along_their_longer_axis(mask,out_path = None):
	center_list=[]
	angles = []
	major_lengths = []

	for i in range(mask.shape[0]):
		center = calculate_center(mask[i])
		center_list.append(center)
		centers = np.array(center_list)
		dict = axis_angle(mask[i])
		angle = dict["angle"]
		angles.append(angle)

	angles = np.array(angles)
	angle_deg = np.degrees(angles)
	# print('angle_degrees', angle_deg)

	centers = centers[:, ::-1]
	# print('centers',centers)

	for i in range(mask.shape[0]):
		major_axis = major_axis_length(mask[i])
		major_lengths.append(major_axis[1])

	major_lengths = np.array(major_lengths)
	# print(major_lengths)

	masks=[]
	num_images = mask.shape[0]
	for i in range(num_images):

		copy = mask[i]
		theta = np.pi/2-1*angles[i]
		start_of_the_cluster = centers[i][0] - (major_lengths[i])/2
		divisor = major_lengths[i]/50
		length_per_divison = major_lengths/divisor

		print('	divide_object_masks',i,'.	no_of_divisor:',divisor)
		for j in range(1,int(divisor)):

			width = start_of_the_cluster + j*length_per_divison
			# print("width: ",width[0])

			binary_mask_image = rotate_boolean_array(copy, theta, centers[i])
			cropped_image_1,cropped_image_2 = crop_binary_mask(binary_mask_image,width)
			cropped_image_1_copy = rotate_boolean_array(cropped_image_1, -1*theta, centers[i])
			cropped_image_2_copy = rotate_boolean_array(cropped_image_2,-1*theta , centers[i])
			copy = cropped_image_2_copy

			print('		divisor',j)
			if out_path is not None:
				new_center_1 = calculate_center(cropped_image_1_copy)
				fig, axes = plt.subplots(1, 5, figsize=(15, 5))
				axes[0].plot(centers[i][0], centers[i][1], 'ro')
				axes[0].plot(new_center_1[1], new_center_1[0], 'go') 
				axes[0].imshow(mask[i], cmap='gray')
				axes[0].set_title("Original Mask image")
				axes[0].axis('off')

				axes[1].plot(centers[i][0], centers[i][1], 'ro') 
				axes[1].imshow(cropped_image_1, cmap='gray')
				axes[1].set_title("Rotated Cropped Image 1")
				axes[1].axis('off')

				axes[2].plot(centers[i][0], centers[i][1], 'ro') 
				axes[2].imshow(cropped_image_2, cmap='gray')
				axes[2].set_title("Rotated Cropped Image 2")
				axes[2].axis('off')

				axes[3].plot(centers[i][0], centers[i][1], 'ro') 
				axes[3].plot(new_center_1[1], new_center_1[0], 'go') 
				axes[3].imshow(cropped_image_1_copy, cmap='gray')
				axes[3].set_title("Cropped Image 1")
				axes[3].axis('off')

				axes[4].plot(centers[i][0], centers[i][1], 'ro') 
				axes[4].imshow(cropped_image_2_copy, cmap='gray')
				axes[4].set_title("Cropped Image 2")
				axes[4].axis('off')

				plt.tight_layout()
				file_name = 'mask_image_' + str(i) + '_' + str(j) + '.png'
				# folder_path = os.path.join(out_path, 'baseline', idx, 'cropped_image')
				# os.makedirs(folder_path, exist_ok=True)
				image_path = os.path.join(out_path, file_name)
				print(image_path)
				plt.savefig(image_path, dpi=300)
				# plt.show()



			if(np.count_nonzero(cropped_image_1_copy)>15):
				masks.append(cropped_image_1_copy)
		if(np.count_nonzero(copy)>15):
			masks.append(copy)
	masks = np.array(masks)
	return masks

def major_axis_length(points):
	X = points[:,0]
	Y = points[:,1]
	x = X - np.mean(X)
	y = Y - np.mean(Y)
	coords = np.vstack([x, y])
	cov = np.cov(coords)
	evals, evecs = np.linalg.eig(cov)
	sort_indices = np.argsort(evals)[::-1]
	x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
	x_v2, y_v2 = evecs[:, sort_indices[1]]
	theta = -np.arctan((y_v1)/(x_v1))

	#find the end point indexes for major and minor axis
	rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
					  [np.sin(theta), np.cos(theta)]])
	transformed_mat = rotation_mat * np.vstack([X, Y])
	x_transformed, y_transformed = transformed_mat.A
	major = np.max(x_transformed)-np.min(x_transformed)
	minor = np.max(y_transformed)-np.min(y_transformed)
	return theta,major,minor
	
def remove_outlier(points):
	mean = np.mean(points, axis=0)
	sd = np.std(points, axis=0)
	mask = (points > (mean - 2 * sd)) & (points < (mean + 2 * sd))
	mask = mask.all(axis=1)
	# print('removed ',points.shape[0]-np.count_nonzero(mask),' outliers')
	return points[mask], mask

def pixels_to_xyz(px,py,d,w=320,h=240,fx=307.36,fy=307.07):
	P[:,0] = np.where(P[:,0] < 0,0,P[:,0])
	P[:,1] = np.where(P[:,1] < 0,0,P[:,1])
	P[:,0] = np.where(P[:,0] > self.w-1,self.w-1,P[:,0])
	P[:,1] = np.where(P[:,1] > self.h-1,self.h-1,P[:,1])

	N = P.shape[0]
	Points = np.zeros((N,3))
	Z = P[:,2]
	Points[:,0] = (P[:,0] - (w/2))*(Z/(fx))
	Points[:,1] = (P[:,1] - (h/2))*(Z/(fy))
	Points[:,2] = Z
	return Points 

def pixel_to_xyz(px,py,d,w=320,h=240,fx=307.36,fy=307.07):
	#cartesian coordinates
	if px < 0:
		px = 0
	if py < 0:
		py = 0
	if px > w-1:
		px = w-1
	if py > h-1:
		py = h-1
	z = d
	x = (px - (w/2))*(z/(fx))
	y = (py - (h/2))*(z/(fy))
	return np.array([x,y,z])

def pixels_to_point_projection(P,w=320,h=240,fx=307.36,fy=307.07):
	P[:,0] = np.where(P[:,0] < 0,0,P[:,0])
	P[:,1] = np.where(P[:,1] < 0,0,P[:,1])
	P[:,0] = np.where(P[:,0] > self.w-1,self.w-1,P[:,0])
	P[:,1] = np.where(P[:,1] > self.h-1,self.h-1,P[:,1])

	N = P.shape[0]
	Points = np.zeros((N,3))
	Z = P[:,2]
	Points[:,0] = (P[:,0] - (w/2))*(Z/(fx))
	Points[:,1] = (P[:,1] - (h/2))*(Z/(fy))
	Points[:,2] = Z
	return Points 

def points_to_pixels_projection(points,w=320,h=240,fx=307.36,fy=307.07):
	X = points[:,0]
	Y = points[:,1]
	Z = points[:,2]

	print(points.shape)
	PX = (np.divide(X,Z)*fx + w/2).astype(np.int32)
	PY = (np.divide(Y,Z)*fx + h/2).astype(np.int32)

	P = np.zeros((PY.shape[0],3))
	P[:,0] = PX
	P[:,1] = PY
	P[:,2] = Z

	return P

def draw_clusters_into_image(image,P,labels,gqs_score):
	image1 = image.copy()
	image2 = image.copy()
	PX = P[:,0]
	PY = P[:,1]
	num_points = PY.shape[0]

	for k in range(num_points):
		# if gqs_score[k] < 30:
		# 	csize = 1
		# 	# green_part = 50 #int((labels[k]*1)%255)
		# 	# blue_part = 50 #int((labels[k]*90)%255)
		# 	# red_part = 256 #int((labels[k]*213)%255)
		# else:
		csize = int(gqs_score[k]/4)
			# green_part = 256 #int((labels[k]*1)%255)
			# blue_part = 50 #int((labels[k]*90)%255)
			# red_part = 50 #int((labels[k]*213)%255)
			
		cv2.circle(image1, (int(PX[k]), int(PY[k])), csize, (0,0,255), -1)
		cv2.circle(image2, (int(PX[k]), int(PY[k])), 2, (0,0,0), -1)

	return image1,image2


def draw_top_indices(image,P,length,directions=6):
	PX = P[:,0]
	PY = P[:,1]
	num_points = PY.shape[0]
	for k in range(num_points):
		u = int(180/directions)
		# csize = int(gqs_score[k]/5)
		
		for i in range(0,180,u):
			angle = radians(i)
			angle = keep_angle_bounds(angle) 
			[x1, y1] = [int(PX[k] - length * 0.5 * cos(angle)),
						int(PY[k] - length * 0.5 * sin(angle))]
			[x2, y2] = [int(PX[k] + length * 0.5 * cos(angle)),
						int(PY[k] + length * 0.5 * sin(angle))]
			# cv2.line(image, (x1,y1), (x2,y2),color=(0,255,0), thickness=2)
		cv2.circle(image, (int(PX[k]), int(PY[k])), 5, (0,0,255), -1)
	return image

def draw_contours_around_clusters(image,P,labels):
	PX = P[:,0]
	PY = P[:,1]
	ids = np.unique(labels)
	for i in ids:
		mask = (labels==i)
		contour = np.hstack((PX[mask][:, np.newaxis], PY[mask][:, np.newaxis]))
		convexHull = cv2.convexHull(contour.astype(np.int))
		cv2.drawContours(image, [convexHull], -1, (255,0,0), 2)
	return image

def choose_a_target_cluster(P,labels,gqs_score):
	PX = P[:,0]
	PY = P[:,1]
	num_points = PY.shape[0]

	num_cluster = len(np.unique(labels))
	cluster_wise_gqs = np.zeros((num_cluster,))
	cluster_wise_valids = np.zeros((num_cluster,))

	for k in range(num_points):
		if gqs_score[k] > 20 :
			cluster_wise_valids[labels[k]] += 1
			cluster_wise_gqs[labels[k]] += gqs_score[k]

	# ids = np.unique(labels)
	# for i in ids:
	# 	mask = (labels==i)
	# 	cP = P[mask]
	# 	mean = np.mean(cP, axis=0)
 #    	sd = np.std(cP, axis=0)
 #    	filter1 = (cP > (mean - 2 * sd)) & (cP < (mean + 2 * sd))
 #    	filter2 = (cP > (mean -  sd)) & (cP < (mean + sd))
 #    	filter1 = filter1.all(axis=1)
 #    	filter2 = filter2.all(axis=1)


	cluster_wise_gqs = np.where(cluster_wise_valids>0,np.divide(cluster_wise_gqs,cluster_wise_valids),0.)
	cluster_wise_valids = np.where(cluster_wise_valids>=10,10,cluster_wise_valids)

	best_cluster = np.argmax(cluster_wise_valids+cluster_wise_gqs)

	return best_cluster


def select_top_N_grasping_points_via_top_points_method(points,gqs_score,topN = 10):

	filter_high_graspability_points = gqs_score > 15 # for real-world 30
	gqs = gqs_score[filter_high_graspability_points]
	indices = np.array(np.where(filter_high_graspability_points))[0]
	
	
	if topN > gqs.shape[0]:
		topN = gqs.shape[0]
	sub_indices = np.argpartition(gqs, -topN)[-topN:]
	indices = indices[sub_indices]

	return indices

def select_top_N_grasping_points_via_distance_sampling(points,gqs_score,topN = 10,dist_thrs=10):
	gqs = gqs_score.copy()
	topN_indices = []
	while len(topN_indices) < topN:
		topI = np.argmax(gqs)
		for topJ in topN_indices:
			dist_ij = np.linalg.norm(points[topI,0:2] - points[topJ,0:2])
			if dist_ij < dist_thrs:
				gqs[topI] = 0.0
				continue
		if gqs[topI] > 0.0:
			topN_indices.append(topI)
			gqs[topI] = 0.0

	return np.array(topN_indices)

def select_top_N_grasping_points_via_top_cluster_method(points,labels,gqs_score):
	target_cluster = choose_a_target_cluster(points,labels,gqs_score)

	cluster_filter = (labels.ravel() == target_cluster)
	cluster = np.array(points[cluster_filter, :], float32)
	gqs = np.array(gqs_score[cluster_filter], float32)
	indices = np.array(np.where(cluster_filter))[0]

	filter_low_graspability_points = gqs > 15 # for real-world 30
	indices = indices[filter_low_graspability_points]
	cluster = cluster[filter_low_graspability_points]
	gqs = gqs[filter_low_graspability_points]

	# middle position score
	mps = np.zeros(gqs.shape)
	mean = np.mean(cluster, axis=0)
	sd = np.std(cluster, axis=0)
	filter1 = (cluster > (mean - 2 * sd)) & (cluster < (mean + 2 * sd))
	filter2 = (cluster > (mean -  sd)) & (cluster < (mean + sd))
	filter1 = filter1.all(axis=1)
	filter2 = filter2.all(axis=1)
	filter1 = filter1 & (~filter2) 
	mps[filter2] = 10.0
	mps[filter1] = 5.0
	

	topN = 10
	if topN > gqs.shape[0]:
		topN = gqs.shape[0]
	sub_indices = np.argpartition(gqs+mps, -topN)[-topN:]
	indices = indices[sub_indices]

	return indices



# def draw_clusters_into_image(points,labels,gqs_score,num_cluster,image,w=320,h=240,fx=307.36,fy=307.07):
#     X = points[:,0]
#     Y = points[:,1]
#     Z = points[:,2]
	
#     print(points.shape)
#     PX = (np.divide(X,Z)*fx + w/2).astype(np.int32)
#     PY = (np.divide(Y,Z)*fx + h/2).astype(np.int32)

#     # PX = PX[PX < 320 and PX >=0
#     num_points = PY.shape[0]
#     cluster_wise_gqs = np.zeros((num_cluster,))
#     cluster_wise_valids = np.zeros((num_cluster,))

#     for k in range(num_points):
#         csize = int(gqs_score[k]/10) + 1
#         green_part = int((labels[k]*1)%255)
#         blue_part = int((labels[k]*90)%255)
#         red_part = int((labels[k]*213)%255)
#         cv2.circle(image, (int(PX[k]), int(PY[k])), csize, (blue_part,green_part,red_part), -1)
		
#         if gqs_score[k] > 20 :
#             cluster_wise_valids[labels[k]] += 1
#             cluster_wise_gqs[labels[k]] += gqs_score[k]


#     cluster_wise_gqs = np.where(cluster_wise_valids>0,np.divide(cluster_wise_gqs,cluster_wise_valids),0.)
#     # print('cluster_wise_valids',cluster_wise_valids)
#     # print('cluster_wise_gqs',cluster_wise_gqs)

#     cluster_wise_valids = np.where(cluster_wise_valids>=10,10,cluster_wise_valids)

#     P = np.zeros((PY.shape[0],3))
#     P[:,0] = PX
#     P[:,1] = PY
#     P[:,2] = Z

#     # P = np.concatenate((PX,PY),axis=0)1
#     ids = np.unique(labels)
#     for i in ids:
#         green_part = int((i*1)%255)
#         blue_part = int((i*90)%255)

#         red_part = int((i*213)%255)
#         mask = (labels==i)
#         contour = np.hstack((PX[mask][:, np.newaxis], PY[mask][:, np.newaxis]))
#         # contour = P[mask]
#         # print(contour)
#         # print(contour.shape)

#         convexHull = cv2.convexHull(contour)
#         # print(convexHull)
#         cv2.drawContours(image, [convexHull], -1, (255,0,0), 2)

#         x = PX[mask][0]
#         y = PY[mask][0]
#         dist = cv2.pointPolygonTest(convexHull,(x,y),True)
#         # cv2.imwrite('log'+'/clusters_{0}.png'.format(i),image)
#         print(i,cluster_wise_valids[i],cluster_wise_gqs[i],cluster_wise_valids[i]+cluster_wise_gqs[i])
#         # c = input('continue to press 1')
	
#     print(np.argmax(cluster_wise_valids+cluster_wise_gqs))
#     return image,P,np.argmax(cluster_wise_valids+cluster_wise_gqs)

def select_best_rectangles_gdi_old_way(rectangle_list,GDI_calculator_all,top_rectangles_needed=3):
	rectangle_array = np.array(rectangle_list)
	GDI = []
	for gdi2 in GDI_calculator_all:
		GDI.append(gdi2.calculate_gdi_score_old_way())
	GDI_array = np.array(GDI)

	selected_idx = np.argpartition(GDI_array, -top_rectangles_needed)[-top_rectangles_needed:] # indices of top k elements
	selected_idx = selected_idx[np.argsort(GDI_array[selected_idx])[::-1]] # sorting the top k indices

	selected_rectangles = rectangle_array[selected_idx]
	GDI_plus_array = np.zeros(GDI_array.shape)
	return selected_rectangles,selected_idx,GDI_array,GDI_plus_array
# rospy.wait_for_service('point_cloud_access_service')
# get_3d_cam_point = rospy.ServiceProxy('point_cloud_access_service', point_cloud_service)


def query_point_cloud_client(x, y):
	rospy.wait_for_service('point_cloud_access_service')
	try:
		get_3d_cam_point = rospy.ServiceProxy('point_cloud_access_service', point_cloud_service)
		resp = get_3d_cam_point(np.array([x, y]))
		return resp.cam_point
	except rospy.ServiceException as e:
		print("Point cloud Service call failed: %s"%e)


def final_axis_angle(points):
	
	cx = np.mean(points[:, 0])
	cy = np.mean(points[:, 1])
	x34 = 0.5*(points[3,0]+points[4,0])

	y34 = 0.5*(points[3,1]+points[4,1])
	if y34-cy == 0.0:
		angle = pi/2
	else:
		angle = atan((cx-x34)/(y34-cy))
	return angle

def keep_angle_bounds(angle):
	# print('raw angle',angle)
	if angle > radians(90):
		angle = angle - radians(180)
	elif angle < radians(-90):
		angle = angle + radians(180)
	assert angle >= radians(-90) and angle <= radians(90)
	return angle


def height_difference_consideration(best_rectangles, their_idx,darray):
	if len(their_idx) < 2:
		return best_rectangles, their_idx
	new_best_rectangle = best_rectangles.copy()
	if (darray[best_rectangles[0][0][1],best_rectangles[0][0][0]] - darray[best_rectangles[1][0][1],best_rectangles[1][0][0]]) > 0.1:
		new_best_rectangle[0] = best_rectangles[1]
		new_best_rectangle[1] = best_rectangles[0]
		temp = their_idx[0]
		their_idx[0] = their_idx[1]
		their_idx[1] = their_idx[0]
	elif len(their_idx) > 2 and darray[best_rectangles[0][0][1],best_rectangles[0][0][0]] - darray[best_rectangles[2][0][1],best_rectangles[2][0][0]] > 0.1:
		new_best_rectangle[0] = best_rectangles[2]
		new_best_rectangle[2] = best_rectangles[0]
		temp = their_idx[0]
		their_idx[0] = their_idx[2]
		their_idx[2] = their_idx[0]
	# print('height_difference_consideration', their_idx)
	return new_best_rectangle, their_idx

def select_best_rectangles(rectangle_list,GDI,GDI_plus,GCS=None,top_rectangles_needed=3,final_attempt=False, inputs=None,angle_list=None):
	if len(GDI) < top_rectangles_needed:
		top_rectangles_needed = len(GDI)
	rectangle_array = np.array(rectangle_list)
	GDI_array_org = np.array(GDI)
	GDI_plus_array = np.array(GDI_plus)
	# GDI_plus_array = np.zeros(GDI_plus_array.shape)
	GDI_array = GDI_plus_array.astype(float) + GDI_array_org
	if GCS is not None:
		GDI_array += GCS # most likely GQS is none
	# GDI_array = GDI_array_org
	# selected_idx = np.argpartition(GDI_array, -top_rectangles_needed)[-top_rectangles_needed:] # indices of top k elements
	# selected_idx = selected_idx[np.argsort(GDI_array[selected_idx])[::-1]] # sorting the top k indices
	


	# selection of appropiate grasp poses based on analytical reasoning and cnn prediction
	if inputs is not None:
		loop_count = 0
		count = 0
		
		selected_idx = []
		
		while count < top_rectangles_needed and loop_count < GDI_array.shape[0]:
			pivot = np.argmax(GDI_array)
			# print(pivot)
			centroid = rectangle_list[pivot][0]
			d = inputs['darray'][centroid[1],centroid[0]]
			centroid_xyz = pixel_to_xyz(centroid[0],centroid[1],d)
			angle = angle_list[pivot]
			valid = True #check_for_validity_by_cnn(inputs,centroid_xyz,angle)
			if valid:
				selected_idx.append(pivot)
				count += 1
			GDI_array[pivot] = float('-inf') # from math
			loop_count += 1
	selected_idx = np.array(selected_idx)
	selected_rectangles = rectangle_array[selected_idx]
	return selected_rectangles,selected_idx

def draw_grasp_pose_as_a_line(img, rectangle):
	centroid = rectangle[0]

	a1,b1 = ((rectangle[1] + rectangle[2])/2).astype(int)
	a2,b2 = ((rectangle[3] + rectangle[4])/2).astype(int)
	cv2.line(img, (a1,b1), (a2,b2), color=(69,24,255), thickness=2)
	cv2.circle(img, (int(centroid[0]), int(centroid[1])), 5, (255,0,0), -1)

def draw_rectified_rect(img, pixel_points,path=None,gdi=None,gdi_plus=0,color=(0, 255, 0),pos=(10,20),gdi_positives=None,gdi_negatives=None,gdi_plus_positives=None,gdi_plus_negatives=None,thickness = 4):
	# print(pixel_points)
	pixel_points = np.array(pixel_points,dtype=np.int16)
	color = (0,255,0)

	color1 = (69,24,255)
	color = (25,202,242)
	cv2.line(img, (pixel_points[1][0], pixel_points[1][1]), (pixel_points[2][0], pixel_points[2][1]),
			 color=color1, thickness=thickness)
	cv2.line(img, (pixel_points[2][0], pixel_points[2][1]), (pixel_points[3][0], pixel_points[3][1]),
			 color=color, thickness=thickness)
	cv2.line(img, (pixel_points[3][0], pixel_points[3][1]), (pixel_points[4][0], pixel_points[4][1]),
			 color=color1, thickness=thickness)
	cv2.line(img, (pixel_points[4][0], pixel_points[4][1]), (pixel_points[1][0], pixel_points[1][1]),
			 color=color, thickness=thickness)

	# cv2.line(img, (int((pixel_points[1][0]+pixel_points[2][0])/2), int((pixel_points[1][1]+pixel_points[2][1])/2)), 
	# 		 (int((pixel_points[3][0]+pixel_points[4][0])/2), int((pixel_points[3][1]+pixel_points[4][1])/2)),color=color, thickness=3)
	
	cv2.circle(img, (pixel_points[0][0], pixel_points[0][1]), 4,(255,0,0), -1)
	# mid_2_3 = [int((pixel_points[2][0]+pixel_points[3][0])/2),int((pixel_points[2][1]+pixel_points[3][1])/2)]
	# mid_23_c = [int((pixel_points[0][0]+mid_2_3[0])/2),int((pixel_points[0][1]+mid_2_3[1])/2)]
	if gdi is not None:
		cv2.putText(img, '({0},{1})'.format(gdi,gdi_plus), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
	if gdi_positives is not None:
		for point in gdi_positives:
			cv2.circle(img, (point[0],point[1]),1,(255,255,255), -1)
		for point in gdi_negatives:
			cv2.circle(img, (point[0],point[1]),1,(0,0,0), -1)
		for point in gdi_plus_positives:
			cv2.circle(img, (point[0],point[1]),1,(255,255,0), -1)
		for point in gdi_plus_negatives:
			cv2.circle(img, (point[0],point[1]),1,(0,255,255), -1)
	if path is not None:
		cv2.imwrite(path, img)
	# cv2.putText(img, name, (pixel_points[0][0], pixel_points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 3)

def draw_rectified_rect_plain(image_plain, pixel_points,color=(0, 0, 255), index=None):
	# print('plotting here')
	pixel_points = np.array(pixel_points,dtype=np.int16)
	cv2.line(image_plain, (pixel_points[1][0], pixel_points[1][1]), (pixel_points[2][0], pixel_points[2][1]),
			 color=color, thickness=1)
	cv2.line(image_plain, (pixel_points[2][0], pixel_points[2][1]), (pixel_points[3][0], pixel_points[3][1]),
			 color, thickness=1)
	cv2.line(image_plain, (pixel_points[3][0], pixel_points[3][1]), (pixel_points[4][0], pixel_points[4][1]),
			 color=color, thickness=1)
	cv2.line(image_plain, (pixel_points[4][0], pixel_points[4][1]), (pixel_points[1][0], pixel_points[1][1]),
			 color=color, thickness=1)
	cv2.circle(image_plain, (pixel_points[0][0], pixel_points[0][1]), 2,color, -1)
	if index is not None:
		cv2.putText(image_plain, str(index), (pixel_points[0][0]+2, pixel_points[0][1]+2), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
	return image_plain



def normalize_gdi_score(gdi):
	gdi = np.array(gdi).astype(float)
	gdi = (100*(gdi/gdi.max())).astype(np.int8)
	return gdi

def interpolate_noisy_2d_map(map):
	points = np.where(map != 0)
	values = map[points]
	xi = np.where(map == 0)
	map[xi] = griddata(points, values, xi, method='nearest')
	return map

def draw_a_depth_image(dmap,path):
	dmap_vis = (dmap / dmap.max())*255
	counts, bins = np.histogram(dmap_vis)
	# print(counts)
	# print(bins)
	plt.hist(bins[:-1], bins, weights=counts)
	# plt.savefig(path+'/hist1.png')
	dmap_vis[np.where(dmap_vis < bins[1])] = 255
	dmap_vis = ((dmap_vis-dmap_vis.min())/dmap_vis.max())*255
	cv2.imwrite(path,dmap_vis)

def draw_samples(img,pixels):
	try:
		l,_ = pixels.shape
		for i in range(l):
			cx = int(pixels[i][0])
			cy = int(pixels[i][1])
			cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)
	except:
		print('no filtered pixels')
	return img

def create_directory(dname):
	if not os.path.exists(dname):
		print('creating directory:',dname)
		os.makedirs(dname)

def draw_grasp_map1(grasp_map,image):
	m,n = grasp_map.shape
	for i in range(m):
		for j in range(n):
			if grasp_map[i][j] > 0:
				color = (255,0,0)
			else:
				color = (0,0,255)
			cv2.circle(image, (2*j, 2*i), 1, color, -1)

def draw_grasp_map(grasp_map,path):

	# heatmap = plt.imshow( grasp_map , cmap = 'coolwarm' , interpolation = 'nearest' )
	# plt.colorbar(heatmap)
	# plt.savefig(path)

	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.imshow( grasp_map , cmap = 'coolwarm' , interpolation = 'nearest' )
	plt.savefig(path)

def draw_top_N_points(top_grasp_points,image):
	for k,point in enumerate(top_grasp_points):
		color = (0,0,255)
		cv2.circle(image, (int(point[0]), int(point[1])), 5, (255,0,0), -1)
		cv2.circle(image, (int(point[0]), int(point[1])), 8, color , 2)


def get_seg_mask(mask):
	n,w,h = mask.shape
	seg_mask = np.zeros((w,h))
	for i in range(n):
		seg_mask = np.where(mask[i],i+1,seg_mask)
	return seg_mask

def get_pivot_heat_value( value, minimum=0, maximum=100):
	if value < 0:
		value = 0
	minimum, maximum = float(minimum), float(maximum)
	ratio = 2 * (value-minimum) / (maximum - minimum)
	b = int(max(0, 255*(1 - ratio)))
	r = int(max(0, 255*(ratio - 1)))
	g = 255 - b - r
	return r, g, b

def draw_pivots_n_gcs_scores(pivots,scores,img):
	for i,pivot in enumerate(pivots):
		print(i,scores[i])
		score = scores[i]
		r,g,b = (255,255,255) #get_pivot_heat_value(score)
		cx = int(pivot[0])
		cy = int(pivot[1])
		cv2.circle(img,(cx,cy),7, (r,g,b), thickness=-1)
	return img


class Parameters:
	def __init__(self,w,h):

		self.cut_length = 70
		self.w = w #int(self.mw*200)
		self.h = h #int(self.mh*200)
		self.mw = float(w)/200 #1.6
		self.mh = float(h)/200 #1.2
		# if len(sys.argv) > 2:
		#     mw = int(sys.argv[1])/200
		#     mh = int(sys.argv[2])/200
		
		self.hfov = 69.4 #50 #54.3 # 55.0
		self.vfov = 42.5 #42.69
		self.pi = 3.14
		self.f_x = self.mw*192 #w/(2*tan(hfov*pi/360))
		self.f_y = self.mh*256 #h/(2*tan(vfov*pi/360))
		# print('focal length',f_x,f_y)
		# print('gripper width',2.0*f_x/53.0) # = 8 pixels
		# print('gripper coverage in free space',1.5*f_x/53.0) # = 6 pixels
		# print('total space',200*53.0/f_x) # = 6 pixels
		# if len(sys.argv) > 2 :
		#     self.THRESHOLD1 = int(sys.argv[1])#20
		#     self.THRESHOLD2 = 0.01*int(sys.argv[2])#0.02
		#     self.THRESHOLD3 = int(sys.argv[3])
		# else:
		self.THRESHOLD1 = int(self.mh*15)
		self.THRESHOLD2 = 0.02
		self.THRESHOLD3 = int(self.mh*7)

		self.gripper_width = int(self.mh*15)
		self.gripper_height = int(self.mh*70)  
		self.gripper_max_opening_length = 0.15 #0.133
		self.gripper_finger_space_max = 0.12 #0.103
		# gripper_max_free_space = 35
		self.gdi_max = int(self.gripper_height/2)
		self.gdi_plus_max = 2*(self.gripper_width/2)*self.THRESHOLD3
		self.cx = int(self.gripper_width/2)
		self.cy = int(self.gripper_height/2)
		self.pixel_finger_width = self.mh*8 # width in pixel units.
		# GDI_calculator = []
		# GDI_calculator_all = []
		self.Max_Gripper_Opening_value = 1.0
		self.datum_z = 0.54 #0.640 # empty bin depth value
		self.gdi_plus_cut_threshold = 60 #70

		self.cone_thrs = 200.0

		self.crop_radius_pixels = int(45*self.mh)

		self.DBCC_enable = True

		self.angle_shift_list = []
		self.asc_list = []


	def pixel_to_xyz(self,px,py,z):
		#cartesian coordinates
		if px < 0:
			px = 0
		if py < 0:
			py = 0
		if px > self.w-1:
			px = self.w-1
		if py > self.h-1:
			py = self.h-1
		# z = darray[py][px]
		x = (px - (self.w/2))*(z/(self.f_x))
		y = (py - (self.h/2))*(z/(self.f_y))
		return x,y 

	def axis_angle(self, points):
		# major_axis_length = int(param.mh*150)
		minor_axis_length = self.gripper_height
		cx = np.mean(points[:, 0])
		cy = np.mean(points[:, 1])
		modi_x = np.array(points[:, 0] - cx)
		modi_y = np.array(points[:, 1] -  cy)
		num = np.sum(modi_x * modi_y)
		den = np.sum(modi_x ** 2 - modi_y ** 2)
		angle = 0.5 * atan2(2 * num, den)  # between -180 to 180
		# [x1_ma, y1_ma] = [int(cx + 0.5 * major_axis_length * cos(angle)),
						  # int(cy + 0.5 * major_axis_length * sin(angle))]
		# [x2_ma, y2_ma] = [int(cx - 0.5 * major_axis_length * cos(angle)),
						  # int(cy - 0.5 * major_axis_length * sin(angle))]
		[x1_mi, y1_mi] = [int(cx + 0.5 * minor_axis_length * cos(angle + radians(90))),
						  int(cy + 0.5 * minor_axis_length * sin(angle + radians(90)))]
		[x2_mi, y2_mi] = [int(cx - 0.5 * minor_axis_length * cos(angle + radians(90))),
						  int(cy - 0.5 * minor_axis_length * sin(angle + radians(90)))]
		
		# calculate the angle in terms of minor axis angle with downward vertical direction (positive x-axis in image reference frame)
		# angle = angle + radians(90)
		# angle = keep_angle_bounds (angle) # since geometrical symmetry exist in angle space, we keep the angles in [-90,90]. 
		# vertical downward is zero. As you move right (towards positive y in image frame, angle goes towards +90, in opposite direction angle goes to -90)
		axis_dict = {
			# "major_axis_points": [(x1_ma, y1_ma), (x2_ma, y2_ma)],
			"minor_axis_points": [(x1_mi, y1_mi), (x2_mi, y2_mi)],
			"angle": angle,
			"centroid": (cx, cy)}
		return axis_dict


	def draw_rect(self, centroid, angle,color=(0, 0, 255),directions=1):
		angle_org = angle
		return_list = []
		angle_list = []
		for i in range(directions):
			if i == 1:
				angle = angle_org + radians(45)
			elif i == 2:
				angle = angle_org - radians(45) 
			elif i == 3:
				angle = angle_org - radians(90)
			angle = keep_angle_bounds(angle) 
			[x1, y1] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] - self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
			[x2, y2] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] - self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
			[x3, y3] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] + self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
			[x4, y4] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] + self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
			angle_list.append(angle)
			return_list.append(np.array([[int(centroid[0]), int(centroid[1])], [x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
		return return_list, angle_list, centroid  


	def calculate_asc_indices(self,directions=1):
		u = 180/directions
		start_angle = -90
		for i in range(directions):
			angle_shift = start_angle+(i+1)*u
			asc = 1 - abs(angle_shift/u)/directions
			# print('asc:',asc)
			self.angle_shift_list.append(angle_shift)
			self.asc_list.append(asc)

			# self.angle_shift_list.append(0.0)
			# self.asc_list.append(1)
		# print(self.angle_shift_list)
		# print(self.asc_list)


	def draw_rect_generic(self, centroid, angle,color=(0, 0, 255),directions=1):
		# Please note: this function is not properly implemented, due to mismatch in angle calculations.
		# Either look for original vibhu code or reimplement it.
		# angle_org = keep_angle_bounds(angle)

		# calculate the angle in terms of minor axis angle with downward vertical direction (positive x-axis in image reference frame)
		# angle = angle + radians(90)
		# angle = keep_angle_bounds (angle) # since geometrical symmetry exist in angle space, we keep the angles in [-90,90]. 
		# vertical downward is zero. As you move right (towards positive y in image frame, angle goes towards +90, in opposite direction angle goes to -90)
		# angle = -angle

		angle_org = -angle # the existing formulations work with this simple hack found through manual observations.
		# To find out the exact bug, all the calculations have to be redone.

		return_list = []
		angle_list = []
		# asc_list = [] # asc - angle stability cofficient
		# u = 180/directions
		# start_angle = -90
		for i in range(directions):
			angle_shift = self.angle_shift_list[i]
			angle = angle_org + radians(angle_shift)

			# asc = 1 - abs(angle_shift/u)/directions
			# print('asc:',asc)
			# asc_list.append(asc)
			# if i == 1:
			# 	angle = angle_org + radians(45)
			# elif i == 2:
			# 	angle = angle_org - radians(45) 
			# elif i == 3:
			# 	angle = angle_org - radians(90) 
			angle = keep_angle_bounds(angle) 
			# print(angle_org,angle)

			# [x1, y1] = [int(centroid[0] - self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * cos(angle )),
			# 			int(centroid[1] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * sin(angle ))]
			# [x2, y2] = [int(centroid[0] + self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * cos(angle )),
			# 			int(centroid[1] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * sin(angle))]
			# [x3, y3] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * sin(angle)),
			# 			int(centroid[1] + self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * cos(angle))]
			# [x4, y4] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * sin(angle)),
			# 			int(centroid[1] - self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * cos(angle))]


			# [x1, y1] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * sin(angle )),
			# 			int(centroid[1] + self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * cos(angle ))]
			# [x2, y2] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * sin(angle)),
			# 			int(centroid[1] - self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * cos(angle ))]
			# [x3, y3] = [int(centroid[0] + self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * cos(angle)),
			# 			int(centroid[1] + self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * sin(angle))]
			# [x4, y4] = [int(centroid[0] + self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * cos(angle)),
			# 			int(centroid[1] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * sin(angle))]



			# [x1, y1] = [int(centroid[0] - self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * cos(angle )),
			# 			int(centroid[1] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * sin(angle ))]
			# [x2, y2] = [int(centroid[0] + self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * cos(angle)),
			# 			int(centroid[1] + self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * sin(angle ))]
			# [x3, y3] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * sin(angle)),
			# 			int(centroid[1] + self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * cos(angle))]
			# [x4, y4] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * sin(angle)),
			# 			int(centroid[1] - self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * cos(angle))]


			[x1, y1] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] - self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
			[x2, y2] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] - self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
			[x3, y3] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] + self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
			[x4, y4] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] + self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
			# angle = angle - radians(90)
			# angle = keep_angle_bounds (angle)
			angle_list.append(angle)
			return_list.append(np.array([[int(centroid[0]), int(centroid[1])], [x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
		return return_list, angle_list, self.asc_list

	def draw_rect_generic_fix_angles(self, centroid,color=(0, 0, 255),directions=6):
		return_list = []
		angle_list = []
		u = int(180/directions)
		for i in range(0,180,u):
			angle = radians(i)
			angle = keep_angle_bounds(angle) 
			[x1, y1] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] - self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
			[x2, y2] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] - self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
			[x3, y3] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] + self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
			[x4, y4] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
						int(centroid[1] + self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
			angle_list.append(angle)
			return_list.append(np.array([[int(centroid[0]), int(centroid[1])], [x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
		return return_list, angle_list, self.asc_list

	def draw_rect_over_image(self,rectangle,image,color=[0,0,255]):
		centroid = rectangle[0]
		[x1,y1] = rectangle[1]
		[x2,y2] = rectangle[2]
		[x3,y3] = rectangle[3]
		[x4,y4] = rectangle[4]

		cv2.line(image, (x1,y1), (x2,y2),
			 color=color, thickness=2)
		cv2.line(image, (x2,y2), (x3,y3),
				 color=color, thickness=2)
		cv2.line(image, (x3,y3), (x4,y4),
				 color=color, thickness=2)
		cv2.line(image, (x4,y4), (x1,y1),
				 color=color, thickness=2)
		cv2.circle(image, (centroid[0], centroid[1]), 3,color, -1)
		return image

	def draw_rect_cnn(self,image, centroid, angle,width,color=(0, 0, 255)):
		angle = keep_angle_bounds(angle) 
		[x1, y1] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] - self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
		[x2, y2] = [int(centroid[0] - self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] - self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
		[x3, y3] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] + self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
		[x4, y4] = [int(centroid[0] + self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] + self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
		rectangle_outer = np.array([[int(centroid[0]), int(centroid[1])], [x1, y1], [x2, y2], [x3, y3], [x4, y4]])
		image = self.draw_rect_over_image(rectangle_outer,image,color)

		[x1, y1] = [int(centroid[0] - width*self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] - width*self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
		[x2, y2] = [int(centroid[0] - width*self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] - width*self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
		[x3, y3] = [int(centroid[0] + width*self.gripper_height * 0.5 * cos(angle) + self.gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] + width*self.gripper_height * 0.5 * sin(angle) + self.gripper_width * 0.5 * sin(angle + radians(90)))]
		[x4, y4] = [int(centroid[0] + width*self.gripper_height * 0.5 * cos(angle) - self.gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] + width*self.gripper_height * 0.5 * sin(angle) - self.gripper_width * 0.5 * sin(angle + radians(90)))]
		rectangle_inner = np.array([[int(centroid[0]), int(centroid[1])], [x1, y1], [x2, y2], [x3, y3], [x4, y4]])
		return self.draw_rect_over_image(rectangle_inner,image,color=(0,0,0))


	def draw_rect_gqcnn(self,image,centroid, angle,color=(0, 0, 255), scale = 1):
		gripper_height = self.gripper_height-20
		gripper_width = self.gripper_width
		[x1, y1] = [int(centroid[0] - gripper_height * 0.5 * cos(angle) - gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] - gripper_height * 0.5 * sin(angle) - gripper_width * 0.5 * sin(angle + radians(90)))]
		[x2, y2] = [int(centroid[0] - gripper_height * 0.5 * cos(angle) + gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] - gripper_height * 0.5 * sin(angle) + gripper_width * 0.5 * sin(angle + radians(90)))]
		[x3, y3] = [int(centroid[0] + gripper_height * 0.5 * cos(angle) + gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] + gripper_height * 0.5 * sin(angle) + gripper_width * 0.5 * sin(angle + radians(90)))]
		[x4, y4] = [int(centroid[0] + gripper_height * 0.5 * cos(angle) - gripper_width * 0.5 * cos(angle + radians(90))),
					int(centroid[1] + gripper_height * 0.5 * sin(angle) - gripper_width * 0.5 * sin(angle + radians(90)))]
		# angle_list.append(angle)
		# return_list.append(np.array([[int(centroid[0]), int(centroid[1])], [x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
		# color1 = (255,255,0) # cyan
		# # color = (0,160,255) # orange
		# color = (178, 83, 70) # Liberty
		color1 = (69,24,255)
		color = (25,202,242)
		# color = (255,80,0)
		cv2.line(image, (scale*x1,scale*y1), (scale*x2,scale*y2),
			 color=color1, thickness=4)
		cv2.line(image, (scale*x2,scale*y2), (scale*x3,scale*y3),
				 color=color, thickness=4)
		cv2.line(image, (scale*x3,scale*y3), (scale*x4,scale*y4),
				 color=color1, thickness=4)
		cv2.line(image, (scale*x4,scale*y4), (scale*x1,scale*y1),
				 color=color, thickness=4)
		cv2.circle(image, (scale*centroid[0], scale*centroid[1]), 4,color1, -1)
		return image, np.array([[int(centroid[0]), int(centroid[1])], [x1, y1], [x2, y2], [x3, y3], [x4, y4]])



	def median_depth_based_filtering(self,darray,median_depth_map, filter_ratio=0.95):
		filtered = []
		mask = ((median_depth_map - darray) > self.THRESHOLD2) &  (darray!=0)
		objectness_ratio = float(np.count_nonzero(mask))/(self.w*self.h)
		for i in range(self.w):
			for j in range(self.h):
				if mask[j][i] and np.random.random() > filter_ratio:
					filtered.append([i,j])
		return np.array(filtered), objectness_ratio
																				  

	def sample_random_grasp_pose(self,point):
		minor_axis_length = self.gripper_height
		cx = point[0]
		cy = point[1]
		angle = np.random.uniform(-pi/2,pi/2)
		[x1_mi, y1_mi] = [int(cx + 0.5 * minor_axis_length * cos(angle + radians(90))),
						  int(cy + 0.5 * minor_axis_length * sin(angle + radians(90)))]
		[x2_mi, y2_mi] = [int(cx - 0.5 * minor_axis_length * cos(angle + radians(90))),
						  int(cy - 0.5 * minor_axis_length * sin(angle + radians(90)))]
		axis_dict = {
			"minor_axis_points": [(x1_mi, y1_mi), (x2_mi, y2_mi)],
			"angle": angle,
			"centroid": (cx, cy)}
		return axis_dict

	def random_colour_masks(self,image):
		colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
		r = np.zeros_like(image).astype(np.uint8)
		g = np.zeros_like(image).astype(np.uint8)
		b = np.zeros_like(image).astype(np.uint8)
		r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
		coloured_mask = np.stack([r, g, b], axis=2)
		return coloured_mask

	def draw_minor_axis(self,minor_points,mask,score,gcs_score,image,color=(0, 0, 255),color1=(255, 0, 255),color2=(255, 0, 0)):
		cv2.line(image, (minor_points[0][1],minor_points[0][0]), (minor_points[1][1],minor_points[1][0]),
				 color=color, thickness=4)
		cv2.circle(image, (minor_points[0][1],minor_points[0][0]), 4,color1, -1)
		cv2.circle(image, (minor_points[1][1],minor_points[1][0]), 4,color2, -1)	

		rgb_mask = self.random_colour_masks(mask)
		image = cv2.addWeighted(image, 0.25, rgb_mask, 0.75, 0)	

		#adding text
		font = cv2.FONT_HERSHEY_SIMPLEX 
		org = (50, 50)
		fontScale = 1 
		color = (255, 0, 0)
		thickness = 2
		cv2.putText(image, '{0:0.2f},{1:0.2f}'.format(score,gcs_score), org, font,fontScale, color, thickness, cv2.LINE_AA)
		# cv2.putText(image, (str(gcs_score)), org, font,fontScale, color, thickness, cv2.LINE_AA) 

		return image
