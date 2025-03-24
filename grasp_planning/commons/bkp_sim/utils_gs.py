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

def select_best_rectangles(rectangle_list,GDI,GDI_plus,top_rectangles_needed=3,final_attempt=False):
    if len(GDI) < top_rectangles_needed:
        top_rectangles_needed = len(GDI)
    rectangle_array = np.array(rectangle_list)
    GDI_array_org = np.array(GDI)
    GDI_plus_array = np.array(GDI_plus)
    # GDI_plus_array = np.zeros(GDI_plus_array.shape)
    GDI_array = GDI_array_org+GDI_plus_array
    # GDI_array = GDI_array_org
    selected_idx = np.argpartition(GDI_array, -top_rectangles_needed)[-top_rectangles_needed:] # indices of top k elements
    selected_idx = selected_idx[np.argsort(GDI_array[selected_idx])[::-1]] # sorting the top k indices
    selected_rectangles = rectangle_array[selected_idx]
    return selected_rectangles,selected_idx

def draw_rectified_rect(img, pixel_points,path=None,gdi=None,gdi_plus=0,color=(0, 0, 255),pos=(10,20),gdi_positives=None,gdi_negatives=None,gdi_plus_positives=None,gdi_plus_negatives=None):
    # print(pixel_points)
    pixel_points = np.array(pixel_points,dtype=np.int16)
    cv2.line(img, (pixel_points[1][0], pixel_points[1][1]), (pixel_points[2][0], pixel_points[2][1]),
             color=color, thickness=2)
    cv2.line(img, (pixel_points[2][0], pixel_points[2][1]), (pixel_points[3][0], pixel_points[3][1]),
             color=color, thickness=2)
    cv2.line(img, (pixel_points[3][0], pixel_points[3][1]), (pixel_points[4][0], pixel_points[4][1]),
             color=color, thickness=2)
    cv2.line(img, (pixel_points[4][0], pixel_points[4][1]), (pixel_points[1][0], pixel_points[1][1]),
             color=color, thickness=2)
    cv2.circle(img, (pixel_points[0][0], pixel_points[0][1]), 3,color, -1)
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
    gdi = np.array(gdi).astype(np.float)
    gdi = (100*(gdi/gdi.max())).astype(np.int8)
    return gdi

class Parameters:
    def __init__(self):
       
        self.mw = 1.6
        self.mh = 1.2
        # if len(sys.argv) > 2:
        #     mw = int(sys.argv[1])/200
        #     mh = int(sys.argv[2])/200
        self.w = int(self.mw*200)
        self.h = int(self.mh*200)
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
        self.THRESHOLD2 = 0.01
        self.THRESHOLD3 = int(self.mh*7)

        self.gripper_width = int(self.mh*15)
        self.gripper_height = int(self.mh*70)  
        self.gripper_max_opening_length = 0.133
        self.gripper_finger_space_max = 0.103
        # gripper_max_free_space = 35
        self.gdi_max = int(self.gripper_height/2)
        self.gdi_plus_max = 2*(self.gripper_width/2)*self.THRESHOLD3
        self.cx = int(self.gripper_width/2)
        self.cy = int(self.gripper_height/2)
        self.pixel_finger_width = self.mw*8 # width in pixel units.
        # GDI_calculator = []
        # GDI_calculator_all = []
        self.Max_Gripper_Opening_value = 1.0
        self.datum_z = 0.54 #0.640 # empty bin depth value
        self.gdi_plus_cut_threshold = 60 #70

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
        modi_y = np.array(points[:, 1] - cy)
        num = np.sum(modi_x * modi_y)
        den = np.sum(modi_x ** 2 - modi_y ** 2)
        angle = 0.5 * atan2(2 * num, den)
        # [x1_ma, y1_ma] = [int(cx + 0.5 * major_axis_length * cos(angle)),
                          # int(cy + 0.5 * major_axis_length * sin(angle))]
        # [x2_ma, y2_ma] = [int(cx - 0.5 * major_axis_length * cos(angle)),
                          # int(cy - 0.5 * major_axis_length * sin(angle))]
        [x1_mi, y1_mi] = [int(cx + 0.5 * minor_axis_length * cos(angle + radians(90))),
                          int(cy + 0.5 * minor_axis_length * sin(angle + radians(90)))]
        [x2_mi, y2_mi] = [int(cx - 0.5 * minor_axis_length * cos(angle + radians(90))),
                          int(cy - 0.5 * minor_axis_length * sin(angle + radians(90)))]
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

    def median_depth_based_filtering(darray,median_depth_map):
        filtered = []
        mask = ((median_depth_map - darray) > self.THRESHOLD2) &  (darray!=0)
        for i in range(w):
            for j in range(h):
                if mask[j][i] and np.random.random()>0.95:
                    filtered.append([i,j])
        return np.array(filtered)


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

