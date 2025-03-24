import numpy as np
from math import *
import cv2
import copy

class GDI2:
    def __init__(self,rotation_point,angle,darray,param):
        self.rotation_point = rotation_point
        x = rotation_point[0]
        y = rotation_point[1]
        t = angle
        self.rotation_matrix = np.array([[cos(t), -sin(t), -x*cos(t)+y*sin(t)+x], [sin(t), cos(t), -x*sin(t)-y*cos(t)+y], [0,0,1]])
        self.tx = x - int(param.gripper_height/2)
        self.ty = y - int(param.gripper_width/2)
        self.bmap = None
        self.dmap = None
        self.new_centroid = np.array([7,35])
        self.gripper_opening = param.gripper_height
        self.gdi_score_old_way = 0
        self.gripper_opening_meter = 0.1
        self.object_width = 0.05
        self.boundary_pose = False
        self.min_depth_difference = 0.03
        self.laplacian = None
        self.param = param
        self.darray = darray
        self.dmap_fwd = None
        self.dmap_bkd = None

    def rotate(self,point):
        point_homo = np.ones(3)
        point_homo[0:2] = point
        new_point = np.matmul(self.rotation_matrix,point_homo)
        return int(new_point[0]), int(new_point[1])

    def map_the_point(self,i,j):
        xp = i+self.tx
        yp = j+self.ty
        mapped_loc = self.rotate(np.array([xp,yp]))
        xo,yo = mapped_loc
        # if xo<0:
        #     xo=0
        # elif xo > 199:
        #     xo = 199
        # if yo<0:
        #     yo=0
        # elif yo > 199:
        #     yo = 199
        return xo,yo

    def calculate_gdi_score_old_way(self):
        cy = self.param.cy
        bmap = self.bmap
        gdi_count = bmap[:,0:cy-self.param.THRESHOLD1].sum() + bmap[:,cy+self.param.THRESHOLD1:].sum()
        gdi_count_normalized = int(100*gdi_count/self.param.gdi_max)
        self.gdi_score_old_way = gdi_count_normalized
        return self.gdi_score_old_way

    def calculate_gdi_plus_score(self):
        bmap = self.bmap
        gdi_plus_count = self.param.gdi_plus_max - bmap[:,self.new_centroid[1]-self.param.THRESHOLD3:self.new_centroid[1]+self.param.THRESHOLD3].sum()
        gdi_plus_count_normalized = int(100*gdi_plus_count/self.param.gdi_plus_max)
        if gdi_plus_count_normalized < self.param.gdi_plus_cut_threshold:
            return None
        else:
            return gdi_plus_count_normalized

    def calculate_gdi_plus_score_better_way(self):
        bmap = self.bmap
        gdi_plus_count_upper = self.param.gdi_plus_max - bmap[:self.new_centroid[0],self.new_centroid[1]-self.param.THRESHOLD3:self.new_centroid[1]+self.param.THRESHOLD3].sum()
        gdi_plus_count_lower = self.param.gdi_plus_max - bmap[self.new_centroid[0]:,self.new_centroid[1]-self.param.THRESHOLD3:self.new_centroid[1]+self.param.THRESHOLD3].sum()
        gdi_plus_count = min(int(gdi_plus_count_upper),int(gdi_plus_count_lower))
        gdi_plus_count_normalized = int(100*gdi_plus_count/self.param.gdi_plus_max)
        if gdi_plus_count_normalized < self.param.gdi_plus_cut_threshold:
            # print('rejected',gdi_plus_count_normalized,gdi_plus_count_upper,gdi_plus_count_lower)
            return None
        else:
            # print('selected',gdi_plus_count_normalized,gdi_plus_count_upper,gdi_plus_count_lower)
            return gdi_plus_count_normalized


    # def calculate_gdi_plus_score_new_way(self):
    #     cy = self.new_centroid[1]
    #     s = cy - int(self.gripper_opening/2) + 1
    #     e = cy + int(self.gripper_opening/2) 
    #     # print(cy,s,e,self.gripper_opening)
    #     total_score = 0
    #     completeness_count = 0
    #     for y in range(s,e):
    #         total_score += gripper_width - self.bmap[:,y].sum()
    #         if self.bmap[:,y].sum() == 0:
    #             completeness_count += 1

    #     completeness_score = completeness_count/(e-s)
    #     avg_score =  float(total_score)/(e-s)
    #     gdi_plus_count_normalized = int(50*(avg_score/gripper_width)+50*completeness_score)
    #     # if gdi_plus_count_normalized < 10:
    #     #     return None
    #     # else:
    #     return gdi_plus_count_normalized


    def calculate_pixel_meter_ratio(self,FRs,FLs):
        x,y = self.map_the_point(int((3*FLs+FRs)/4),cx)
        # x2,y2 = self.map_the_point(FRs,cx)
        px = self.rotation_point[0]
        py = self.rotation_point[1]

        # X,Y,_ = query_point_cloud_client(3.2*x, 2.4*y)
        # pX,pY,_ = query_point_cloud_client(3.2*px, 2.4*py)

        z = self.darray[y][x]
        X = (x - (w/2))*(z/(f_x))
        Y = (y - (h/2))*(z/(f_y))
        z = self.darray[py][px]
        pX = (px - (w/2))*(z/(f_x))
        pY = (py - (h/2))*(z/(f_y))
        # # print('pixels',x1,x2,y1,y2,z)
        # # print('meter',X1,X2,Y1,Y2)
        # print(sqrt((x1-x2)**2 + (y1-y2)**2))
        d = sqrt((X-pX)**2 + (Y-pY)**2)
        d_pixel = (FRs-FLs)/4

        meter_to_pixel_ratio = d/d_pixel 

        return meter_to_pixel_ratio

    def calculate_width_in_meter(self,FRs,FLs):
        cx = self.param.cx
        
        x1,y1 = self.map_the_point(FLs,cx)
        x2,y2 = self.map_the_point(FRs,cx)
        px = self.rotation_point[0]
        py = self.rotation_point[1]

        # X,Y,_ = query_point_cloud_client(3.2*x, 2.4*y)
        # pX,pY,_ = query_point_cloud_client(3.2*px, 2.4*py)

        z = self.darray[py][px]
        X1,Y1 = self.param.pixel_to_xyz(x1,y1,z)
        X2,Y2 = self.param.pixel_to_xyz(x2,y2,z)
        # # print('pixels',x1,x2,y1,y2,z)
        # # print('meter',X1,X2,Y1,Y2)
        # print(sqrt((x1-x2)**2 + (y1-y2)**2))
        d = sqrt((X1-X2)**2 + (Y1-Y2)**2)
        return d




    def pose_refinement(self):
        bmap = self.bmap
        dmap = self.dmap
        dmap_fwd = []
        dmap_bkd = []
        cy = self.param.cy
        cx = self.param.cx
        FLs = 0
        FLe = 0
        FRs = self.param.gripper_height-1
        FRe = self.param.gripper_height-1
        # Free space left
        for i in range(cy-2,-1,-1): # for looping backward
            # dmap_fwd.append(dmap[cx,i])
            if self.param.gripper_width-bmap[:,i].sum() ==0: #free space
                FLs = i
                break
        for j in range(i-1,-1,-1):
            # dmap_fwd.append(dmap[cx,j])
            if self.param.gripper_width-bmap[:,j].sum() > 0: #collision space
                FLe = j
                break
        # Free space right
        for i in range(cy,self.param.gripper_height): # for looping forward
            # dmap_bkd.append(dmap[cx,i])
            if self.param.gripper_width-bmap[:,i].sum() ==0: #free space
                FRs = i
                break
        for j in range(i+1,self.param.gripper_height):
            # dmap_bkd.append(dmap[cx,j])
            if self.param.gripper_width-bmap[:,j].sum() > 0: #collision space
                FRe = j
                break

        # self.dmap_bkd = dmap_bkd
        # self.dmap_fwd = dmap_fwd
                        
        # print(FLe,FLs,FRs,FRe)
        #check validity
        valid = False
        
        xo,yo = self.map_the_point(cy,cx)
        # meter_to_pixel_ratio = self.calculate_pixel_meter_ratio(FRs,FLs)
        self.object_width = self.calculate_width_in_meter(FLs,FRs) #meter_to_pixel_ratio*(FRs-FLs)
        # print(self.object_width)
        # print('meter_to_pixel_ratio',meter_to_pixel_ratio,'object_width',object_width)
        # gripper_finger_space_max = 0.103*f_x/darray[yo,xo] # Gripper finger space is 0.103 which is in meter, we need space in pixel units
        # print(FLe,FLs,FRs,FRe)
        # self.laplacian = cv2.Laplacian(bmap,cv2.CV_64F)
        # self.laplacian = (self.laplacian/self.laplacian.max())*255
        if (FRe-FRs) > self.param.pixel_finger_width and (FLs-FLe) > self.param.pixel_finger_width and self.object_width < self.param.gripper_finger_space_max: 
            # cone detection code here
            cy_new = int((FLs+FRs)/2)
            
            # sobelx = cv2.Sobel(self.dmap,cv2.CV_64F,1,0,ksize=-1)
            # self.laplacian = cv2.Sobel(sobelx,cv2.CV_64F,1,0,ksize=-1)
            # print('boundary check',self.laplacian[cx,FLe+1:FLs].sum(),self.laplacian[cx+2,FRs:FRe].sum())
            sobely = cv2.Sobel(bmap,cv2.CV_64F,0,1,ksize=-1)
            # left_gradient = self.laplacian[int(cx/2):int(3*cx/2),FLs:cy_new]
            # right_gradient = self.laplacian[int(cx/2):int(3*cx/2),cy_new:FRs]
            # print('gradients left',left_gradient,left_gradient.sum())
            # print('gradients right',right_gradient,right_gradient.sum())

            # left_gradient = sobelx[int(cx/2):int(3*cx/2),FLs:cy_new]
            # right_gradient = sobelx[int(cx/2):int(3*cx/2),cy_new:FRs]
            # print('gradients left',left_gradient,left_gradient.sum())
            # print('gradients right',right_gradient,right_gradient.sum())

            left_gradient = sobely[int(cx/2):int(3*cx/2),FLs:cy_new]
            right_gradient = sobely[int(cx/2):int(3*cx/2),cy_new:FRs]
            # print('gradients left',left_gradient.sum())
            # print('gradients right',right_gradient.sum())
            lg = left_gradient.sum()
            rg = right_gradient.sum()

            if abs(lg) > 50 and abs(rg) > 50 and lg*rg > 0:
                # print('well done bro!!')
                # c = input('dekho!!')
                valid = False
            else:
                valid = True 

            
                                      
        if valid:
            #calculate new pose params
            
            self.new_centroid = np.array([cx,int(cy_new)])

            min_left = self.dmap[:,FLe+2:FLs-1].mean() - np.min([self.dmap[cx,cy],self.dmap[cx,cy_new]])
            min_right = self.dmap[:,FRs+2:FRe-1].mean() - np.min([self.dmap[cx,cy],self.dmap[cx,cy_new]])
            self.min_depth_difference = np.min([min_right,min_left])
            # print('self.min_depth_difference',self.min_depth_difference)
            if self.min_depth_difference < self.param.THRESHOLD2:
                valid = False
                return None
            # if self.boundary_pose:
            #     self.gripper_opening = int(2*min(cy_new-(FLe+3*FLs)/4, (3*FRs+FRe)/4-cy_new)) #(FRe-FLe)/gripper_height #fraction of original opening
            # else:
            self.gripper_opening = int(2*min(cy_new-(FLe+FLs)/2, (FRs+FRe)/2-cy_new))
            free_space_score = self.gripper_opening  - (FRs-FLs)
            self.gripper_opening_meter = self.calculate_width_in_meter(cy_new+int(self.gripper_opening/2),cy_new-int(self.gripper_opening/2))
            # free_space_score = int(500*free_space_score/min(FLs,2*cy-FRs))
            # if free_space_score > 500 :
            #     free_space_score = 500
            # print(self.gripper_opening,self.gripper_opening_meter)
            free_space_score_normalized = 100*(float(free_space_score)/self.param.gdi_max)
            return free_space_score_normalized
        else:
            return None

    def draw_refined_pose(self,image,path=None):
        xmin = 0
        xmax = self.param.gripper_width-1
        ymin = self.new_centroid[1] - int(self.gripper_opening/2)
        ymax = self.new_centroid[1] + int(self.gripper_opening/2)

        point0 = np.array(self.map_the_point(self.new_centroid[1],self.new_centroid[0]))
        point1 = np.array(self.map_the_point(ymax,xmax))
        point2 = np.array(self.map_the_point(ymin,xmax))
        point3 = np.array(self.map_the_point(ymin,xmin))
        point4 = np.array(self.map_the_point(ymax,xmin))
        # print(point1)
        cv2.line(image, (point1[0], point1[1]), (point2[0], point2[1]),
             color=[0,0,0], thickness=2)
        cv2.line(image, (point2[0], point2[1]), (point3[0], point3[1]),
                 color=[0,0,0], thickness=2)
        cv2.line(image, (point3[0], point3[1]), (point4[0], point4[1]),
                 color=[0,0,0], thickness=2)
        cv2.line(image, (point4[0], point4[1]), (point1[0], point1[1]),
                 color=[0,0,0], thickness=2)
        cv2.circle(image, (point0[0], point0[1]), 3,[0,0,0], -1)
        if path is not None:
            cv2.imwrite(path, image)
        return point0,self.gripper_opening_meter, self.object_width

    def point_is_within_image(self,xo,yo):
        w = self.param.w
        h = self.param.h
        if xo<0 or xo > w-1 or yo<0 or yo > h-1:
            return False
        else:
            return True

    def denoise_bmap(self,bmap, dmap):
        # kernel = np.ones((3,3),np.uint8)
        # close = cv2.morphologyEx(bmap, cv2.MORPH_CLOSE, kernel)
        close = cv2.medianBlur(bmap,5)
        bmap_denoised = close.astype(np.uint8)
        bmap_diff = bmap_denoised - bmap
        # print('bmap_diff', np.where(bmap_diff == 1))
        dmap[np.where(bmap_diff == 1)] = self.param.datum_z
        return bmap_denoised, dmap
        # cv2.imwrite('denoised_' + fn+'.jpg',plt_denoised)

    def bound_the_point(self,xc,yc):
        w = self.param.w
        h = self.param.h
        if xc<0:
            xc = 0
        if xc > w-1:
            xc = w-1
        if yc<0:
            yc = 0
        if yc > h-1:
            yc = h-1
        return xc,yc

def calculate_GDI2(rectangle,darray,angle,param):
    gdi2 = GDI2(rectangle[0],angle,darray,param)
    dmap = np.zeros((param.gripper_width,param.gripper_height))
    binary_map = np.zeros((param.gripper_width,param.gripper_height),np.uint8)
    compr_depth = darray[rectangle[0][1],rectangle[0][0]]

    #later this can be replaced with paralalization (vectarization)

    boundary_pose_distance = param.gripper_height
    
    gw = param.gripper_width
    gh = param.gripper_height
    cx = param.cx
    cy = param.cy

    for j in range(gw):
        for i in range(gh):
            xo,yo = gdi2.map_the_point(i,j)
            if gdi2.point_is_within_image(xo,yo):
                depth_value = darray[yo,xo]
            else:
                # centroid_distance = np.sqrt((xo-rectangle[0][0])**2 + (yo-rectangle[0][1])**2)
                # if centroid_distance < boundary_pose_distance:
                #     boundary_pose_distance = centroid_distance
                depth_value = 0.0
            dmap[j,i] = depth_value

    diff_map = dmap-compr_depth

    # dh_map is differentiation of diff_map
    # dh_map is for detection of continuous slanted surface
    dh_map = np.ones(diff_map.shape,np.bool)

    dh1 = diff_map[:,cy:] - diff_map[:,cy-1:gh-1]
    dh2 = diff_map[:,:cy-1] - diff_map[:,1:cy]
    dh_map[:,cy-1] = dh_map[:,cy] 
    dh1 = dh1 > 0.01 
    dh2 = dh2 > 0.01
    for i in range(1,dh1.shape[1]):
        dh1[:,i] = dh1[:,i] | dh1[:,i-1]
    for i in range(cy-3,-1,-1):
        dh2[:,i] = dh2[:,i] | dh2[:,i+1]
    dh_map[:,cy:] = dh1
    dh_map[:,:cy-1] = dh2
    dh_map[:,cy-1] = dh_map[:,cy]

    binary_map = (diff_map > param.THRESHOLD2) & dh_map
    binary_map = binary_map.astype(np.uint8)
            # depth_difference = (depth_value - compr_depth)
            # contact =  depth_difference < param.THRESHOLD2
            # binary_map[j,i] = 1-int(contact)

            
            

    # for i in range(param.gripper_height):
    #     for j in range(param.gripper_width):
    #         if 1-binary_map[j,i] and i-2 >=0 and i+2 < param.gripper_height and j-2 >=0 and j+2 < param.gripper_width:
    #             print('contact',1-binary_map[j,i],'cendroid_depth',compr_depth)
    #             print(grasp_pose_dmap_aligned[j-2:j+2,i-2:i+2])
    # print('grasp_pose_dmap_aligned',grasp_pose_dmap_aligned)
    gdi2.bmap, gdi2.dmap = gdi2.denoise_bmap(binary_map.copy(),dmap.copy())
    bmap_vis = (binary_map / binary_map.max())*255
    bmap_vis_denoised = (gdi2.bmap / gdi2.bmap.max())*255
    gdi_score = gdi2.pose_refinement()
    # if gdi2.gripper_opening > 2*boundary_pose_distance:
    #     gdi2.gripper_opening_meter = gripper_finger_space_max
    gdi_plus_score = gdi2.calculate_gdi_plus_score_better_way()
    # gdi_plus_score_new_way = gdi2.calculate_gdi_plus_score_new_way()
    # print('gdi_plus',gdi_plus_score)
    # if gdi_score:
        # gdi_plus_score = gdi2.calculate_gdi_plus_score_new_way()
    # else:
    #     gdi_plus_score = gdi2.calculate_gdi_plus_score()
    # if gdi_score is not None and gdi_plus_score is None:
        # gdi2.calculate_gdi_score_old_way() # In case no valid grasp pose found
        
    # print(gdi2.pose_refinement(binary_map))
    # print('GDI2',grasp_pose_dmap_aligned)
    # gpd_map = (grasp_pose_dmap_aligned / grasp_pose_dmap_aligned.max())*255
    # gpd_map_denoised_vis = (gdi2.dmap / gdi2.dmap.max())*255
    i = gdi2.new_centroid[0]
    j = gdi2.new_centroid[1]
    xc,yc = gdi2.map_the_point(j,i)
    xc,yc = gdi2.bound_the_point(xc,yc)

    return bmap_vis,gdi_score,gdi_plus_score,gdi2, bmap_vis_denoised,xc,yc