import numpy as np
from math import *
import cv2

class GDI2:
    def __init__(self,rotation_point,angle,darray,param):
        self.rotation_point = rotation_point
        x = rotation_point[0]
        y = rotation_point[1]
        t = angle
        self.rotation_matrix = np.array([[cos(t), -sin(t), -x*cos(t)+y*sin(t)+x], [sin(t), cos(t), -x*sin(t)-y*cos(t)+y], [0,0,1]])
        self.tx = x - int(param.gripper_height/2)
        self.ty = y - int(param.gripper_width/2)
        self.bmap_vis_denoised = None
        self.bmap_ws = None
        self.bmap = None
        self.dmap = None
        self.smap = None
        self.pmap = None
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
        self.FLS_score = None # aka gdi_score
        self.CRS_score = None # aka gdi_plus_score
        self.final_center = None
        self.surface_normal_score = None
        self.cone_detection = False
        self.invalid_reason = 'NA'
        self.invalid_id = 0
        self.final_image = None
        self.lg = None
        self.rg = None
        self.target = None
        self.ADD_multiplier = 1.0

    def rotate(self,point):
        point_homo = np.ones(3)
        point_homo[0:2] = point
        new_point = np.matmul(self.rotation_matrix,point_homo)
        return int(new_point[0]), int(new_point[1])


    def map_the_point_vectorize(self,I):
        I[0] = I[0] + self.tx
        I[1] = I[1] + self.ty

        # rotation part
        I = np.concatenate((I,np.ones((1,I.shape[1])))) # homofie 2xN -> 3xN
        O = np.matmul(self.rotation_matrix,I).astype(np.int32)[0:2,:]
        return O

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
        bmap = self.dmap
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
            self.CRS_score = gdi_plus_count_normalized
            return gdi_plus_count_normalized


    # def calculate_gdi_plus_score_new_way(self):
    #     cy = self.new_centroid[1]
    #     s = cy - int(self.gripper_opening/2) + 1
    #     e = cy + int(self.gripper_opening/2) 
    #     # print(cy,s,e,self.gripper_opening)
    #     total_score = 0
    #     completeness_count = 04 (FLs-FLe) 20 self.object_width 0.1300949046355436

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

        z = self.param.datum_z
        X1,Y1 = self.param.pixel_to_xyz(x1,y1,z)
        X2,Y2 = self.param.pixel_to_xyz(x2,y2,z)
        # # print('pixels',x1,x2,y1,y2,z)
        # # print('meter',X1,X2,Y1,Y2)
        # print(sqrt((x1-x2)**2 + (y1-y2)**2))
        d = sqrt((X1-X2)**2 + (Y1-Y2)**2)
        return d




    def pose_refinement(self,param,DBCC_enable=False):  # DBCC stands for depth-based collision checking
        dmap = self.dmap
        smap = self.smap


        gw = param.gripper_width
        gh = param.gripper_height

        cy = self.param.cy
        cx = self.param.cx
        # print(cx,  cy)
        # print(np.shape(dmap))
        
        
        target = smap[cx,cy]
        if param.target != target:
            # print(param.tar)
            print('******* occluded object**********',param.target, target)
            return None
        # else:
        #     print('******* nice object**********',param.target, target)
        self.target = target
         
        # gdi2.bmap_ws = (diff_map > param.THRESHOLD2)

        rect_region = np.ones(np.shape(smap))

        contact_region_mask = (smap == target)

        compr_depth = dmap[cx,cy]
        # compr_depth = np.where(contact_region_mask,dmap,0.0).sum()/contact_region_mask.sum()
        # print(dmap[cx,cy],compr_depth)

        a = smap != target
        b = smap != 0
        # c = rect_region
        if DBCC_enable:
            print('DBCC_enable')
            diff_map = dmap-compr_depth
            c = diff_map < param.THRESHOLD2
            collision_region_mask = (np.logical_and(np.logical_and(a, b),c))
            # collision_region_mask = (np.logical_or(contact_region_mask,c))
            
        else:
            collision_region_mask = np.logical_and(a, b)
            self.ADD_multiplier = 1.0

        free_region_mask = (np.logical_and(rect_region != contact_region_mask , rect_region != collision_region_mask))
        

        # ADD = 0.0
        # AD_contact = 0.0
        # if DBCC_enable:
        #     dmap_free = np.where(free_region_mask,dmap,0.0) 
        #     AD_free = abs(dmap_free.sum()/free_region_mask.sum())

        #     dmap_contact = np.where(contact_region_mask,dmap,0.0) 
        #     AD_contact = abs(dmap_contact.sum()/contact_region_mask.sum())

        #     ADD = AD_free - AD_contact


             


        b = np.ones((gw ,cy))
        c = np.zeros((gw ,gh-cy))
        d = np.concatenate((b,c),axis=1)
        free_region_mask_left = np.multiply(free_region_mask,d)

        b = np.zeros((gw ,cy))
        c = np.ones((gw ,gh-cy))
        d = np.concatenate((b,c),axis=1)
        free_region_mask_right = np.multiply(free_region_mask,d)

        for row in range(gw):
            for col in range(cy,-1):
                if free_region_mask_left[row][col] == 1 and free_region_mask_left[row][col-1] == 0:
                    c = np.ones(gh-col)
                    b = np.zeros(col)
                    d = np.concatenate((b,c),axis=0)
                    free_region_mask_left[row] = np.multiply(free_region_mask_left[row],d)

        for row in range(gw):
            for col in range(cy,gh-1):
                    if free_region_mask_right[row][col] == 1 and free_region_mask_right[row][col+1] == 0:
                        c = np.zeros(gh-col)
                        b = np.ones(col)
                        d = np.concatenate((b,c),axis=0)
                        free_region_mask_right[row] = np.multiply(free_region_mask_right[row],d)
        #modify free region mask right and left so that all free regions on the extremes are nullified
        

        


        
        valid = False
        # decide if pose is valid or invalid
        # check if contact region size is smaller then a threshold (gripper max opening)
        # check if free space size is greater then a threshold (the gripper finger)
        # ******** code here *************
        max_obj_width = contact_region_mask.sum(axis=1).max(axis=0)
        self.object_width = self.calculate_width_in_meter(0,max_obj_width)

        min_fsl_width = free_region_mask_left.sum(axis=1).min(axis=0)
        min_fsr_width = free_region_mask_right.sum(axis=1).min(axis=0)

        if min_fsl_width > self.param.pixel_finger_width and min_fsr_width > self.param.pixel_finger_width and self.object_width < self.param.gripper_finger_space_max: 
            valid = True

        else:
            # print(self.param.pixel_finger_width,self.param.gripper_finger_space_max,'(FRe-FRs)',(FRe-FRs),'(FLs-FLe)',(FLs-FLe),'self.object_width',self.object_width)
            self.invalid_reason = 'large object or less free space'
            self.invalid_id = 1

        if valid:
            # calculate new center based on the contact region mask
            # ******** code here *************
            # store in self.new_centroid
            _,cy_new = self.calc_centre(contact_region_mask)
            self.new_centroid = np.array([cx,int(cy_new)])

            # calculate new gripper opening based on the left and right parts of free_region_mask
            # ******** code here *************
            # store in self.gripper_opening
            _,c_fsl = self.calc_centre(free_region_mask_left)
            _,c_fsr = self.calc_centre(free_region_mask_right)
            min_gripper_opening = int(c_fsr - c_fsl)
            self.gripper_opening = min_gripper_opening


            # Calculate score1 based on the number of pixels in the free region (min of left and right parts of free_region_mask)
            # ******** code here *************
            # store in self.score1
            free_space_score = min(free_region_mask_left.sum(),free_region_mask_right.sum())
            normalize_param = 1/2*gw*gh
            free_space_score_normalized = 100*(float(free_space_score)/normalize_param)

        if valid:
            self.FLS_score = free_space_score_normalized

            # code here for CRS score
            #******************************
            crs_score = contact_region_mask[:,cy-int(gw/2):cy+int(gw/2)].sum()
            self.CRS_score = crs_score/(gw*gw)*100
            # print('crs_score',self.CRS_score,crs_score,gw,gh,cx,contact_region_mask.shape)
            # if DBCC_enable:
                # print('ADD:',ADD)
                # print(np.where(free_region_mask,diff_map,0.0))
                # print(free_region_mask)
                # print(collision_region_mask)
                # print(c)
                # print(dmap[cx-10:cx+10,cy-10:cy+10])
            return self.FLS_score

        else: 
            return None

        # # calculate left and right parts of free_region_mask
        # FLs = 0
        # FLe = 0
        # FRs = self.param.gripper_height-1
        # FRe = self.param.gripper_height-1

        # #free space left
        # for i in range(cy,-1,-1): # for looping backward
        #     if self.param.gripper_width - free_region_mask[:,i].sum() == 0: #free space
        #         FLs = i
        #         break

        # #Free space right
        # for i in range(cy,self.param.gripper_height): # for looping forward
        #     if self.param.gripper_width - free_region_mask[:,i].sum() == 0: #free space
        #         FRs = i
        #         break
        
        # for j in range(FLs,-1,-1):
        #     if collision_region_mask[:,j].sum() == 0: #collision space
        #         FLe = j
        #         continue

        # for j in range(FRs,self.param.gripper_height):
        #     if collision_region_mask[:,j].sum() == 0: #collision space
        #         FRe = j
        #         continue

        


        # valid = False
        # # decide if pose is valid or invalid
        # # check if contact region size is smaller then a threshold (gripper max opening)
        # # check if free space size is greater then a threshold (the gripper finger)
        # # ******** code here *************
        # self.object_width = self.calculate_width_in_meter(FLs,FRs)

        # if (FRe-FRs) > self.param.pixel_finger_width and (FLs-FLe) > self.param.pixel_finger_width and self.object_width < self.param.gripper_finger_space_max: 
        #     valid = True

        # else:
        #     # print(self.param.pixel_finger_width,self.param.gripper_finger_space_max,'(FRe-FRs)',(FRe-FRs),'(FLs-FLe)',(FLs-FLe),'self.object_width',self.object_width)
        #     self.invalid_reason = 'large object or less free space'
        #     self.invalid_id = 1

        # cy_new = int((FLs + FRs)/2)




        # if valid:
        #     # calculate new center based on the contact region mask
        #     # ******** code here *************
        #     # store in self.new_centroid
        #     cy_new = int((FLs + FRs)/2)
        #     self.new_centroid = np.array([cx,int(cy_new)])

        #     # calculate new gripper opening based on the left and right parts of free_region_mask
        #     # ******** code here *************
        #     # store in self.gripper_opening
        #     min_gripper_opening = int(2*min(cy_new-(FLe+FLs)/2, (FRs+FRe)/2-cy_new))
        #     self.gripper_opening = min_gripper_opening


        #     # Calculate score1 based on the number of pixels in the free region (min of left and right parts of free_region_mask)
        #     # ******** code here *************
        #     # store in self.score1
        #     free_space_score = min_gripper_opening  - (FRs-FLs)
        #     free_space_score_normalized = 100*(float(free_space_score)/self.param.gdi_max)

        # if valid:
        #     self.FLS_score = free_space_score_normalized
        #     return free_space_score_normalized
        # else: 
        #     return None

    def calc_centre(self,array: np.ndarray):
        total = array.sum()
        # alternatively with np.arange as well
        x_coord = round((array.sum(axis=1) @ range(array.shape[0])) / total)
        y_coord = round((array.sum(axis=0) @ range(array.shape[1])) / total)
        return x_coord, y_coord

    def draw_refined_pose(self,image,path=None,scale=1, thickness = 2):
        xmin = 0
        xmax = self.param.gripper_width-1
        ymin = self.new_centroid[1] - int(self.gripper_opening/2)
        ymax = self.new_centroid[1] + int(self.gripper_opening/2)

        point0 = scale*np.array(self.map_the_point(self.new_centroid[1],self.new_centroid[0]))
        point1 = scale*np.array(self.map_the_point(ymax,xmax))
        point2 = scale*np.array(self.map_the_point(ymin,xmax))
        point3 = scale*np.array(self.map_the_point(ymin,xmin))
        point4 = scale*np.array(self.map_the_point(ymax,xmin))
        # refined_pose = np.concatenate(point)
        # print(point1)
        # color1 = (255,255,0) # cyan
        # # color = (0,160,255) # orange
        # color = (178, 83, 70) # Liberty
        color = (17,233,135) # green
        color = (69,24,255)
        color1 = (25,202,242)
        # color1 = (255,80,0) 
        
        cv2.line(image, (point1[0], point1[1]), (point2[0], point2[1]),
             color=color1, thickness=thickness)
        cv2.line(image, (point2[0], point2[1]), (point3[0], point3[1]),
                 color=color, thickness=thickness)
        cv2.line(image, (point3[0], point3[1]), (point4[0], point4[1]),
                 color=color1, thickness=thickness)
        cv2.line(image, (point4[0], point4[1]), (point1[0], point1[1]),
                 color=color, thickness=thickness)
        cv2.circle(image, (point0[0], point0[1]), thickness,color, -1)
        if path is not None:
            cv2.imwrite(path, image)
        self.final_center = point0/scale
        return self.final_center,self.gripper_opening_meter, self.object_width

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

def calculate_GDI2_Lite(inputs,rectangle,angle,vectorization=True):
    import time
    st = time.time()
    darray = inputs['darray']
    param = inputs['param']
    seg_mask = inputs['seg_mask']

    gdi2 = GDI2(rectangle[0],angle,darray,param)
    dmap = np.zeros((param.gripper_width,param.gripper_height))
    smap = np.zeros((param.gripper_width,param.gripper_height))
    binary_map = np.zeros((param.gripper_width,param.gripper_height),np.uint8)
    # print(rectangle[0])
    # compr_depth = darray[rectangle[0][1],rectangle[0][0]]

    #later this can be replaced with paralalization (vectarization)

    boundary_pose_distance = param.gripper_height

    gw = param.gripper_width
    gh = param.gripper_height

    
    
    # st = time.time()
    # vectorization
    Imap = np.mgrid[0:gh:1,0:gw:1].reshape(2,-1)
    Omap = gdi2.map_the_point_vectorize(Imap) # 2xN
    
    # filter for points within image boundaries
    w = param.w
    h = param.h

    within_points_filter = (Omap[0,:] < 0) + (Omap[0,:] > w-1) + (Omap[1,:] < 0) + (Omap[1,:] > h-1)
    Omap[0] = np.where(within_points_filter,0.0,Omap[0])
    Omap[1] = np.where(within_points_filter,0.0,Omap[1])
    dmap = np.where(within_points_filter,0.0,darray[Omap[1],Omap[0]])
    dmap = dmap.reshape(gh,gw).T

    smap = np.where(within_points_filter,-1.0,seg_mask[Omap[1],Omap[0]])
    smap = smap.reshape(gh,gw).T
    # print('time in dmap1',time.time()-st)

    gdi2.dmap = dmap
    gdi2.smap = smap

    # diff_map = dmap-compr_depth 
    # gdi2.bmap_ws = (diff_map > param.THRESHOLD2)
    # binary_map = gdi2.bmap_ws
    # binary_map = binary_map.astype(np.uint8)

    # # visualize
    # ri = np.random.randint(1,1000)
    # v0 = 255*dmap/dmap.max()
    # v1 = 255*dmap1/dmap1.max()
    # cv2.imwrite('temp/{0}_dmap.png'.format(ri),v0)
    # cv2.imwrite('temp/{0}_dmap1.png'.format(ri),v1)

    
    
    # gdi2.bmap, gdi2.dmap = gdi2.denoise_bmap(binary_map.copy(),dmap.copy())

    # if param.DBCC_enable:
    #     gdi_score = gdi2.pose_refinement(param,DBCC_enable=True)
    # else:
    #     gdi_score = gdi2.pose_refinement(param)
    if inputs['run_with_dbcc']:
        gdi_score = gdi2.pose_refinement(param,DBCC_enable=True)
    else:
        gdi_score = gdi2.pose_refinement(param)
        
    # if gdi_score is None and param.DBCC_enable:
    #     # print("******** The current evaluated pose is detected as invalid. Going with depth-based collison checking **********")
    #     gdi_score = gdi2.pose_refinement(param,DBCC_enable=True)
    #     if gdi_score is not None:
    #         gdi_score = (3/4)*gdi_score #penelty to give higher priority to without DBCC_enabled

    gdi_plus_score = gdi2.CRS_score
    

    return None,gdi_score,gdi_plus_score,gdi2, None,None,None