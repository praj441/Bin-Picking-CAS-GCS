import pybullet as p
import random
import numpy as np
from math import *
import cv2
import os
from glob import glob
# import open3d as o3d
# from rotation_conversion import rotationMatrixToEulerAngles,eulerAnglesToRotationMatrix


        
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
wild_texture_list = []
# open file and read the content in a list
# with open('textures/wild/file_list.txt', 'r') as filehandle:
#     for line in filehandle:
#         # remove linebreak which is the last character of the string
#         currentName = line[:-1]
#         # add item to the list
#         wild_texture_list.append(currentName)

texture_path = '../../data/textures/wild'
wild_texture_list = [y for x in os.walk(texture_path) for y in glob(os.path.join(x[0], '*.jpg'))]

table_n_floor_texture_path = '../../data/textures/table_n_floor'
table_n_floor_texture_list = os.listdir(table_n_floor_texture_path)


def get_segmented_pcd(pc_arr,mask,maskIds):
    m,n = mask.shape
    pc_arr = pc_arr.reshape((m,n,-1))
    sp_arr = {}
    for id in maskIds:
        sp_arr['{0}'.format(id)] = []
    for i in range(m):
        for j in range(n):
            for id in maskIds:
                if mask[i,j] == id:
                    sp_arr['{0}'.format(id)].append(pc_arr[i,j,:])
    xyz_list = []
    for id in maskIds:
        xyz = np.array(sp_arr['{0}'.format(id)])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # o3d.io.write_point_cloud("obj_{0}.pcd".format(id), pcd)
        xyz_list.append(xyz)
    return xyz_list

def get_3d_bboxes(pc_arr,mask,maskIds):
    seg_points = get_segmented_pcd(pc_arr,mask,tableObjectUids) #returns list of segmented points arrays
        
    num_bbox = len(tableObjectUids)
    bboxes = np.zeros((num_bbox,8))
    for i in range(num_bbox):
        angle,major,minor = major_minor_axis(seg_points[i][:,0:2]) #pass only x and y coordinates
        z_min = np.min(seg_points[i][:,2])
        z_max = 0.775 # depth of the table surface from camera

        cx = np.mean(seg_points[i][:,0])
        cy = np.mean(seg_points[i][:,1])
        cz = (z_max+z_min)/2
        l = major/2
        w = minor/2
        h = (z_max-z_min)/2
        # angle = -angle
        bboxes[i] = np.array([cx,cy,cz,l,w,h,angle,labelIds[i]])
    return bboxes

def generate_edge_flag_map(pc_arr,mask,maskIds):
    bboxes = get_3d_bboxes(pc_arr,mask,maskIds)
    flag_map = np.zeros(mask.shape)
    h,w = mask.shape

    for i in range(h):
        for j in range(w):
            obj_id = mask[i,j]
            bbox


w = 200
h = 200
fov = 50
pi = 3.14
f_x = w/(2*tan(fov*pi/360))
f_y = h/(2*tan(fov*pi/360))

def pixel_to_xyz(px,py,z):
    #cartesian coordinates
    x = (px - (w/2))*(z/(f_x))
    y = (py - (h/2))*(z/(f_y))
    return x,y,z





def draw_grasp(img,centroid, angle,grasp_width_px,color=(0, 0, 255)):
  gripper_width = 12
  gripper_height = grasp_width_px # different naming conventions
  [x1, y1] = [int(centroid[0] - gripper_height * 0.5 * cos(angle) - gripper_width * 0.5 * cos(angle + radians(90))),
              int(centroid[1] - gripper_height * 0.5 * sin(angle) - gripper_width * 0.5 * sin(angle + radians(90)))]
  [x2, y2] = [int(centroid[0] - gripper_height * 0.5 * cos(angle) + gripper_width * 0.5 * cos(angle + radians(90))),
              int(centroid[1] - gripper_height * 0.5 * sin(angle) + gripper_width * 0.5 * sin(angle + radians(90)))]
  [x3, y3] = [int(centroid[0] + gripper_height * 0.5 * cos(angle) + gripper_width * 0.5 * cos(angle + radians(90))),
              int(centroid[1] + gripper_height * 0.5 * sin(angle) + gripper_width * 0.5 * sin(angle + radians(90)))]
  [x4, y4] = [int(centroid[0] + gripper_height * 0.5 * cos(angle) - gripper_width * 0.5 * cos(angle + radians(90))),
              int(centroid[1] + gripper_height * 0.5 * sin(angle) - gripper_width * 0.5 * sin(angle + radians(90)))]

  cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
  cv2.line(img, (x2, y2), (x3, y3),color , 1)
  cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), 1)
  cv2.line(img, (x4, y4), (x1, y1), color, 1)
  cv2.circle(img, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)
  # print('draw',(x1, y1))
  return img

def axis_angle(points):
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    modi_x = np.array(points[:, 0] - cx)
    modi_y = np.array(points[:, 1] - cy)
    num = np.sum(modi_x * modi_y)
    den = np.sum(modi_x ** 2 - modi_y ** 2)
    angle = 0.5 * atan2(2 * num, den)
    return angle

def major_minor_axis(points):
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

def load_others_urdf(xpos,ypos,obj_idx):
    urdf_path = "../urdf_objects/others/urdf/{0:03d}.urdf".format(obj_idx)
    # print(urdf_path)
    uid = p.loadURDF(urdf_path,[xpos,ypos,0.625],cubeStartOrientation)
    return uid

def load_random_urdf(xpos,ypos,d,obj_idx,globalScaling=1.5,FixedBase=0):
    urdf_path = "../urdfs/random_urdfs/{0:03d}/{0:03d}.urdf".format(obj_idx)
    # print(urdf_path)
    uid = p.loadURDF(urdf_path,[xpos,ypos,d],cubeStartOrientation,useFixedBase=FixedBase,globalScaling=globalScaling)
    return uid

def load_shapenet(xpos,ypos,obj_idx):
    urdf_path = "../urdf_objects/shapenet/urdfs/{0:05d}.urdf".format(obj_idx)
    # print(urdf_path)
    uid = p.loadURDF(urdf_path,[xpos,ypos,0.625],cubeStartOrientation,useFixedBase=1)
    return uid

def load_shapenet_natural_scale(xpos,ypos):
    obj_idx = random.randint(0,5396)
    urdf_path = "../urfd_objects/shapenet/urdfs1/{0:05d}.urdf".format(obj_idx)
    # print(urdf_path)
    uid = p.loadURDF(urdf_path,[xpos,ypos,0.02],cubeStartOrientation,useFixedBase=1)
    return uid

def load_random_table_n_floor_texture(objId):
    if random.random() > 0.5:
        rid = random.randint(0,len(table_n_floor_texture_list)-1)
        tex = table_n_floor_texture_list[rid]
        # print('******************',tex)
        tId = p.loadTexture(table_n_floor_texture_path+'/'+tex)
        p.changeVisualShape(objId,-1,textureUniqueId=tId)

def load_random_object_texture(objId):
    if random.random() > 0.5:
        rid = random.randint(0,len(wild_texture_list)-1)
        tex = wild_texture_list[rid]
        tId = p.loadTexture(tex)
        p.changeVisualShape(objId,-1,textureUniqueId=tId)
#     else:
#         p.changeVisualShape(objId,-1,rgbaColor=[random.random(),random.random(),random.random(),1],specularColor=[random.randint(0,100),random.randint(0,100),random.randint(0,100)])

# def load_random_wild_texture(objId):
#     if random.random() > 0.5:
#         total_files = len(wild_texture_list)
#         random_file_index = random.randint(0,total_files-1)
#         tex = 'textures/wild/' + wild_texture_list[random_file_index] 
#         tId = p.loadTexture(tex)
#         p.changeVisualShape(objId,-1,textureUniqueId=tId)
#     else:
#         index = random.randint(0,18)
#         tex = 'textures/table/table{0:03d}.jpg'.format(index)
#         tId = p.loadTexture(tex)
#         p.changeVisualShape(objId,-1,textureUniqueId=tId)
#         p.changeVisualShape(objId,-1,rgbaColor=[random.random(),random.random(),random.random(),1],specularColor=[random.randint(0,100),random.randint(0,100),random.randint(0,100)])


def init_robot_and_default_camera(roboId,required_joints):
    random_rgba = [random.random(),random.random(),random.random(),1]
    random_spec=[random.randint(0,100),random.randint(0,100),random.randint(0,100)]
    for i in range(1,7):
        p.resetJointState(bodyUniqueId=roboId,
                                jointIndex=i,
                                targetValue=required_joints[i-1])
        p.changeVisualShape(roboId,i,rgbaColor=random_rgba,specularColor=random_spec)

    p.resetDebugVisualizerCamera( cameraDistance=2.2, cameraYaw=140, cameraPitch=-60, cameraTargetPosition=[0,0,0])

def sample_camera_displacement():
    x = random.uniform(-0.1,0.1)
    y = random.uniform(-0.1,0.1)
    z = random.uniform(-0.5,-0.1)
    # print('z',z)
    rx = random.uniform(-0.15,0.15)
    ry = random.uniform(-0.15,0.15)
    rz = random.uniform(-0.5,0.5)
    return x,y,z,rx,ry,rz

def relative_ee_pose_to_ee_world_pose(robotId,xe,ye,ze,rxe,rye,rze):
    ee_link_state = p.getLinkState(robotId,linkIndex=7,computeForwardKinematics=1)
    ee_pos_W=ee_link_state[-2]
    ee_ort_W=ee_link_state[-1]
    eeTargetPos = [xe,ye,ze]
    eeTargetOrn = p.getQuaternionFromEuler([rxe,rye,rze])
    return p.multiplyTransforms(ee_pos_W,ee_ort_W,eeTargetPos,eeTargetOrn)

def relative_ee_pose_to_ee_world_pose1(robotId,eeTargetPos,eeTargetOrn):
    ee_link_state = p.getLinkState(robotId,linkIndex=7,computeForwardKinematics=1)
    ee_pos_W=(0.01, 0.0, 1.3)
    ee_ort_W=(0.0734, 0.728, -0.067, 0.677)
    return p.multiplyTransforms(ee_pos_W,ee_ort_W,eeTargetPos,eeTargetOrn)



def cam_to_ee_calibration(x,y,z,rx,ry,rz):
    # print('cam',x,y,z,rx,ry,rz)
    p1 = [x,y,z]
    q1 = p.getQuaternionFromEuler([rx,ry,rz])
    # print(p1,q1)
 #camera to ee calibration
    xe = z 
    ye = -x
    ze = -y
    rxe = rz
    rye = -rx
    rze = -ry
    # print('ee1',xe,ye,ze,rxe,rye,rze)

    p2 = [xe,ye,ze]
    q2 = p.getQuaternionFromEuler([rxe,rye,rze])

    # p3,q3 = p.invertTransform(p2,q2)
    # p4,q4 = p.multiplyTransforms(p1,q1,p2,q2)
    # print('eTc',p4,p.getEulerFromQuaternion(q4))
    # # # print(p2,q2)
    # p1 = [0,0,0]
    # q1 = p.getQuaternionFromEuler([-1.57,0,-1.57])
    # p3,q3 = p.multiplyTransforms(p1,q1,p2,q2)
    # eu = p.getEulerFromQuaternion(q3)

    # print('ee2',p3,eu)

    return p2,q2

def ee_to_cam_calibration(pe,qe):
    # p1 = [0,0,0]
    # q1 = p.getQuaternionFromEuler([-1.57,0,-1.57])
    # p1,q1 = p.invertTransform(p1,q1)
    # return p.multiplyTransforms(p1,q1,pe,qe)
    eu = p.getEulerFromQuaternion(qe)
    return [-pe[1],-pe[2],pe[0]],p.getQuaternionFromEuler([-eu[1],-eu[2],eu[0]])



# def servo_control_law(x,w):
#     # print(x,w)
#     w = np.array(w)
#     R = eulerAnglesToRotationMatrix(w)
#     t = x
#     l1 = 0.1
#     l2 = 0.1
#     v = -l1*np.matmul(np.transpose(R),t)
#     w =  -l2*w
#     v = np.concatenate((v, w), axis=0)
#     return v

def getMotorJointStates(robot):
  joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
  joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
  joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques

def get_jecobian(roboId):
    mpos, mvel, mtorq = getMotorJointStates(roboId)

    result = p.getLinkState(roboId,
                            7,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)
    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    # Get the Jacobians for the CoM of the end-effector link.
    # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
    # The localPosition is always defined in terms of the link frame coordinates.

    zero_vec = [0.0] * len(mpos)
    jac_t, jac_r = p.calculateJacobian(roboId,7, com_trn, mpos, zero_vec, zero_vec)
    return jac_t, jac_r

def get_joint_velocities(roboId,pos,ort): 
    Jt,Jr = get_jecobian(roboId)
    ort_E = p.getEulerFromQuaternion(ort)
    J1 = np.array(list(Jt))
    J2 = np.array(list(Jr))
    Jinv = np.linalg.inv(np.concatenate((J1,J2), axis=0))
    V = np.concatenate((pos,ort_E),axis=0)
    print('V',V)
    qdot = np.matmul(Jinv,V)
    return qdot

def servo_error(pos1,ort1,pos2,ort2):
    p1 = np.array(pos1)
    r1 = np.array(ort1)
    p2 = np.array(pos2)
    r2 = np.array(ort2)
    Et,Er = np.linalg.norm(p1-p2),np.linalg.norm(r1-r2)
    return Et,Er

def save_training_data(I1,I2,P,Q,index,task):
    pose = np.concatenate((P,Q))
    I1.save('data/train/task{1}/image{0:06d}_0.jpg'.format(index,task))
    I2.save('data/train/task{1}/image{0:06d}_1.jpg'.format(index,task))
    np.savetxt('data/train/task{1}/{0:06d}.txt'.format(index,task),pose,fmt='%0.3f',delimiter=',')


def objects_picked_succesfully(num_obj,tableObjectUids,tId):
    pick_count = 0
    picked_target = False
    for everyuid in tableObjectUids:
        position,orientation = p.getBasePositionAndOrientation(everyuid)
        # print("position ", position[2])
        if position[2]<0.5:
            pick_count = pick_count+1
            if everyuid == tId:
                picked_target = True
    return pick_count,picked_target


# x,y,z,rx,ry,rz=sample_camera_displacement()
# servo_control_law([x,y,z],[rx,ry,rz])