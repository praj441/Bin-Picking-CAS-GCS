import pybullet as p
import numpy as np
# import pybullet_data
import time
# import random
from PIL import Image
from math import*
import random

hfov = 55.0
vfov = 42.69
aspect = 1.334
near = 0.01
far = 10
# width = 320
# height = 240
pi=3.14159265359
# ort=p.getQuaternionFromEuler([0.0,1.57,0]) #(0.0734, 0.728, -0.067, 0.677)
default_orn = p.getQuaternionFromEuler([0,0,0])

def cam_to_ee_calibration(pos,orn):
    # print('cam',x,y,z,rx,ry,rz)
    eu = p.getEulerFromQuaternion(orn)
    #camera to ee calibration
    xe = pos[2] 
    ye = -pos[0]
    ze = -pos[1]
    rxe = eu[2]
    rye = -eu[0]
    rze = -eu[1]
    p2 = [xe,ye,ze]
    q2 = p.getQuaternionFromEuler([rxe,rye,rze])
    return p2,q2

def cam_pose_to_world_pose(camTargetPos,camTargetOrn=default_orn):
    epos,eorn = cam_to_ee_calibration(camTargetPos,camTargetOrn)
    return p.multiplyTransforms(pos,ort,epos,eorn)

def get_image(width=320,height=240):
    prob = 0.5
    if np.random.random() > prob:
        pos=(0.15+0.1*np.random.uniform(-1,1), 0.0+0.1*np.random.uniform(-1,1), 1.34+0.1*np.random.uniform(-1,1)) # 1.34
    else:
        pos=(0.15,0.0,1.34)
    if np.random.random() > prob:
        random_ort_euler = [0.0175*np.random.uniform(-3,3),1.57+0.0175*np.random.uniform(-3,3),0.0175*np.random.uniform(-3,3)]
    else:
        random_ort_euler = [0.0,1.57,0.0]
    ort=p.getQuaternionFromEuler(random_ort_euler)


    rot_mat=np.array(p.getMatrixFromQuaternion(ort))
    rot_mat=np.reshape(rot_mat,[3,3])
    dir=rot_mat.dot([1,0,0])
    up_vector=rot_mat.dot([0,0,1])
    s = 0.01
    view_matrix=p.computeViewMatrix(pos,pos+s*dir,up_vector)
    # p.addUserDebugText(text=".",textPosition=pos+s*dir,textColorRGB=[1,0,0],textSize=10)
    projection_matrix = p.computeProjectionMatrixFOV(vfov,aspect,near,far)
    # f_len = projection_matrix[0]
    ld = [random.uniform(-20,20),random.uniform(-20,20),random.uniform(10,20)]
    lc = [random.random(),random.random(),random.random()]
    lightDirection = [random.uniform(-0.4,0.6),random.uniform(-0.5,0.5),pos[2]]
    if random.random() < 0.0:
        _,_,rgbImg,depthImg_buffer,segImg=p.getCameraImage(width,height,view_matrix,projection_matrix,lightDirection=lightDirection,renderer=p.ER_BULLET_HARDWARE_OPENGL)
    else:
        _,_,rgbImg,depthImg_buffer,segImg=p.getCameraImage(width,height,view_matrix,projection_matrix,renderer=p.ER_TINY_RENDERER,lightDirection=lightDirection,shadow=1,
        lightColor=lc,lightAmbientCoeff=random.random(),lightDiffuseCoeff=random.random(),lightSpecularCoeff=random.random())
    A = np.reshape(rgbImg, (height,width, 4))[:, :, :3]
    depthImg_buf = np.reshape(depthImg_buffer, [height,width])
    # print(depthImg_buffer[50,50])
    depthImg = far * near / (far - (far - near) * depthImg_buf)
    # print(depthImg[50,50])
    I = np.float64(Image.fromarray(A))
    temp = I[:,:,0].copy()
    I[:,:,0] = I[:,:,2]
    I[:,:,2] = temp
    return I,depthImg,segImg

# def get_point_cloud_from_