import os
import numpy as np
import torch
from PIL import Image
from scipy import stats as st
# import cv2

class BinDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, type, label_issue=False, real_noise=False, train=True,eval_data_type = 'ours-sim',has_gcs_branch=False):
        self.root = root
        self.transforms = transforms
        self.has_gcs_branch = has_gcs_branch
        # load all image files, sorting them to
        # ensure that they are aligned
        if not real_noise:
            self.dps = list(sorted(os.listdir(os.path.join(root, "depth_ims"))))
        else:
            self.dps = list(sorted(os.listdir(os.path.join(root, "depth_ims_noised"))))
        self.imgs = list(sorted(os.listdir(os.path.join(root, "color_ims"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "modal_segmasks"))))
        if train:
            self.dmaps = list(sorted(os.listdir(os.path.join(root, "depth_maps"))))
        if has_gcs_branch:
            self.gcs_files = list(sorted(os.listdir(os.path.join(root, "grasp_confidense_scores"))))
        self.type = type
        self.label_issue = label_issue
        self.train = train
        self.eval_data_type = eval_data_type
        # if not train: 
        #     print(self.imgs)
        self.max_depth = 0.0
        self.min_depth = 1000.0
        self.label_freq_count = np.zeros(14)
    def __getitem__(self, idx):
        # if self.has_gcs_branch:
        #     print(self.dps[idx],self.imgs[idx],self.masks[idx],self.dmaps[idx],self.gcs_files[idx])
        if self.type == 'LDD':
            # load images and masks
            img_path = os.path.join(self.root, "color_ims", self.imgs[idx])
            L = Image.open(img_path).convert("L")
            L = np.array(L)

            depth_path = os.path.join(self.root, "depth_ims", self.dps[idx])
            img = Image.open(depth_path).convert("RGB")
            img = np.array(img)
            img[:,:,0] = L
            img = Image.fromarray(img.astype(np.uint8))

        elif self.type == 'LLL':
            img_path = os.path.join(self.root, "color_ims", self.imgs[idx])
            L = Image.open(img_path).convert("L")
            L = np.array(L)

            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            img[:,:,0] = L
            img[:,:,1] = L
            img[:,:,2] = L
            img = Image.fromarray(img.astype(np.uint8))

        elif self.type == 'RGD':
            # load images and masks
            img_path = os.path.join(self.root, "color_ims", self.imgs[idx])
            RGB = Image.open(img_path).convert("RGB")
            RGB = np.array(RGB)

            depth_path = os.path.join(self.root, "depth_ims", self.dps[idx])
            img = Image.open(depth_path).convert("RGB")
            img = np.array(img)
            img[:,:,0] = RGB[:,:,0]
            img[:,:,1] = RGB[:,:,1]
            img = Image.fromarray(img.astype(np.uint8))

        elif self.type == 'RGB':
            # load images and masks
            img_path = os.path.join(self.root, "color_ims", self.imgs[idx])
            img = Image.open(img_path).convert("RGB")

        elif self.type == 'DDD':
            # load images and masks
            if self.train:
                depth_path = os.path.join(self.root, "noisy_depth", self.dps[idx])
            else:
                depth_path = os.path.join(self.root, "depth_ims", self.dps[idx])
            img = Image.open(depth_path).convert("RGB")

        elif self.type == 'final':
            img_path = os.path.join(self.root, "color_ims", self.imgs[idx])
            img = Image.open(img_path).convert("RGB")

        if self.train:
            depth_path = os.path.join(self.root, "depth_maps", self.dmaps[idx])
            depth_map = np.load(depth_path).astype(np.float32)
            aux_target = torch.as_tensor(depth_map)
            focal = np.array([614.72,614.72])

        else:
            depth_path = os.path.join(self.root, "depth_ims", self.dps[idx])
            depth = Image.open(depth_path)
            depth_map = np.array(depth).astype(float)
            
            depth_map = depth_map/255.0
            aux_target = torch.as_tensor(depth_map)
            if self.eval_data_type == 'wisdom':
                focal = np.array([1105.0,1105.0]) # wisdom-real data
            elif self.eval_data_type == 'ours':
                focal = np.array([614.72,614.72])
            elif self.eval_data_type == 'ours-sim':
                focal = np.array([614.72,614.72])
        focal = torch.as_tensor(focal)

        if self.has_gcs_branch:
            gcs_path = os.path.join(self.root, "grasp_confidense_scores", self.gcs_files[idx])
            gcs_scores = np.loadtxt(gcs_path)/100  # to bring the scores between 0 and 1
        else:
            gcs_scores = None
        # max_depth = depth_map.max()
        # min_depth = np.where(depth_map < 80,255,depth_map).min()

        # if self.max_depth < max_depth:
        #     self.max_depth = max_depth
        # if self.min_depth > min_depth and min_depth > 80:
        #     self.min_depth = min_depth

        # print('max depth',self.max_depth)
        # print('min depth',self.min_depth)


        # # print(idx,self.masks[idx])
        # darray = np.load(os.path.join(self.root, "depth_maps")+'/dmap_{0:06d}.npy'.format(idx))
        # darray = darray - darray.min()
        # darray = darray/darray.max()
        # print(darray.shape)
        # img = np.tile(darray[:,:,np.newaxis],(1,1,3))
        

        mask_path = os.path.join(self.root, "modal_segmasks", self.masks[idx])
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        if self.label_issue:
            # print('***************** label issue')
            mask = np.where(mask <4 , 0, mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # print(obj_ids)
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        masks_final = []
        count = 0
        labels_new = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            # seg_depth = depth_map[pos]
            # print(idx,i,seg_depth)
            # print(idx,i,'mode',st.mode(seg_depth))
            # print(idx,i,'mean',seg_depth.mean())
            # print(idx,i,'max',seg_depth.max())
            # print(idx,i,'min',seg_depth.min())
            if self.train:
                label = 1
                # mode = 100*st.mode(seg_depth,keepdims=False)[0]
                # if mode < 57:
                #     label = 1
                # elif mode < 65:
                #     label = 2
                # else:
                #     label = 3 #int((mode-56)/2)
            else:
                label = 1
            # self.label_freq_count[label-1] += 1
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if not ((xmin==xmax) | (ymin==ymax)):
                boxes.append([xmin, ymin, xmax, ymax])
                masks_final.append(masks[i])
                count += 1
                labels_new.append(label)
        # print(labels_new)
        # print(self.label_freq_count)
        # print(boxes)
        # convert everything into a torch.Tensor
        boxes = np.array(boxes)
        masks_final = np.array(masks_final)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # if self.train:
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.tensor(np.array(labels_new), dtype=torch.int64)
        # else:
        #     labels = torch.ones((count,), dtype=torch.int64)

        #*********** for distance based categories ***********************


        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        masks = torch.as_tensor(masks_final, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # if not self.train:
        #     print(idx,boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # print('area',area)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((count,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # target["gcs_scores"] = gcs_scores
        # target["depth_map"] = depth_map

        if self.has_gcs_branch:
            target["gcs_scores"] = torch.tensor(gcs_scores, dtype=torch.float32) 

        if self.transforms is not None:
            img, target, aux_target = self.transforms(img, target, aux_target)



        return img, target, aux_target, focal
        # if self.train:
        #     return img, target, aux_target, focal
        # else:
        #     return img, target

    def __len__(self):
        return len(self.masks)