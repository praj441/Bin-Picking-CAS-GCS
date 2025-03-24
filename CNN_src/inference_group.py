import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine_final import train_one_epoch, evaluate
import utils
import transforms_final as T
# from torchvision.transforms import transforms as T
from dataset import BinDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os 
from tqdm import tqdm
from model import custom_model
import time

mean = (128,128,128)
std = (64,64,64)

def get_transform(data_type, depth):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.PILToTensor())
    transforms.append(T.Normalize(mean,std))
    # if data_type == 'wisdom' and depth:
    #   transforms.append(T.FixedSizeCrop((768,1024)))
    return T.Compose(transforms)

import argparse
parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=True)
parser.add_argument("--type", default="final", type=str, help="input type")
parser.add_argument("--depth_path", default="../cas_data_regular/depth_ims/image_000000.png", type=str, help="path of checkpoint")
parser.add_argument("--img_path", default="../cas_data_regular/color_ims/image_000000.png", type=str, help="path of checkpoint")
parser.add_argument("--rgb_path", default="../cas_data_regular/color_ims/image_000000.png", type=str, help="path of checkpoint")
parser.add_argument("--thrs", default=0.5, type=float, help="path of checkpoint")

# for DA_only
# parser.add_argument("--resume", default="/home/prem/ur_grasping_test/src/cas/src/sd_maskrcnn_pytorch/checkpoints_bigger_30k_again/model_14.pth", type=str, help="path of checkpoint")
# for DA_GCS
parser.add_argument("--resume", default="/home/prem/ur_grasping_test/src/cas/data/checkpoints/checkpoints_DA_GCS_branch_v1/model_15.pth", type=str, help="path of checkpoint")
parser.add_argument("--data_path", default="/home/prem/ur_grasping_test/src/cas/data/ours_tase_high_clutter_data", type=str, help="dataset path")
parser.add_argument("--out_folder_name", default="DA_GCS_V1", type=str, help="dataset path")



parser.add_argument("--train_depth_only", action="store_true", help="train_depth_only")
parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
parser.add_argument('--min_depth_eval',type=float, help='minimum depth for evaluation', default=1e-1)
parser.add_argument('--max_depth_eval',type=float, help='maximum depth for evaluation', default=1.0)
parser.add_argument('--finetune',type=str, help='to finetune a particular decoder head', default=None)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--input_height',type=int,   help='input height', default=480)
parser.add_argument('--input_width',type=int,   help='input width',  default=640)
parser.add_argument(
  "--max_depth",
  default=1.0,
  type=float,
  help="max possible value of depth in the scene",
)

parser.add_argument("--has_gcs_branch", action="store_true", help="indicate if model has an gcs branch")

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = custom_model(num_classes=2,params=args)

model.to(device)

checkpoint = torch.load(args.resume, map_location="cpu")
model.load_state_dict(checkpoint["model"])

model.eval()


INSTANCE_CATEGORY_NAMES = [
    '__background__', 'object'
]

from PIL import Image 
import cv2
import matplotlib.pyplot as plt
import random


if 'cas_sim_dm' in args.data_path:
    data_type = 'ours-sim'
    detection = True
    depth = False
    # real = False
elif 'ours' in args.data_path:
    data_type = 'ours'
    detection = True
    depth = False #True
    # real = True
elif 'wisdom' in args.data_path:
    data_type = 'wisdom'
    detection = True
    depth = True
    # real = True
elif 'cas_sim' in args.data_path:
    data_type = 'ours-sim'
    detection = True
    depth = False

img_transform = get_transform(data_type, depth)

print('detection',detection)
print('data_type',data_type)

def get_prediction(depth_path, img_path, threshold):
  if args.type == 'LDD':
    L = Image.open(img_path).convert("L")
    L = np.array(L)

    img = Image.open(depth_path).convert("RGB")
    img = np.array(img)
    img[:,:,0] = L
    img = Image.fromarray(img.astype(np.uint8))
  elif args.type == 'LLL':
    L = Image.open(img_path).convert("L")
    L = np.array(L)

    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img[:,:,0] = L
    img[:,:,1] = L
    img[:,:,2] = L
    img = Image.fromarray(img.astype(np.uint8))

  elif args.type == 'RGD':
    # load images and masks
    RGB = Image.open(img_path).convert("RGB")
    RGB = np.array(RGB)

    img = Image.open(depth_path).convert("RGB")
    img = np.array(img)
    img[:,:,0] = RGB[:,:,0]
    img[:,:,1] = RGB[:,:,1]
    img = Image.fromarray(img.astype(np.uint8))

  elif args.type == 'RGB':
    # load images and masks
    img = Image.open(img_path).convert("RGB")

  elif args.type == 'DDD':
    # load images and masks
    img = Image.open(depth_path).convert("RGB")

  elif args.type == 'final':
    print('final',img_path)
    img = Image.open(img_path).convert("RGB")
    if data_type == 'wisdom':
        focal = np.array([1105.0,1105.0]) # wisdom-real data
    elif data_type == 'ours':
        focal = np.array([614.72,614.72])
    elif data_type == 'ours-sim':
        focal = np.array([307.36,307.07])

  focal = torch.as_tensor(focal)
  img,_,_ = img_transform(img,None,None)

  model([img.to(device)],focal=focal.to(device))
  st = time.time()
  inference_time = 0.0
  if depth:
    pred, depth_output = model.inference([img.to(device)],focal.to(device))
    depth_output = 255*depth_output.detach().cpu().numpy()
    inference_time = time.time()-st
  else:
    st = time.time()
    pred = model.inference([img.to(device)])
    inference_time = time.time()-st
    print('Inference time:segmentation',inference_time)
    depth_output = None
  print('inference time',time.time()-st)
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())

  
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  if len(masks.shape) < 3:
    masks = np.expand_dims(masks,axis=0)
  pred_class = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
  pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]

  pred_score = pred_score[:pred_t+1]

  if args.has_gcs_branch:
    gcs_scores = list(pred[0]['gcs'].detach().cpu().numpy())[:pred_t+1]
  else:
    gcs_scores = None
  print('pred_score',pred_score)
  print('gcs_score',gcs_scores)
  return masks, pred_boxes, pred_score , pred_class , gcs_scores, depth_output, inference_time

def random_colour_masks(image):
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask

def instance_segmentation_api(depth_path,img_path, threshold=0.25, rect_th=3, text_size=1, text_th=3, seq=0, out_path='.'):
  
  masks, boxes, pred_score,pred_cls, gcs_scores, depth_output, inference_time  = get_prediction(depth_path,img_path, threshold)
  # print(boxes)
  


  # saving masks and bboxes
  np.save(out_path+'/masks_{0}.npy'.format(seq),masks)
  np.save(out_path+'/boxes_{0}.npy'.format(seq),boxes)
  np.save(out_path+'/scores_{0}.npy'.format(seq),pred_score)
  np.save(out_path+'/gcs_scores_{0}.npy'.format(seq),gcs_scores)

  if depth:
    print('depth out size',depth_output.shape)
    cv2.imwrite(out_path+'/depth_{0}.jpg'.format(seq),depth_output)
  if detection:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,(512,512))
    mask_img = np.zeros_like(img).astype(np.uint8)
    for i in range(len(masks)):
      rgb_mask = random_colour_masks(masks[i])
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      mask_img += rgb_mask
      if args.has_gcs_branch:
        cv2.putText(img,str(int(100*gcs_scores[i].item())), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0),thickness=text_th)
    img = cv2.addWeighted(img, 0.25, mask_img, 0.75, 0)
      # img = rgb_mask
      
      
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path+'/seg_{0}.jpg'.format(seq),img)
    # cv2.imwrite(out_path+'/seg_mask_{0}.jpg'.format(seq),mask_img)
  return inference_time
  # plt.figure(figsize=(20,30))
  # plt.imshow(img)
  # plt.xticks([])
  # plt.yticks([])
  # # plt.show()
  # plt.savefig(out_path+'/{0}'.format(seq))

def create_directory(dname):
  if not os.path.exists(dname):
      print('creating directory:',dname)
      os.makedirs(dname)



# ************* Main Code ****************************************
data_path = args.data_path
out_path = data_path + '/out_{0}'.format(args.out_folder_name)
create_directory(out_path)
imgs = list(sorted(os.listdir(os.path.join(data_path, "color_ims"))))
try:
  dmaps = list(sorted(os.listdir(os.path.join(data_path, "depth_ims"))))
except:
  dmaps = None
# test_indices = np.load(data_path+'/test_indices.npy').tolist()
test_indices = np.arange(len(imgs)).tolist()
# test_indices = [0,6]

avg_time = 0.0
total = 0
# For loop to iterate over all the samples
for idx in tqdm(test_indices):
  i = int(idx)
  img_path = os.path.join(data_path, "color_ims", imgs[i])
  if dmaps is not None:
    depth_path = os.path.join(data_path, "depth_ims", dmaps[i])
  depth_path = None
  inference_time = instance_segmentation_api(depth_path,img_path,threshold=args.thrs, seq=i, out_path=out_path)
  avg_time += inference_time
  total += 1

avg_time = avg_time/total
print('avg inference time',1000*avg_time)
