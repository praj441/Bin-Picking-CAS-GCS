import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine_final import train_one_epoch, evaluate
import utils
import transforms_final as T
# from torchvision.transforms import transforms as Tf
from dataset import BinDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
from model import custom_model


mean = (128,128,128)
std = (64,64,64)

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.PILToTensor())
    transforms.append(T.Normalize(mean,std))
    # if train:
    #     # during training, randomly flip the training images
    #     # and ground-truth for data augmentation
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


import argparse
parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=True)
parser.add_argument("--resume", default="/home/prem/ur_grasping_test/src/cas/data/checkpoints/checkpoints_DA_GCS_branch_v1/model_15.pth", type=str, help="path of checkpoint")
parser.add_argument("--type", default="final", type=str, help="input type")
# parser.add_argument("--data_type", type=str,required = True, help="data type (e.g. ours, wisdom)")
parser.add_argument("--data_path", default='../../data/wisdom-real-cropped', type=str, help="dataset path")
parser.add_argument("--dataset", default="coco", type=str, help="dataset name")

parser.add_argument("--train_depth_only", action="store_true", help="train_depth_only")
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--finetune',type=str, help='to finetune a particular decoder head', default=None)
parser.add_argument('--min_depth_eval',type=float, help='minimum depth for evaluation', default=1e-1)
parser.add_argument('--max_depth_eval',type=float, help='maximum depth for evaluation', default=1.0)
parser.add_argument(
    "--max_depth",
    default=1.0,
    type=float,
    help="max possible value of depth in the scene",
)

parser.add_argument("--has_gcs_branch", action="store_true", help="indicate if model has an gcs branch")
# parser.add_argument("--real", default=False, type=bool, help="dataset name")
args = parser.parse_args()

# if args.data_path == '../cas_sim':
#     real = False
# else:
real = True

if 'cas_sim_dm' in args.data_path:
    eval_data_type = 'ours-sim'
    detection_only = True
elif 'ours' in args.data_path:
    eval_data_type = 'ours'
    detection_only = False
elif 'wisdom' in args.data_path:
    eval_data_type = 'wisdom'
    detection_only = False
elif 'wisdom-real-cropped' in args.data_path:
    eval_data_type = 'wisdom'
    detection_only = False

if 'sim' in args.data_path:
    label_issue = True
else:
    label_issue = False

print('label_issue',label_issue)
print('eval_data_type',eval_data_type)
print('detection_only',detection_only)
# data_path = '../sd-maskrcnn/dataset/wisdom/wisdom-real/high-res'
data_path = args.data_path
dataset_test = BinDataset(data_path, get_transform(train=False), type=args.type, label_issue=label_issue,train=False,eval_data_type=eval_data_type, has_gcs_branch=args.has_gcs_branch)


test_indices = np.load(data_path+'/test_indices.npy').tolist()
dataset_test = torch.utils.data.Subset(dataset_test, test_indices)
# split the dataset in train and test set

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=8, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = custom_model(num_classes=2,params=args)

model.to(device)

checkpoint = torch.load(args.resume)#, map_location="cpu")
model.load_state_dict(checkpoint["model"])

model.eval()

evaluate(model, data_loader_test, device=device, args=args,detection_only=detection_only)