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
import matplotlib.pyplot as plt

mean = (128,128,128)
std = (64,64,64)

def get_transform(train,args):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.PILToTensor())
    transforms.append(T.Normalize(mean,std))
    return T.Compose(transforms)


import argparse
parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=True)
parser.add_argument("--resume_path", default="checkpoints_novel_RGB", type=str, help="path of checkpoint")
parser.add_argument("--type", default="final", type=str, help="input type")
parser.add_argument("--data_type", type=str,default='wisdom', help="data type (e.g. ours, wisdom)")
parser.add_argument("--data_path", default='../../data/wisdom-real', type=str, help="dataset path")
parser.add_argument("--dataset", default="coco", type=str, help="dataset name")

parser.add_argument('--min_depth_eval',type=float, help='minimum depth for evaluation', default=1e-1)
parser.add_argument('--max_depth_eval',type=float, help='maximum depth for evaluation', default=1.0)
parser.add_argument('--finetune',type=str, help='to finetune a particular decoder head', default=None)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument(
    "--max_depth",
    default=1.0,
    type=float,
    help="max possible value of depth in the scene",
)
# parser.add_argument("--real", default=False, type=bool, help="dataset name")
args = parser.parse_args()

if 'cas_issue_labels' in args.data_path:
    eval_data_type = 'ours-sim'
    detection_only = True
    label_issue = True
elif 'ours' in args.data_path:
    eval_data_type = 'ours'
    detection_only = False
    label_issue = False
elif 'wisdom' in args.data_path:
    eval_data_type = 'wisdom'
    detection_only = True
    label_issue = False
elif 'cas_sim' in args.data_path:
    eval_data_type = 'ours-sim'
    detection_only = True
    label_issue = True

print(args)
print('label_issue',label_issue)
print('eval_data_type',eval_data_type)
data_path = args.data_path
dataset_test = BinDataset(data_path, get_transform(train=False,args=args), type=args.type, label_issue=label_issue,train=False,eval_data_type=eval_data_type)


test_indices = np.load(data_path+'/test_indices.npy').tolist()
dataset_test = torch.utils.data.Subset(dataset_test, test_indices)
# split the dataset in train and test set

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=8, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = custom_model(num_classes=2,params=args)

model.to(device)

AP_list = []
AR_list = []
for epoch in range(26):
    checkpoint_path = args.resume_path + '/' + 'model_{0}.pth'.format(epoch)
    checkpoint = torch.load(checkpoint_path)#, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model.eval()

    coco_evaluator = evaluate(model, data_loader_test, device=device, args=args, detection_only=detection_only)
    segm =  coco_evaluator.coco_eval['segm']
    AP = segm.stats[0]
    AR = segm.stats[8]
    AP_list.append(AP)
    AR_list.append(AR)
    np.savetxt(data_path+'/AP_{0}.txt'.format(args.type),AP_list,fmt='%0.3f')
    np.savetxt(data_path+'/AR_{0}.txt'.format(args.type),AR_list,fmt='%0.3f')
    print(epoch,AP,AR)

    plt.plot(AP_list)
    plt.plot(AR_list)
    plt.savefig(data_path+'/result_multi_{0}.png'.format(args.type))
    # plt.show()
