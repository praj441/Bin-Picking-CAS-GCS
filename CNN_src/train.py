r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

	python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
		train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
	--lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
	--epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
	--epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
# from coco_utils import get_coco, get_coco_kp
from engine_final import evaluate, train_one_epoch
from group_by_aspect_ratio_final import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms_final import SimpleCopyPaste
import transforms_final as T
from dataset import BinDataset
import numpy as np

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from model import custom_model
import sys
import matplotlib.pyplot as plt

def copypaste_collate_fn(batch):
	copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
	return copypaste(*utils.collate_fn(batch))


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def get_transform(train):
	transforms = []
	# converts the image, a PIL image, into a PyTorch Tensor
	transforms.append(T.ToTensor())
	transforms.append(T.Normalize(mean,std))
	
	if train:
		# transforms.append(T.FixedSizeCrop((224,320)))
		# during training, randomly flip the training images
		# and ground-truth for data augmentation
		transforms.append(T.RandomHorizontalFlip(0.5))
		transforms.append(T.RandomPhotometricDistort(0.5))
		# transforms.append(T.ScaleJitter(target_size=(480,640),scale_range=(0.8,1.2)))
	return T.Compose(transforms)




def get_args_parser(add_help=True):
	import argparse

	parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
	parser.add_argument("--type", default="final", type=str, help="input type")
	parser.add_argument("--data_path", default="../../data/cas_sim_dm7", type=str, help="dataset path")
	parser.add_argument("--data_path_test", default="../../data/wisdom-real", type=str, help="dataset path")
	parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
	parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
	parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
	parser.add_argument("--output_dir", default="/home/prem/ur_grasping_test/src/cas/data/checkpoints/checkpoints_temp", type=str, help="path to save outputs")
	parser.add_argument("--resume", default="", type=str, help="path of checkpoint")

	parser.add_argument("--train_depth_only", action="store_true", help="train_depth_only")
	parser.add_argument("--train_segm_only", action="store_true", help="train_depth_only")

	parser.add_argument('--depth_loss_weight',default=2.0,type=float, help='minimum depth for evaluation')
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
	parser.add_argument(
		"-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
	)
	parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
	parser.add_argument(
		"-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
	)
	parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
	parser.add_argument(
		"--lr",
		default=0.01,
		type=float,
		help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
	)
	parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
	parser.add_argument(
		"--wd",
		"--weight-decay",
		default=1e-4,
		type=float,
		metavar="W",
		help="weight decay (default: 1e-4)",
		dest="weight_decay",
	)
	parser.add_argument(
		"--norm-weight-decay",
		default=None,
		type=float,
		help="weight decay for Normalization layers (default: None, same value as --wd)",
	)
	parser.add_argument(
		"--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
	)
	parser.add_argument(
		"--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
	)
	parser.add_argument(
		"--lr-steps",
		default=[16, 22],
		nargs="+",
		type=int,
		help="decrease lr every step-size epochs (multisteplr scheduler only)",
	)
	parser.add_argument(
		"--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
	)
	parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
	parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
	parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
	parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
	parser.add_argument(
		"--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
	)
	parser.add_argument(
		"--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
	)
	parser.add_argument(
		"--sync-bn",
		dest="sync_bn",
		help="Use sync batch norm",
		action="store_true",
	)
	parser.add_argument(
		"--test-only",
		dest="test_only",
		help="Only test the model",
		action="store_true",
	)

	parser.add_argument(
		"--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
	)

	# distributed training parameters
	parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
	parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
	parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
	parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

	# Mixed precision training parameters
	parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

	# Use CopyPaste augmentation training parameter
	parser.add_argument(
		"--use-copypaste",
		action="store_true",
		help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
	)

	parser.add_argument("--has_gcs_branch", action="store_true", help="indicate if model has an gcs branch")
	
	parser.add_argument("--real_noise", action="store_true", help="use real world camera noise") # Not used


	return parser



def main(args):
	if args.output_dir:
		utils.mkdir(args.output_dir)

	utils.init_distributed_mode(args)
	print(args)

	device = torch.device(args.device)

	if args.use_deterministic_algorithms:
		torch.use_deterministic_algorithms(True)

	# Data loading code
	print("Loading data")

	data_path = args.data_path
	data_path_test = args.data_path_test
	if not os.path.exists(data_path_test):
		print('path',data_path_test,'does not exist')
		sys.exit()
	if 'cas_sim_dm' in args.data_path_test:
		eval_data_type = 'ours-sim'
		detection_only = True
	elif 'ours' in args.data_path_test:
		eval_data_type = 'ours'
		detection_only = False
	elif 'wisdom' in args.data_path_test:
		eval_data_type = 'wisdom'
		detection_only = True

	if 'sim' in data_path:
		label_issue = True
	else:
		label_issue = False

	print('train_depth_only',args.train_depth_only)
	print('label_issue',label_issue)

	dataset = BinDataset(data_path, get_transform(train=True), type=args.type,label_issue=label_issue, real_noise=args.real_noise, has_gcs_branch=args.has_gcs_branch)
	dataset_test = BinDataset(data_path_test, get_transform(train=False), type=args.type,train=False,eval_data_type=eval_data_type,has_gcs_branch=args.has_gcs_branch)

	# split the dataset in train and test set

	train_indices = np.load(data_path+'/train_indices.npy').tolist()
	test_indices = np.load(data_path_test+'/test_indices.npy').tolist()

	dataset = torch.utils.data.Subset(dataset, train_indices)
	dataset_test = torch.utils.data.Subset(dataset_test, test_indices)

	print("Creating data loaders")
	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
		test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
	else:
		train_sampler = torch.utils.data.RandomSampler(dataset)
		test_sampler = torch.utils.data.SequentialSampler(dataset_test)

	if args.aspect_ratio_group_factor >= 0:
		group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
		train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
	else:
		train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

	train_collate_fn = utils.collate_fn
	if args.use_copypaste:
		if args.data_augmentation != "lsj":
			raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

		train_collate_fn = copypaste_collate_fn

	data_loader = torch.utils.data.DataLoader(
		dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
	)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=4, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
	)

	print("Creating model")
	kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
	if args.data_augmentation in ["multiscale", "lsj"]:
		kwargs["_skip_resize"] = True
	if "rcnn" in args.model:
		if args.rpn_score_thresh is not None:
			kwargs["rpn_score_thresh"] = args.rpn_score_thresh

	model = custom_model(num_classes=2,params=args)
	model.to(device)
	if args.distributed and args.sync_bn:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

	model_without_ddp = model
	if args.distributed:
		if args.finetune is not None or args.train_depth_only or args.train_segm_only:
			find_unused_parameters = True 
		else:
			find_unused_parameters = False
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=find_unused_parameters)
		model_without_ddp = model.module

	if args.norm_weight_decay is None:
		parameters = [p for p in model.parameters() if p.requires_grad]
	else:
		param_groups = torchvision.ops._utils.split_normalization_params(model)
		wd_groups = [args.norm_weight_decay, args.weight_decay]
		parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

	opt_name = args.opt.lower()
	if opt_name.startswith("sgd"):
		optimizer = torch.optim.SGD(
			parameters,
			lr=args.lr,
			momentum=args.momentum,
			weight_decay=args.weight_decay,
			nesterov="nesterov" in opt_name,
		)
	elif opt_name == "adamw":
		optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
	else:
		raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

	scaler = torch.cuda.amp.GradScaler() if args.amp else None

	args.lr_scheduler = args.lr_scheduler.lower()
	if args.lr_scheduler == "multisteplr":
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
	elif args.lr_scheduler == "cosineannealinglr":
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	else:
		raise RuntimeError(
			f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
		)

	if args.resume:
		checkpoint = torch.load(args.resume, map_location="cpu")
		model_without_ddp.load_state_dict(checkpoint["model"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
		args.start_epoch = checkpoint["epoch"] + 1
		if args.amp:
			scaler.load_state_dict(checkpoint["scaler"])

	if args.test_only:
		torch.backends.cudnn.deterministic = True
		evaluate(model, data_loader_test, device=device)
		return

	print("Start training")
	start_time = time.time()
	AP_list = []
	AR_list = []
	depth_loss_list = []
	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)
		# evaluate(model, data_loader_test, device=device, args=args, detection_only=detection_only)
		_,checkpoint, AP, AR,avg_depth_loss = train_one_epoch(model, optimizer, data_loader, data_loader_test, device, epoch, args.print_freq, scaler, data_path=args.output_dir,args=args,detection_only=detection_only)
		AP_list.append(AP)
		AR_list.append(AR)
		depth_loss_list.append(avg_depth_loss.item())
		lr_scheduler.step()
		if args.output_dir:
			checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
			if args.amp:
				checkpoint["scaler"] = scaler.state_dict()
			utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
			utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
			print('Checkpoint saved:',os.path.join(args.output_dir, f"model_{epoch}.pth"))
		# evaluate after every epoch
		# evaluate(model, data_loader_test, device=device, args=args)
		plt.plot(AP_list)
		plt.plot(AR_list)
		plt.savefig(args.output_dir+'/AP_AR_plot.png')
		plt.clf()

		plt.plot(depth_loss_list)
		plt.savefig(args.output_dir+'/depth_loss_plot.png')
		plt.clf()
	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print(f"Training time {total_time_str}")


if __name__ == "__main__":
	args = get_args_parser().parse_args()
	main(args)
