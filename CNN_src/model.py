import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from bts import bts, weights_init_xavier,silog_loss
import transforms_final as T
from torch.nn.functional import interpolate

# a dict to store the activations
activation = {}
def getActivation(name):
	# the hook signature
	def hook(model, input, output):
		activation[name] = output #.detach()
	return hook


class GCSPredictor(nn.Module):
    """
    Standard GCS score regression layers
    for Fast R-CNN. # GCS stands for grasp confidence score

    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()
        self.hidden1 = nn.Linear(in_channels, 512) #in_channels=1024
        self.hidden2 = nn.Linear(512, 128)
        self.gcs_score = nn.Linear(128, 1)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        out1 = F.relu(self.hidden1(x))
        out2 = F.relu(self.hidden2(out1))
        gcs_scores = F.sigmoid(self.gcs_score(out2))
        return gcs_scores

def get_model_instance_segmentation(num_classes,has_gcs_branch=False):
	# load an instance segmentation model pre-trained on COCO
	model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	# now get the number of input features for the mask classifier
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	hidden_layer = 256
	# and replace the mask predictor with a new one
	model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
														 hidden_layer,
														 num_classes)

	# add gcs_predictor in the model
	if has_gcs_branch:
		model.roi_heads.gcs_predictor = GCSPredictor(in_features)
		model.roi_heads.gcs_loss = torch.nn.MSELoss()
		model.roi_heads.has_gcs_branch = True
	else:
		model.roi_heads.has_gcs_branch = False

	return model


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

mse_loss = torch.nn.MSELoss()
class custom_model(torch.nn.Module):
	def __init__(self,num_classes=2,params=None,has_gcs_branch=False):
		super().__init__()
		self.params = params
		try:
			has_gcs_branch = params.has_gcs_branch
		except:
			has_gcs_branch = False
		self.main_model = get_model_instance_segmentation(num_classes,has_gcs_branch)
		
		grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=480, max_size=640, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],_skip_resize=True)
		self.main_model.transform = grcnn
		
		self.feat_out_channels = [64, 256, 512, 1024, 2048] # for resnet-50

		self.depth_decoder = bts(params, self.feat_out_channels, params.bts_size)                                                                  
		self.depth_decoder.apply(weights_init_xavier)

		h0 = self.main_model.transform.register_forward_hook(getActivation('transform'))
		h1 = self.main_model.backbone.body.relu.register_forward_hook(getActivation('relu'))
		h2 = self.main_model.backbone.body.layer1[2].register_forward_hook(getActivation('layer1'))
		h3 = self.main_model.backbone.body.layer2[3].register_forward_hook(getActivation('layer2'))
		h4 = self.main_model.backbone.body.layer3[5].register_forward_hook(getActivation('layer3'))
		h5 = self.main_model.backbone.body.layer4[2].register_forward_hook(getActivation('layer4'))
		# h6 = self.main_model.roi_heads.box_head.register_forward_hook(getActivation('box_features'))
		# h7 = self.main_model.roi_heads.register_forward_hook(getActivation('results'))

		self.silog_criterion = silog_loss(variance_focus=params.variance_focus)

	def forward(self,images, targets=None, depth_gt=None, focal=None):
		h,w = images[0].shape[-2:]
		# print('*********',h,w)
		if targets is not None:   # training branch
			# forward pass for segmentation branch
			loss_dict = self.main_model(images, targets)

			if self.params.train_segm_only:
				return loss_dict
			else:
				# forward pass for depth branch
				lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est,_ = self.forward_depth_decoder(focal,h,w)
				depth_gt = depth_gt.unsqueeze(1)
				mask = depth_gt > 0.1
				depth_loss = self.silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
				print('depth_loss',depth_loss.item())
				loss_dict['depth_loss'] = self.params.depth_loss_weight*depth_loss
				return loss_dict
					
		else:    # inference branch
			if focal is not None: 
				return self.main_model(images),self.forward_depth_decoder(focal,h,w)[4]
			else:
				return self.main_model(images)

	def inference(self,image, focal=None):
		input_shape = image[0].shape[-2:]
		preds = self.main_model(image)
		if focal is not None:
			h,w = input_shape
			# print('*********',h,w)
			lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est,aux_features = self.forward_depth_decoder(focal,h,w)
			output = depth_est.squeeze(1).squeeze(0)
			return preds, output
		else:
			return preds

	def forward_depth_decoder(self,focal,h,w):
		features = []

		f1 = activation['relu']
		f2 = activation['layer1']
		f3 = activation['layer2']
		f4 = activation['layer3']
		f5 = activation['layer4']

		features.append(f1)
		features.append(f2)
		features.append(f3)
		features.append(f4)
		features.append(f5)

		return self.depth_decoder(features, focal)