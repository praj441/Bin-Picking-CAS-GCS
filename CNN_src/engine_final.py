import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils_final import get_coco_api_from_dataset
import numpy as np
import copy

def train_one_epoch(model, optimizer, data_loader, data_loader_test, device, epoch, print_freq, scaler=None, data_path='temp',args=None,detection_only=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    AP_list = []
    AR_list = []
    avg_depth_loss = 0.0
    max_AP = -1.0
    max_AR = -1.0
    state_dict = None
    for i,data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if i%100 == 0 and i > 0:
            coco_evaluator = evaluate(model, data_loader_test, device=device,args=args,detection_only=detection_only)
            segm =  coco_evaluator.coco_eval['segm']
            AP = segm.stats[0]
            AR = segm.stats[8]
            

            AP_list.append(AP)
            AR_list.append(AR)

            np.savetxt(data_path+'/AP_{0}.txt'.format(epoch),AP_list,fmt='%0.3f')
            np.savetxt(data_path+'/AR_{0}.txt'.format(epoch),AR_list,fmt='%0.3f')
            print(epoch,i,AP,AR)

            if AP > max_AP:
                if args.distributed:
                    state_dict = copy.deepcopy(model.module.state_dict())
                else:
                    state_dict = copy.deepcopy(model.state_dict())
                checkpoint = {
                    "model": state_dict, #model_without_ddp.state_dict(),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                    "epoch": epoch,
                }
                max_AP = AP
                max_AR = AR




        model.train()
        images, targets, aux_targets , focal = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        aux_targets = torch.stack(aux_targets)
        aux_targets = aux_targets.to(device)
        focal =  torch.stack(focal)
        focal = focal.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # for target in targets:
                # print('train',target['boxes'])
            loss_dict = model(images, targets,aux_targets, focal) #forward pass
            if args.train_depth_only:
                    losses = loss_dict['depth_loss']/args.depth_loss_weight
            else:
                losses = sum(loss for loss in loss_dict.values())

        avg_depth_loss += loss_dict['depth_loss']/args.depth_loss_weight
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            # sys.exit(1)
            continue

        optimizer.zero_grad()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    avg_depth_loss = avg_depth_loss/i
    return metric_logger, checkpoint, max_AP, max_AR,avg_depth_loss


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

@torch.inference_mode()
def evaluate(model, data_loader, device, args, detection_only=False):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    iou_types = ["bbox","segm"]
    # print('*******',iou_types)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    eval_measures = torch.zeros(10).to(device)

    for images, targets, aux_targets , focal in metric_logger.log_every(data_loader, 100, header):
        # args.max_depth = aux_targets[0].max().item()
        # print('keep going')
        images = list(img.to(device) for img in images)
        focal =  torch.stack(focal)
        focal = focal.to(device)
        aux_targets = torch.stack(aux_targets)
        aux_targets = aux_targets.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        if detection_only:
            outputs = model(images)
        else:
            outputs,pred_depth = model(images,focal=focal)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        #********* For distance based category training *************
        for t in outputs:
            t['labels'] = np.where(t['labels']>1,1,t['labels'])
            # print(t['labels'])
            # input('ruk')
        #************************************************************


        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)


        #evaluation code for depth estimation
        if not detection_only:
            gt_depth = aux_targets
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            if len(gt_depth.shape)==4:
                if gt_depth.shape[3]==3:
                    gt_depth = gt_depth[:,:,:,0]

            max_depth_eval = gt_depth.max()
            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > max_depth_eval] = max_depth_eval
            pred_depth[np.isinf(pred_depth)] = max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval
            # print(gt_depth)
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < max_depth_eval)

            # if args.garg_crop or args.eigen_crop:
            #     gt_height, gt_width = gt_depth.shape
            #     eval_mask = np.zeros(valid_mask.shape)

            #     if args.garg_crop:
            #         eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            #     elif args.eigen_crop:
            #         if args.dataset == 'kitti':
            #             eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            #         else:
            #             eval_mask[45:471, 41:601] = 1

            #     valid_mask = np.logical_and(valid_mask, eval_mask)

            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            eval_measures[:9] += torch.tensor(measures).to(device)
            eval_measures[9] += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                 'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                 'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.3f}'.format(eval_measures_cpu[8]))


    return coco_evaluator
