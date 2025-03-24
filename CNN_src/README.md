# CNN Source code 

![Alt Text](data/images/segmentation_with_gcs.png)



# Training with multiple GPUs:
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --batch-size 5 --world-size 2 --lr 0.005 --has_gcs_branch

#Single GPU training
python train.py --batch-size 5 --lr 0.0005 --has_gcs_branch

# Config
Check the config options in the train.py file
Some of the important config options are:
(1) --data_path
(2) --train_depth_only
(3) --train_segm_only
(4) --min_depth_eval, --max_depth_eval
(5) --max_depth
(6) --output_dir
(7) --resume
(8) --test-only



#training without aspect ratio
--aspect-ratio-group-factor -1


#for group inference
python inference_group.py --resume /checkpoint_path/checkpoint.pth --has_gcs_branch
