# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Modified by Ankoor Bhagat, based on code from Jiasen Lu, Jianwei Yang,
# which is based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob
import sys
import pdb
import pprint
import time
import _init_paths
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient, apply_augmentations
import json


# Data Sampler
# Every Sampler subclass has to provide an __iter__ method, providing a way to iterate over indices of dataset
# elements, and a __len__ method that returns the length of the returned iterators.
class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def delete_older_checkpoints(path):
    files = list(glob.glob(path))
    files.sort(key=lambda x: os.path.getmtime(x), reverse=False)
    files.pop()
    for f in files:
        print("removing older checkpoint file {}".format(f))
        os.remove(f)


def train(dataset="kaggle_pna", train_ds="train", arch="couplenet", net="res152", start_epoch=1, max_epochs=20,
          disp_interval=100, save_dir="save", num_workers=4, cuda=True, large_scale=False, mGPUs=True, batch_size=4,
          class_agnostic=False, anchor_scales=4, optimizer="sgd", lr_decay_step=10, lr_decay_gamma=.1, session=1,
          resume=False, checksession=1, checkepoch=1, checkpoint=0, use_tfboard=False, flip_prob=0.0, scale=0.0,
          scale_prob=0.0, translate=0.0, translate_prob=0.0, angle=0.0, dist="cont", rotate_prob=0.0,
          shear_factor=0.0, shear_prob=0.0, rpn_loss_cls_wt=1, rpn_loss_box_wt=1, RCNN_loss_cls_wt=1,
          RCNN_loss_bbox_wt=1, **kwargs):
    print("Train Arguments: {}".format(locals()))

    # Import network definition
    if arch == 'rcnn':
        from model.faster_rcnn.resnet import resnet
    elif arch == 'rfcn':
        from model.rfcn.resnet_atrous import resnet
    elif arch == 'couplenet':
        from model.couplenet.resnet_atrous import resnet

    from roi_data_layer.pnaRoiBatchLoader import roibatchLoader
    from roi_data_layer.pna_roidb import combined_roidb

    print('Called with kwargs:')
    print(kwargs)

    # Set up logger
    if use_tfboard:
        from model.utils.logger import Logger
        # Set the logger
        logger = Logger('./logs')

    # Anchor settings: ANCHOR_SCALES: [8, 16, 32] or [4, 8, 16, 32]
    if anchor_scales == 3:
        scales = [8, 16, 32]
    elif anchor_scales == 4:
        scales = [4, 8, 16, 32]

    # Dataset related settings: MAX_NUM_GT_BOXES: 20, 30, 50
    if train_ds == "train":
        imdb_name = "pna_2018_train"
    elif train_ds == "trainval":
        imdb_name = "pna_2018_trainval"

    set_cfgs = ['ANCHOR_SCALES', str(scales), 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

    import model
    model_repo_path = os.path.dirname(os.path.dirname(os.path.dirname(model.__file__)))

    cfg_file = "cfgs/{}_ls.yml".format(net) if large_scale else "cfgs/{}.yml".format(net)

    if cfg_file is not None:
        cfg_from_file(os.path.join(model_repo_path, cfg_file))
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)

    train_kwargs = kwargs.pop("TRAIN", None)
    resnet_kwargs = kwargs.pop("RESNET", None)
    mobilenet_kwargs = kwargs.pop("MOBILENET", None)

    if train_kwargs is not None:
        for key, value in train_kwargs.items():
            cfg["TRAIN"][key] = value

    if resnet_kwargs is not None:
        for key, value in resnet_kwargs.items():
            cfg["RESNET"][key] = value

    if mobilenet_kwargs is not None:
        for key, value in mobilenet_kwargs.items():
            cfg["MOBILENET"][key] = value

    if kwargs is not None:
        for key, value in kwargs.items():
            cfg[key] = value

    print('Using config:')
    cfg.MODEL_DIR = os.path.abspath(cfg.MODEL_DIR)
    cfg.TRAIN_DATA_CLEAN_PATH = os.path.abspath(cfg.TRAIN_DATA_CLEAN_PATH)
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    print("LEARNING RATE: {}".format(cfg.TRAIN.LEARNING_RATE))

    # Warning to use cuda if available
    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Train set
    # Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
    train_size = len(roidb)

    print('{:d} roidb entries'.format(len(roidb)))

    # output_dir = os.path.join(save_dir, arch, net, dataset)
    output_dir = cfg.MODEL_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler_batch,
                                             num_workers=num_workers)

    # Initilize the tensor holder
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # Copy tensors in CUDA memory
    if cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # Make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if cuda:
        cfg.CUDA = True

    # Initilize the network:
    if net == 'vgg16':
        # model = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
        print("Pretrained model is not downloaded and network is not used")
    elif net == 'res18':
        model = resnet(imdb.classes, 18, pretrained=False, class_agnostic=class_agnostic)  # TODO: Check dim error
    elif net == 'res34':
        model = resnet(imdb.classes, 34, pretrained=False, class_agnostic=class_agnostic)  # TODO: Check dim error
    elif net == 'res50':
        model = resnet(imdb.classes, 50, pretrained=False, class_agnostic=class_agnostic)  # TODO: Check dim error
    elif net == 'res101':
        model = resnet(imdb.classes, 101, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res152':
        model = resnet(imdb.classes, 152, pretrained=True, class_agnostic=class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    # Create network architecture
    model.create_architecture()

    # Update model parameters
    lr = cfg.TRAIN.LEARNING_RATE
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    # Optimizer
    if optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # Resume training
    if resume:
        load_name = os.path.join(output_dir,
                                 '{}_{}_{}_{}.pth'.format(arch, checksession, checkepoch, checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        session = checkpoint['session'] + 1
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    # Train on Multiple GPUS
    if mGPUs:
        model = nn.DataParallel(model)

    # Copy network to CUDA memroy
    if cuda:
        model.cuda()

    # Training loop
    iters_per_epoch = int(train_size / batch_size)

    sys.stdout.flush()

    for epoch in range(start_epoch, max_epochs + 1):
        # remove batch re-sizing for augmentation or adjust?
        dataset.resize_batch()

        # Set model to train mode
        model.train()
        loss_temp = 0
        start = time.time()

        # Update learning rate as per decay step
        if epoch % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        # Get batch data and train
        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            sys.stdout.flush()
            data = next(data_iter)

            # Apply augmentations
            aug_img_tensors, aug_bbox_tensors = apply_augmentations(data[0], data[2], flip_prob=flip_prob, scale=scale,
                                                                    scale_prob=scale_prob, translate=translate,
                                                                    translate_prob=translate_prob, angle=angle,
                                                                    dist=dist, rotate_prob=rotate_prob,
                                                                    shear_factor=shear_factor, shear_prob=shear_prob)

            # im_data.data.resize_(data[0].size()).copy_(data[0])
            im_data.data.resize_(aug_img_tensors.size()).copy_(aug_img_tensors)
            im_info.data.resize_(data[1].size()).copy_(data[1])
            # gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            gt_boxes.data.resize_(aug_bbox_tensors.size()).copy_(aug_bbox_tensors)
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            # Compute multi-task loss
            model.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = model(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls_wt * rpn_loss_cls.mean() + rpn_loss_box_wt * rpn_loss_box.mean() + \
                   RCNN_loss_cls_wt * RCNN_loss_cls.mean() + RCNN_loss_bbox_wt * RCNN_loss_bbox.mean()
            loss_temp += loss.data[0]

            # Backward pass to compute gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            if net == "vgg16":
                clip_gradient(model, 10.)
            optimizer.step()

            # Display training stats on terminal
            if step % disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= disp_interval

                if mGPUs:
                    batch_loss = loss.data[0]
                    loss_rpn_cls = rpn_loss_cls.mean().data[0]
                    loss_rpn_box = rpn_loss_box.mean().data[0]
                    loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                    loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    batch_loss = loss.data[0]
                    loss_rpn_cls = rpn_loss_cls.data[0]
                    loss_rpn_box = rpn_loss_box.data[0]
                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    loss_rcnn_box = RCNN_loss_bbox.data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\t batch_loss: %.4f, rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (batch_loss, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                if use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, step)

                loss_temp = 0
                start = time.time()

                # Save model at checkpoints
        if mGPUs:
            save_name = os.path.join(output_dir, '{}_{}_{}_{}.pth'.format(arch, session, epoch, step))
            save_checkpoint({
                'session': session,
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir, '{}_{}_{}_{}.pth'.format(arch, session, epoch, step))
            save_checkpoint({
                'session': session,
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': class_agnostic,
            }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        delete_older_checkpoints(os.path.join(cfg.MODEL_DIR, "couplenet_{}_*.pth".format(i)))
        print("Run Time: ", end - start)


if __name__ == '__main__':
    # load train config file
    train_config = json.loads(open("TRAIN_SETTINGS.json").read())
    # extract model path
    import model

    model_repo_path = os.path.dirname(os.path.dirname(os.path.dirname(model.__file__)))

    # write files in DCMImagesTrain to text file in ImageSets if it doesn't exist
    print("PREPARING TRAINING FILES...")
    data_dir = 'data/PNAdevkit/PNA2018'
    ImageSets_dir = os.path.join(model_repo_path, data_dir, 'ImageSets')
    if not os.path.exists(ImageSets_dir):
        os.makedirs(ImageSets_dir)

    train_files = os.path.join(ImageSets_dir, 'train.txt')

    d = os.path.join(train_config["TRAIN_DATA_CLEAN_PATH"])
    pids = [pid.split('.')[0] for pid in os.listdir(d)]
    with open(train_files, 'w') as f:
        for pid in pids:
            f.write("{}\n".format(pid))

    for i, random_seed in enumerate([14, 18, 20, 30]):
        print("\n\n********* STARTING TRAINING FOR RANDOM SEED {} ({}/4) *********\n\n".format(random_seed, i))
        train_config['session'] = i
        train_config["RNG_SEED"] = random_seed
        train(**train_config)
