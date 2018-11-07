# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import glob
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
import pydicom
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from datetime import datetime
import pdb
import sys
import json

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def delete_irrelevant_files(path):
    files = list(glob.glob(os.path.join(path, "*.*")))
    for f in files:
        if "cls-3" not in f:
            print("Removing {}".format(f))
            os.remove(f)


def test(dataset="kaggle_pna", test_ds="test", arch="couplenet", net="res152", load_dir="save", output_dir="output",
         cuda=True, large_scale=False, class_agnostic=False, checksession=1, checkepoch=1, checkpoint=10021,
         batch_size=1, vis=False, anchor_scales=4, min_conf=.5, **kwargs):
    print("Test Arguments: {}".format(locals()))

    # Import network definition
    if arch == 'rcnn':
        from model.faster_rcnn.vgg16 import vgg16
        from model.faster_rcnn.resnet import resnet
    elif arch == 'rfcn':
        from model.rfcn.resnet_atrous import resnet
    elif arch == 'couplenet':
        from model.couplenet.resnet_atrous import resnet

    from roi_data_layer.pnaRoiBatchLoader import roibatchLoader
    from roi_data_layer.pna_roidb import combined_roidb

    image_read_func = lambda path: pydicom.dcmread(path).pixel_array

    print('Called with kwargs:')
    print(kwargs)

    # Warning to use cuda if available
    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Anchor settings: ANCHOR_SCALES: [8, 16, 32] or [4, 8, 16, 32]
    if anchor_scales == 3:
        scales = [8, 16, 32]
    elif anchor_scales == 4:
        scales = [4, 8, 16, 32]
    else:
        scales = [8, 16, 32]

    # Dataset related settings: MAX_NUM_GT_BOXES: 20, 30, 50
    np.random.seed(cfg.RNG_SEED)

    if test_ds == "val":
        imdbval_name = "pna_2018_val"
    elif test_ds == "test":
        imdbval_name = "pna_2018_test"
    set_cfgs = ['ANCHOR_SCALES', str(scales), 'ANCHOR_RATIOS', '[0.5,1,2]']

    cfg_file = "cfgs/{}_ls.yml".format(net) if large_scale else "cfgs/{}.yml".format(net)

    import model
    model_repo_path = os.path.dirname(os.path.dirname(os.path.dirname(model.__file__)))

    if cfg_file is not None:
        cfg_from_file(os.path.join(model_repo_path, cfg_file))
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)

    test_kwargs = kwargs.pop("TEST", None)
    resnet_kwargs = kwargs.pop("RESNET", None)
    mobilenet_kwargs = kwargs.pop("MOBILENET", None)

    if test_kwargs is not None:
        for key, value in test_kwargs.items():
            cfg["TEST"][key] = value

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
    cfg.SUBMISSION_DIR = os.path.abspath(cfg.SUBMISSION_DIR)
    cfg.TEST_DATA_CLEAN_PATH = os.path.abspath(cfg.TEST_DATA_CLEAN_PATH)
    pprint.pprint(cfg)
    # create output directory
    # output_dir = os.path.join(output_dir, arch, net, dataset)
    output_dir = cfg.SUBMISSION_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg.TRAIN.USE_FLIPPED = False

    imdb, roidb, ratio_list, ratio_index = combined_roidb(imdbval_name, False)
    imdb.competition_mode(on=True)
    imdb.sub_mode = True
    print('{:d} roidb entries'.format(len(roidb)))

    # Trained network weights path
    # input_dir = load_dir + "/" + arch + "/" + net + "/" + dataset
    input_dir = cfg.MODEL_DIR
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             '{}_{}_{}_{}.pth'.format(arch, checksession, checkepoch, checkpoint))

    # Initialize the network:
    if net == 'vgg16':
        # model = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
        print("Pretrained model is not downloaded and network is not used")
    elif net == 'res18':
        model = resnet(imdb.classes, 18, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res34':
        model = resnet(imdb.classes, 34, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res50':
        model = resnet(imdb.classes, 50, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res101':
        model = resnet(imdb.classes, 101, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res152':
        model = resnet(imdb.classes, 152, pretrained=True, class_agnostic=class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    # Create network architecture
    model.create_architecture()

    # Load pre-trained network weights
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    model.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # Initialize the tensor holder
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
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    # Set cuda usage
    if cuda:
        cfg.CUDA = True

    # Copy network to CUDA memroy
    if cuda:
        model.cuda()

    # Start test or evaluation
    start = time.time()
    max_per_image = 100

    # Visualize output bounding boxes
    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = arch + '_' + net
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

    img_dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size,
                                 imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=0,  # args.num_workers
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    # Turn on model evaluation mode, i.e. train=False
    model.eval()

    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    img_dataset.resize_batch()
    for i in range(num_images):

        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = model(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:

            # Apply bounding-box regression deltas

            box_deltas = bbox_pred.data

            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(batch_size, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(batch_size, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        if vis:
            im = image_read_func(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:  # tensor.numel() -> returns number of elements in tensor
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:  # Find any object
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            cv2.imwrite('result.png', im2show)
            pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    print('Kaggle submission file')

    if dataset == 'kaggle_pna':
        cipher = {'rcnn': 'alpha', 'rfcn': 'beta', 'couplenet': 'gamma'}
        created = datetime.now().strftime("%Y%m%d%H%M")
        sub_file = cipher[arch] + '_' + created + '_cls-{}_submission.txt'
        print('Submit file that ends with "_cls-3_submission.txt" file.')
        submission_file = os.path.join(output_dir, sub_file)
        imdb.write_kaggle_submission_file(all_boxes, submission_file, min_conf=min_conf)

    end = time.time()
    print("Deleting irrelevant files...")
    delete_irrelevant_files(cfg.SUBMISSION_DIR)
    print("test time: %0.4fs" % (end - start))


if __name__ == '__main__':
    # load inference config
    inference_config = json.loads(open("PREDICT_SETTINGS.json").read())
    # extract model path
    import model

    model_repo_path = os.path.dirname(os.path.dirname(os.path.dirname(model.__file__)))

    # write files in DCMImagesTest to text file in ImageSets
    data_dir = 'data/PNAdevkit/PNA2018'
    ImageSets_dir = os.path.join(model_repo_path, data_dir, 'ImageSets')

    if not os.path.exists(ImageSets_dir):
        os.makedirs(ImageSets_dir)

    test_files = os.path.join(ImageSets_dir, 'test.txt')
    d = os.path.join(inference_config["TEST_DATA_CLEAN_PATH"])
    pids = [pid.split('.')[0] for pid in os.listdir(d)]
    with open(test_files, 'w') as f:
        for pid in pids:
            f.write("{}\n".format(pid))
    test(**inference_config)
