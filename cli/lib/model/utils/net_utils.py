import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from model.utils.config import cfg
from model.roi_crop.functions.roi_crop import RoICropFunction
import cv2
import pdb
import random

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from 
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
    
    return crops, grid

def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid

def _affine_theta(rois, input_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([\
      (y2 - y1) / (height - 1),
      zero,
      (y1 + y2 - height + 1) / (height - 1),
      zero,
      (x2 - x1) / (width - 1),
      (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta

def compare_grid_sample():
    # do gradcheck
    N = random.randint(1, 8)
    C = 2 # random.randint(1, 8)
    H = 5 # random.randint(1, 8)
    W = 4 # random.randint(1, 8)
    input = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
    input_p = input.clone().data.contiguous()
   
    grid = Variable(torch.randn(N, H, W, 2).cuda(), requires_grad=True)
    grid_clone = grid.clone().contiguous()

    out_offcial = F.grid_sample(input, grid)    
    grad_outputs = Variable(torch.rand(out_offcial.size()).cuda())
    grad_outputs_clone = grad_outputs.clone().contiguous()
    grad_inputs = torch.autograd.grad(out_offcial, (input, grid), grad_outputs.contiguous())
    grad_input_off = grad_inputs[0]


    crf = RoICropFunction()
    grid_yx = torch.stack([grid_clone.data[:,:,:,1], grid_clone.data[:,:,:,0]], 3).contiguous().cuda()
    out_stn = crf.forward(input_p, grid_yx)
    grad_inputs = crf.backward(grad_outputs_clone.data)
    grad_input_stn = grad_inputs[0]
    pdb.set_trace()

    delta = (grad_input_off.data - grad_input_stn).sum()


# ------------------
#   AUGMENTATIONS
# ------------------

import cv2
import torch
import random
import numpy as np


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, clip_box, alpha=None):
    """Clip the bounding boxes to the borders of an image.

    bbox: numpy.ndarray. Numpy array containing bounding boxes of shape N X 4 where N is the
        number of bounding boxes and the bounding boxes are represented in the format x1 y1 x2 y2

    clip_box: numpy.ndarray. An array of shape (4,) specifying the diagonal co-ordinates of the image.
        The coordinates are represented in the format x1 y1 x2 y2

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is less than alpha the
        bounding box is dropped.
    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    if alpha:
        delta_area = ((ar_ - bbox_area(bbox)) / ar_)
        mask = (delta_area < (1 - alpha)).astype(int)
        bbox = bbox[mask == 1, :]

    return bbox


def rotate_img(image, angle):
    """
    Rotate the image. Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored black.

    image : numpy.ndarray. Numpy image

    angle : float. Angle by which the image is to be rotated
    """
    # Grab the dimensions of the image and then determine the centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Grab the rotation matrix (applying the negative of the angle to rotate clockwise), then grab
    # the sine and cosine (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    return image


def get_corners(bboxes):
    """
    Get corners of bounding boxes

    bboxes: numpy.ndarray. Numpy array containing bounding boxes of shape N X 4 where N is the
        number of bounding boxes and the bounding boxes are represented in the format x1 y1 x2 y2
    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    return np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.

    corners : numpy.ndarray. Numpy array of shape N x 8 containing N bounding boxes each described
        by their corner co-ordinates x1 y1 x2 y2 x3 y3 x4 y4

    angle : float. Angle by which the image is to be rotated

    cx : int. x coordinate of the center of image (about which the box will be rotated)

    cy : int. y coordinate of the center of image (about which the box will be rotated)

    h : int. Height of the image

    w : int. Width of the image
    """
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def create_enclosing_box(corners):
    """
    Create an enclosing box for rotated corners of a bounding box

    corners : numpy.ndarray. Numpy array of shape N x 8 containing N bounding boxes each described
        by their corner co-ordinates x1 y1 x2 y2 x3 y3 x4 y4
    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the Image with the probability p

    prob: float. The probability with which the image is flipped
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bboxes):
        img = img.copy()
        bboxes = bboxes.copy()
        img_center = np.array(img.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.prob:
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w

        return img, bboxes


class RandomScale(object):
    """
    Randomly scales an image. Bounding boxes which have an area of less than 25%
    in the remaining in the transformed image is dropped. The resolution is maintained,
    and the remaining area if any is filled by black color.

    scale: float or tuple(float). If float, the image is scaled by a factor drawn
        randomly from a range (1 - scale , 1 + scale). If tuple, the scale is drawn
        randomly from values specified by the tuple
    """

    def __init__(self, scale=0.2, diff=False):
        self.scale = scale
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff

    def __call__(self, img, bboxes):

        img_shape = img.shape

        # Chose a random digit to scale by
        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

        bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

        canvas = np.zeros(img_shape, dtype=np.uint8)

        y_lim = int(min(resize_scale_y, 1) * img_shape[0])
        x_lim = int(min(resize_scale_x, 1) * img_shape[1])

        canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

        img = canvas

        bboxes = clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]])
        bboxes = bboxes.clip(min=0)

        return img, bboxes


class RandomTranslate(object):
    """
    Randomly Translates the image. Bounding boxes which have an area of less than 25% in the
    remaining in the transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    translate: float or tuple(float). If float, the image is translated by a factor drawn randomly
        from a range (1 - translate , 1 + translate). If tuple, translate is drawn randomly from values
        specified by the tuple
    """

    def __init__(self, translate=0.2, diff=False):
        self.translate = translate

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1
        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, bboxes):

        img_shape = img.shape

        # Translate the image
        # Percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)

        if not self.diff:
            translate_factor_y = translate_factor_x

        canvas = np.zeros(img_shape).astype(np.uint8)

        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])

        # Change the origin to the top-left corner of the translated box
        orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]),
                          min(img_shape[1], corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]),
               max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]

        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask

        img = canvas

        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

        bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]])
        bboxes = bboxes.clip(min=0)

        return img, bboxes


class RandomRotate(object):
    """
    Randomly rotates an image. Bounding boxes which have an area of less than 25% in the remaining
    in the transformed image is dropped. The resolution is maintained, and the remaining area if any
    is filled by black color.

    angle: float or tuple(float). If float, the image is rotated by a factor drawn randomly from a
        range (-angle, angle). If tuple, the angle is drawn randomly from values specified by the tuple
    """

    def __init__(self, angle=15, dist="cont"):
        self.angle = angle

        if dist == "cont":
            if type(self.angle) is tuple or type(self.angle) is list:
                assert len(self.angle) == 2, "Invalid range"
            else:
                self.angle = (-self.angle, self.angle)
            self.dist_func = lambda x : random.uniform(*x)
        elif dist == "discrete":
            if type(self.angle) is int or type(self.angle) is float:
                self.angle = (self.angle,)
            self.dist_func = random.choice
        else:
            raise ValueError("Angle Distribution {} not available ".format(dist))
    def __call__(self, img, bboxes):

        angle = self.dist_func(self.angle)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        img = rotate_img(img, angle)

        corners = get_corners(bboxes)

        corners = np.hstack((corners, bboxes[:, 4:]))

        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

        new_bbox = create_enclosing_box(corners)

        scale_factor_x = img.shape[1] / w

        scale_factor_y = img.shape[0] / h

        img = cv2.resize(img, (w, h))

        new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        bboxes = new_bbox

        bboxes = clip_box(bboxes, [0, 0, w, h])
        bboxes = bboxes.clip(min=0)

        return img, bboxes


class RandomShear(object):
    """
    Randomly shears an image in horizontal direction. Bounding boxes which have an area of less
    than 25% in the remaining in the transformed image is dropped. The resolution is maintained,
    and the remaining area if any is filled by black color.

    shear_factor: float or tuple(float). If float, the image is sheared horizontally by a factor
        drawn randomly from a range (-shear_factor, shear_factor). If tuple, the shear_factor is
        drawn randomly from values specified by the tuple
    """

    def __init__(self, shear_factor=0.2):
        self.shear_factor = shear_factor

        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)

    def __call__(self, img, bboxes):

        shear_factor = random.uniform(*self.shear_factor)

        w, h = img.shape[1], img.shape[0]

        if shear_factor < 0:
            img, bboxes = RandomHorizontalFlip(prob=1)(img, bboxes)

        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

        nW = img.shape[1] + abs(shear_factor * img.shape[0])

        bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

        if shear_factor < 0:
            img, bboxes = RandomHorizontalFlip(prob=1)(img, bboxes)

        img = cv2.resize(img, (w, h))

        scale_factor_x = nW / w

        bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]
        bboxes = bboxes.clip(min=0)

        return img, bboxes


def apply_augmentations(img_tensors, bbox_tensors, flip_prob=0.5, scale=0.2, scale_prob=0.5, translate=0.2,
                        translate_prob=0.5, angle=20.0, dist="cont", rotate_prob=0.5, shear_factor=0.2, shear_prob=0.5):
    """
    Applies augmentations (horizontal flip, scale, translate, rotate and shear) to image tensors
    and bounding box tensors and returns agumented image tensors and agumented bounding box tensors.

    flip_prob: float, probability of flipping an image and bounding box
    scale: float, the image is scaled by a factor drawn randomly from a range (1 - scale , 1 + scale)
    scale_prob: probability of scaling an image and bounding box
    translate: float, the image is translated by a factor drawn randomly from a range (1 - translate , 1 + translate)
    translate_prob: float, probability of translating an image and bounding box
    angle: float, the image is rotated by a factor drawn randomly from a range (-angle, angle) or from a sequence
    dist: str, specifies the distribution from which the rotation angle is drawn
    rotate_prob: float, probability of rotating an image and bounding box
    shear_factor: float, the image is sheared horizontally by a factor drawn randomly from a range (-shear_factor, shear_factor)
    shear_prob: float, probability of shearing an image and bounding box

    NOTE: set <aug>_prob = 0 if you do not wish to use a augmentation <aug> and set <aug>_prob = 1 if you wish to use
          augmentation <aug> always.
    """
    # Cache for storing augmented numpy arrays
    img_batch = []
    bbox_batch = []

    assert img_tensors.shape[0] == bbox_tensors.shape[0], "batch_size should be same!"

    # Create augmentation instances
    aug_flip = RandomHorizontalFlip(prob=flip_prob)
    if scale_prob > 0:
        aug_scale = RandomScale(scale=scale)
    if translate_prob > 0:
        aug_translate = RandomTranslate(translate=translate)
    if rotate_prob > 0:
        aug_rotate = RandomRotate(angle=angle, dist=dist)
    if shear_prob > 0:
        aug_shear = RandomShear(shear_factor=shear_factor)

    # Loop over all the image tensors
    for i in range(len(img_tensors)):

        # Convert Pytorch Tensor to np.ndarray
        img = img_tensors[i].numpy().copy()
        bbox = bbox_tensors[i,].numpy().copy()

        # Convert: C x W x H -> W x H x C
        img = np.moveaxis(img, 0, -1)

        # Apply augmentations

        # Horizontal flip
        img, bbox = aug_flip(img, bbox)

        # Scale
        if rand() < scale_prob:
            img, bbox = aug_scale(img, bbox)
            bbox[np.where(bbox[:, -1] == 0), :] *= 0  # Fixes NaN problem

        # Translate
        if rand() < translate_prob:
            img, bbox = aug_translate(img, bbox)
            bbox[np.where(bbox[:, -1] == 0), :] *= 0  # Fixes NaN problem

        # Rotate
        if rand() < rotate_prob:
            img, bbox = aug_rotate(img, bbox)
            bbox[np.where(bbox[:, -1] == 0), :] *= 0  # Fixes NaN problem

        # Shear:
        if rand() < shear_prob:
            img, bbox = aug_shear(img, bbox)
            bbox[np.where(bbox[:, -1] == 0), :] *= 0  # Fixes NaN problem

        # Convert: W x H x C -> C x W x H
        img = np.moveaxis(img, -1, 0)

        # Append augmented images and boxes to create batches
        img_batch.append(img)
        bbox_batch.append(bbox)

    # Create batch of agumented image arrays and bounding box arrays
    a_img_arrays = np.stack(img_batch, axis=0)
    a_bbox_arrays = np.stack(bbox_batch, axis=0)

    # Convert np.ndarray to Pytorch tensor
    return torch.from_numpy(a_img_arrays), torch.from_numpy(a_bbox_arrays)