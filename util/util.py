from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections

from skimage.draw import circle, line_aa, polygon

import pickle

#copy from pose transfer (zhu et al.)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, need_dec=False, imtype=np.uint8):
    if len(image_tensor.shape)==4:
        image_numpy = image_tensor[0].cpu().float().numpy()
    else:
        image_numpy = image_tensor.cpu().float().numpy()
    
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if need_dec:
#        image_numpy = torch.argmax(image_numpy, 2)
        image_numpy = decode_labels(image_numpy.astype(int))
    else:
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
    
    return image_numpy.astype(imtype)
#label color
label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

# draw pose img
LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1

def map_to_cord(pose_map, threshold=0.1):
    all_peaks = [[] for i in range(18)]
    pose_map = pose_map[..., :18]

    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)),
                                     pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(18):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)

def draw_pose_from_map(pose_map, threshold=0.1, **kwargs):
    # CHW -> HCW -> HWC
    pose_map = pose_map[0].cpu().transpose(1, 0).transpose(2, 1).numpy()

    cords = map_to_cord(pose_map, threshold=threshold)
    return draw_pose_from_cords(cords, pose_map.shape[:2], **kwargs)


# draw pose from map
def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def morph(src_bg_mask, ks, mode='erode', kernel=None):
    n_ks = ks ** 2
    pad_s = ks // 2

    if kernel is None:
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32, device=src_bg_mask.device)

    if mode == 'erode':
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=1.0)
        out = F.conv2d(src_bg_mask_pad, kernel)
        out = (out == n_ks).float()
    else:
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=0.0)
        out = F.conv2d(src_bg_mask_pad, kernel)
        out = (out >= 1).float()

    return out

def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=2)

def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    h, w, c = mask.shape
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros(( h, w, 3), dtype=np.uint8)

    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    tmp = []
    tmp1 = []
    for j_, j in enumerate(mask[:, :, 0]):
        for k_, k in enumerate(j):
            #tmp1.append(k)
            #tmp.append(k)
            if k < num_classes:
                pixels[k_,j_] = label_colours[k]
    #np.save('tmp1.npy', tmp1)
    #np.save('tmp.npy',tmp)
    outputs = np.array(img)
    #print(outputs[144,:,0])
    return outputs

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images
