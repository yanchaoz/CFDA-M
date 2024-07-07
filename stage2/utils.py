import numpy as np
from torchvision.transforms import transforms
import torch
from albumentations.augmentations import transforms as at
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, RandomCrop

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

co_trnsform = Compose([
    RandomRotate90(),
    at.Flip(),
    RandomCrop(256, 256)
])

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def Cutout(imgs,labels, n_holes=1, length=32):

    h = imgs.shape[2]
    w = imgs.shape[3]
    num = imgs.shape[0]
    labels_list = []
    imgs_list = []
    masks_list = []

    for i in range(num):
        label = labels[i,:,:,:]
        img = imgs[i,:,:,:]
        mask = np.ones((1, h, w), np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.cuda()

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[0, y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask
        label = label * mask
        imgs_list.append(img)
        labels_list.append(label)
        masks_list.append(mask)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    masks_out = torch.stack(masks_list)

    return imgs_out, labels_out, masks_out



def compute_ei(pred, target):
    smooth = 1.
    pred = pred / 255
    target = target / 255
    num = pred.size
    m1 = pred.reshape(num, 1)  # Flatten
    m2 = target.reshape(num, 1)  # Flatten

    intersection = (m1 * m2).sum()
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    iou = dice / (2 - dice)
    return dice, iou


def tensor2seg_test(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255
    return image_numpy.astype(imtype)


def transforms_for_rot(ema_inputs):
    rot_mask = np.random.randint(0, 4, ema_inputs.shape[0])
    flip_mask = np.random.randint(0, 2, ema_inputs.shape[0])

    for idx in range(ema_inputs.shape[0]):
        if flip_mask[idx] == 1:
            ema_inputs[idx] = torch.flip(ema_inputs[idx], [1])

        ema_inputs[idx] = torch.rot90(ema_inputs[idx], int(rot_mask[idx]), dims=[1, 2])

    return ema_inputs, rot_mask, flip_mask


def transforms_back_rot(ema_output, rot_mask, flip_mask):
    for idx in range(ema_output.shape[0]):

        ema_output[idx] = torch.rot90(ema_output[idx], int(rot_mask[idx]), dims=[2, 1])

        if flip_mask[idx] == 1:
            ema_output[idx] = torch.flip(ema_output[idx], [1])

    return ema_output


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    return ema_model

def update_same_variables(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(0.).add_(1 , param.data)
    return ema_model


def get_current_consistency_weight(epoch):
    return 0.1 * sigmoid_rampup(epoch, 40)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
