from models.networks import define_S
from utils import *
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/***/***')
parser.add_argument('--root', type=str, default='/***/***')

def test():
    img_list = sorted(os.listdir(args.root + 'images'))
    label_list = sorted(os.listdir(args.root + 'labels'))

    dice_list = []
    iou_list = []

    model = define_S(1, 1, 64, 'duseunet', "instance", not True, "xavier", 0.02, False, False, [0])
    model.cuda()
    model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda()))
    model.eval()

    for k in range(len(img_list)):
        print('Processing', k)
        img_ori = cv2.imread(args.root + 'images/' + img_list[k], 0)
        img_label = cv2.imread(args.root + 'labels/' + label_list[k], 0)

        h, w = img_ori.shape
        img_mask = np.zeros([h, w])

        img_ori_pad = cv2.copyMakeBorder(img_ori, 64, 64, 64, 64, cv2.BORDER_REFLECT101)

        for i in range(h // 128):
            for j in range(w // 128):
                img_patch0 = img_ori_pad[128 * i:128 * i + 256, 128 * j:128 * j + 256].astype(np.uint8)
                img_patch = img_patch0.copy()
                img_patch = test_transforms(img_patch)
                img_patch = img_patch.reshape(1, 1, 256, 256)
                img_patch = img_patch.cuda()
                seg_real_B = model.forward(img_patch)
                seg_real_B = seg_real_B.data[:, -1:, :, :]
                seg_real_B[seg_real_B >= 0.5] = 1
                seg_real_B[seg_real_B < 0.5] = 0
                seg_real_B = tensor2seg_test(seg_real_B)
                seg_real_B = seg_real_B.squeeze()

                img_mask[128 * i:128 * i + 128, 128 * j:128 * j + 128] = np.array(seg_real_B)[64:128 + 64, 64:128 + 64]

        img_mask = img_mask.astype(np.uint8)

        cv2.imwrite('./result/%03d.png'%k, img_mask)

        dice = compute_ei(img_mask, img_label)[0]
        iou = compute_ei(img_mask, img_label)[1]
        dice_list.append(dice)
        iou_list.append(iou)

        print('DICE', dice, 'IOU', iou)

    print('DICE_AVE', sum(dice_list) / len(dice_list), 'IOU_AVE', sum(iou_list) / len(iou_list))


if __name__ == "__main__":
    args = parser.parse_args()
    test()
