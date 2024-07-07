import torch.utils.data as data
import os
import cv2
from torchvision.transforms import transforms


def make_dataset(root):
    imgs = []
    n = len(os.listdir(root)) // 2
    for i in range(1, n + 1):
        img = os.path.join(root, "%03d.png" % i)
        imgs.append((img, img))
    return imgs


class Dataset(data.Dataset):
    def __init__(self, root, co_transform=None, transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.co_transform = co_transform
        self.transform = transform
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = cv2.imread(x_path, 0)
        img_y = cv2.imread(y_path, 0)
        img_x = cv2.copyMakeBorder(img_x, 32, 32, 32, 32, cv2.BORDER_REFLECT101)
        img_y = cv2.copyMakeBorder(img_y, 32, 32, 32, 32, cv2.BORDER_REFLECT101)

        augmented = self.co_transform(image=img_x, mask=img_y)

        img_x = augmented['image']
        img_y = augmented['mask']

        img_x = transforms.ToPILImage()(img_x)
        img_y = transforms.ToPILImage()(img_y)

        img_y = self.jitter(img_y)
        img_x = self.transform(img_x)
        img_y = self.transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
