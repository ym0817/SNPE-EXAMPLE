import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torchvision.transforms
from PIL import Image
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import imageio
import cv2
from glob import glob
import numpy as np
from torchvision import transforms

COLOR = ["white", "red", "blue", "black"]
TYPE = ["dress", "jeans", "shirt", "shoe", "bag"]


class ImageLoader(Dataset):
    def __init__(self, dir=r"D:\MyNAS\multi_class\data\train_dataset", image_size=448):
        self.image_size = image_size
        self.dir = dir
        self.EXTENSION = ["*.jpg", "*.png", "*.jpeg"]
        self.Read_Imabe_Label()

    def Read_Imabe_Label(self):
        folds = os.listdir(self.dir)
        self.label = []
        self.imageFiles = []
        for foldName in folds:
            temp = foldName.split("_")
            templabel = [COLOR.index(temp[0]), TYPE.index(temp[1])]
            files = []
            for imageType in self.EXTENSION:
                tempfiles = glob(os.path.join(self.dir, foldName, imageType))
                files += tempfiles
            self.label += [templabel] * len(files)
            self.imageFiles += files

    def __len__(self):
        return len(self.imageFiles)

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomScale(self, bgr):
        # 固定住高度，以0.5-1.5伸缩宽度，做缩放
        if random.random() < 0.5:
            scale = random.uniform(0.6, 1.5)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
        return bgr

    def random_flip_horizon(self, img):

        if np.random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip()
            img = transform(img)
        return img

    def random_flip_vertical(self, img):
        """
        随机垂直翻转
        """
        if np.random.random() > 0.5:
            transform = transforms.RandomVerticalFlip()
            img = transform(img)
        return img

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im

    def __getitem__(self, item):
        imgFile = self.imageFiles[item]
        # print(imgFile)
        label = self.label[item]
        data = cv2.imread(imgFile)

        # if random.random() > 0.5:
        #     data = self.random_bright(data)
        #     data = self.randomBlur(data)
        #     data = self.RandomBrightness(data)
        #     data = self.RandomHue(data)
        #     data = self.RandomSaturation(data)
        #     # data = self.random_flip_horizon(data)
        #     # data = self.RandomScale(data)

        data = cv2.resize(data, (self.image_size, self.image_size))
        data = torchvision.transforms.ToTensor()(data)
        gt = np.zeros((1, len(COLOR) + len(TYPE)))
        gt[0, label[0]] = 1
        gt[0, len(COLOR) + label[1]] = 1
        return data, gt, imgFile


if __name__ == '__main__':
    data = DataLoader()
    for a in data:
        pass
