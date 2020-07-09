import cv2
import os
import glob
import torch
import json

from torch.utils.data import Dataset


class Imagenette(Dataset):
    def __init__(self,
                 root,
                 mode='train',
                 size=320,
                 valid=False,
                 transform=None):
        img_path = os.path.join(root, mode)
        classes_path = glob.glob(''.join([img_path, '/*']))
        with open(os.path.join(root, 'label2Idx.json'), 'r') as file:
            label2Idx = json.load(file)
        if valid is True:
            self.img_data = [[os.path.join(class_path, f), label2Idx[class_path.split('/')[-1]]]
                             for class_path in classes_path
                             for f in os.listdir(class_path) if 'val' in f]
        else:
            self.img_data = [[os.path.join(class_path, f), i] for i, class_path in enumerate(classes_path)
                             for f in os.listdir(class_path) if 'val' not in f]
        self.num_classes = len(label2Idx)
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        img_file, label = self.img_data[index]
        img = cv2.imread(img_file)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = cv2.resize(img, (self.size, self.size)) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        label_onehot[label] = 1.
        return img, label_onehot
