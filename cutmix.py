import numpy as np
import torch
import random

from torch.utils.data.dataset import Dataset


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix(Dataset):
    def __init__(self, dataset, m=1, beta=1.0):
        self.dataset = dataset
        self.m = m
        self.beta = beta

    def __getitem__(self, index):
        m_imgs = []
        m_lbls = []
        for _ in range(self.m):
            img, lbl1 = self.dataset[index]

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lbl2 = self.dataset[rand_index]

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lbl = lbl1 * lam + lbl2 * (1. - lam)
            m_imgs.append(img)
            m_lbls.append(lbl)
        m_imgs = torch.stack(m_imgs, 0)
        m_lbls = torch.stack(m_lbls, 0)
        return m_imgs, m_lbls

    def __len__(self):
        return len(self.dataset)
