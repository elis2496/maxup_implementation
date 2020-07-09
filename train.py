import argparse
import os

import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from albumentations import (
    Compose,
    HorizontalFlip,
    RandomScale
)

from model import init_model
from dataset import Imagenette
from cutmix import CutMix
from maxup_loss import MaxupCrossEntropyLoss
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, default=4,
                    help="Number of augmentations for element in maxup")
parser.add_argument("--device", type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument("--result_path", type=str, default='./result/Exp_m_4')

RESULT_PATH = './result'
os.makedirs(RESULT_PATH, exist_ok=True)


def validation(lossf, valid_loader, model, device):
    model.eval()
    v_loss, acc = 0, 0
    vlen = len(valid_loader)
    all_output = []
    all_gt = []
    with torch.no_grad():
        for img, labels in tqdm(valid_loader):
            img = img.to(device).float()
            labels = labels.to(device)
            output = model(img)
            loss = lossf(output, labels, valid=True)
            v_loss += loss.item()
            all_gt.extend(labels.max(1)[1].cpu().numpy().tolist())
            all_output.extend(output.max(1)[1].cpu().numpy().tolist())
    accuracy = accuracy_score(all_gt, all_output)
    v_loss = v_loss / vlen
    model.train()
    return accuracy, v_loss


def train(args):
    os.makedirs(args.result_path, exist_ok=True)
    transform_train = Compose(
        [
            HorizontalFlip(p=0.3),
            RandomScale(p=0.3)
        ]
    )
    train_dataset = Imagenette(DATAPATH,
                               mode='train',
                               size=IMG_SIZE,
                               transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        CutMix(train_dataset, args.m),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True)
    valid_dataset = Imagenette(DATAPATH,
                               mode='train',
                               size=IMG_SIZE,
                               valid=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=NUM_WORKERS,
                                               pin_memory=True)

    model = init_model()
    model.to(args.device)
    model.train()

    criterion = MaxupCrossEntropyLoss(args.m)
    optimizer = torch.optim.SGD(model.parameters(), LR,
                                momentum=MOMENTUM,
                                weight_decay=1e-4, nesterov=True)
    len_loader = len(train_loader)
    min_loss = 100000000
    iter = 0
    for epoch in range(0, EPOCHS):
        t_loss = 0
        for i, (imgs, labels) in tqdm(enumerate(train_loader)):
            imgs = imgs.reshape((imgs.shape[0] * args.m, 3, IMG_SIZE, IMG_SIZE))
            labels = labels.to(args.device)
            imgs = imgs.to(args.device).float()
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            t_loss += loss.item()
            optimizer.step()
            iter += 1
        val_acc, val_loss = validation(criterion, valid_loader, model, args.device)
        print("\nEpoch {} -> Train Loss: {:.4f}".format(epoch, t_loss / len_loader))
        print("\nEpoch {} -> Valid Loss: {:.4f}\n Valid Accuracy: {:.4f}".format(epoch, val_loss, val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), os.path.join(args.result_path, str(iter) + '.pth'))
            min_loss = val_loss


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
