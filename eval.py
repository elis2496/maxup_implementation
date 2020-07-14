import argparse

import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize
)
from model import init_model
from dataset import Imagenette
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, default='./result/Exp_m_4/2.pth')
parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])


def eval(args):
    model = init_model()
    state_dict = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    transform_test = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = Imagenette(DATAPATH,
                              mode='val',
                              size=IMG_SIZE,
                              transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=NUM_WORKERS,
                                              pin_memory=True)

    all_output = []
    all_gt = []
    with torch.no_grad():
        for img, labels in tqdm(test_loader):
            img = img.to(args.device).float()
            labels = labels.to(args.device)
            output = model(img)
            all_gt.extend(labels.max(1)[1].cpu().numpy().tolist())
            all_output.extend(output.max(1)[1].cpu().numpy().tolist())
    # report = classification_report(all_gt, all_output)
    accuracy = accuracy_score(all_gt, all_output)

    # print(report)
    print("\nTest Accuracy {:.4f}".format(accuracy))


if __name__ == "__main__":
    args = parser.parse_args()
    eval(args)
