import torchvision.models as models

from torch import nn


def init_model(num_classes=10):
    model = models.resnet34(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Linear(in_features=512, out_features=num_classes, bias=True),
    )
    return model
