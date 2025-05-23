import os
import torchvision
from torchvision.transforms import v2
import torch

def Signs2k(root):
    transforms = v2.Compose([

        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # v2.ToTensor(),
        # v2.Resize(size=(256, 144)),
        # v2.RandomInvert(p=0.5),
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomVerticalFlip(p=0.5),
        # v2.RandomRotation(degrees=360),
        # v2.RandomPerspective(),
        # v2.ElasticTransform(),
        v2.Resize(size=(256, 144)),
        # v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    train = torchvision.datasets.ImageFolder(os.path.join(root, "train"),
                                             transform=transforms)

    validate = torchvision.datasets.ImageFolder(os.path.join(root, "train"),
                                                transform=transforms)

    test = torchvision.datasets.ImageFolder(os.path.join(root, "test"),
                                            transform=transforms)

    return (train, validate, test)

