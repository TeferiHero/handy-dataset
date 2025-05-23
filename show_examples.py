import torch
from torch.utils.data import DataLoader

from dataset import Signs2k
from mich_cnn import MichCnn

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MichCnn()
    model.load_state_dict(torch.load("models/MichCnn1_base_lr=0.001_momentum=0.9_net.pth", map_location=device))
    model.to(device)
    model.eval()

    (_, test, _) = Signs2k(root='dataset-small')
    test_loader = DataLoader(test, shuffle=True, batch_size=4)
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)


    def imshow(img):
        img = img.cpu().numpy().transpose((1, 2, 0))  # CHW → HWC
        img = img * 0.229 + 0.485  # Unnormalize (if normalized before)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')


    for i in range(4):
        plt.figure()
        imshow(images[i])
        plt.title(f"Predicted: {preds[i].item()}, True: {labels[i].item()}")
        plt.show()