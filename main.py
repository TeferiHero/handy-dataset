from torch.utils.data import DataLoader
from dataset import Signs2k
from train_net import train_michcnn
from utils import preview_dataset
import torch.optim as optim
import sklearn.metrics as metrics
import sys
import time

def main():
    NAME = 'TestCNN_metrics'
    timestamp = time.time()
    formatted = time.strftime("%Y-%m-%d_%H-%M_%S", time.localtime(timestamp))
    log_file = open(f"./logs/log_{formatted}___{NAME}.txt", "w")
    # sys.stdout = log_file
    (train, test, validate) = Signs2k(root='dataset-small')
    train_dataloader = DataLoader(train, shuffle=True)
    test_loader = DataLoader(test)
    # preview_dataset(train)
    print(NAME)
    train_michcnn(train_dataloader, test_loader, name=NAME)



if __name__ == "__main__":
    main()
