from mich_cnn import MichCnn
import torch.nn as nn
import torch.optim as optim
import torch
import time
import json
from sklearn.preprocessing import label_binarize
from graph_results import graph_stats
from utils import add_stats_from_validate, create_stats


def train_michcnn(train_loader, validate_loader, name=''):

    PATH_PREFIX = './models'
    MODEL_PATH = f'{PATH_PREFIX}/{name}_net.pth'
    BEST_MODEL_PATH = f'{PATH_PREFIX}/{name}_best_net.pth'
    EPOCHS_NUM = 20
    stats = create_stats()

    best_f1 = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MichCnn().to(device)


    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    start_time = time.time()
    for epoch in range(EPOCHS_NUM):
        running_loss = 0.0
        start_epoch_time = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                elapsed = time.time() - start_time
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}, time: {elapsed:.2f} seconds')

        print(f"Epoch time {(time.time() - start_epoch_time):.2f}")
        stats['train_loss'].append(running_loss)
        add_stats_from_validate(model, validate_loader, device, stats)
        last_f1 = stats['f1'][-1]

        if last_f1 > best_f1:
            print(f'Saving best model... Previous F1 = {best_f1}, This F1 {last_f1}')
            best_f1 = last_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)



    print(f"Total training time: {time.time() - start_time:.2f} s")
    print('Finished Training')

    torch.save(model.state_dict(), MODEL_PATH)
    with open(f"./stats/{name}_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    graph_stats(stats, EPOCHS_NUM, name=name)
