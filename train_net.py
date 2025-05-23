from mich_cnn import MichCnn
import torch.nn as nn
import torch.optim as optim
import torch
import time
import sklearn.metrics as mt
import json

from graph_results import graph_stats

def print_stats_test(model, test_loader, device, stats):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    stats['accuracy'].append(mt.accuracy_score(all_labels, all_preds))
    stats['precision'].append(mt.precision_score(all_labels, all_preds, average='weighted'))
    stats['recall'].append(mt.recall_score(all_labels, all_preds, average='weighted'))
    stats['f1'].append(mt.f1_score(all_labels, all_preds, average='weighted'))
    stats['r2'].append(mt.r2_score(all_labels, all_preds))
    print(stats)
    model.train()


def train_michcnn(train_loader, test_loader, name=''):

    EPOCHS_NUM = 20
    stats = {
        'train_loss': [],
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'r2': []
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MichCnn().to(device)


    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
    start_time = time.time()
    for epoch in range(EPOCHS_NUM):
        running_loss = 0.0
        start_epoch_time = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                elapsed = time.time() - start_time
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}, time: {elapsed:.2f} seconds')

        print(f"Epoch time {(time.time() - start_epoch_time):.2f}")
        stats['train_loss'].append(running_loss)
        print_stats_test(model, test_loader, device, stats)


    print(f"Total training time: {time.time() - start_time:.2f} s")
    print('Finished Training')
    PATH = f'./models/{name}_net.pth'
    torch.save(model.state_dict(), PATH)
    with open(f"./stats/{name}_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    graph_stats(stats, EPOCHS_NUM, name=name)
