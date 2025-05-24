import torch
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.preprocessing import label_binarize


def add_stats_from_validate(model, validate_loader, device, stats):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in validate_loader:
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
    stats['confusion_matrix'].append(mt.confusion_matrix(all_labels, all_preds).tolist())

    NUM_CLASSES = 7
    all_labels_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))
    all_preds_bin = label_binarize(all_preds, classes=range(NUM_CLASSES))

    stats['roc_auc_micro'].append(mt.roc_auc_score(all_labels_bin, all_preds_bin, multi_class='ovr', average='micro'))
    stats['roc_auc_macro'].append(mt.roc_auc_score(all_labels_bin, all_preds_bin, multi_class='ovr', average='macro'))
    print(stats)
    model.train()


def create_stats():
    stats = {
        'train_loss': [],
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'r2': [],
        'confusion_matrix': [],
        'roc_auc_micro': [],
        'roc_auc_macro': []
    }
    return stats

def preview_dataset(dataset):
    labels_map = {
        0: "Nieznane",
        1: "B",
        2: "M",
        3: "N",
        4: "O",
        5: "S",
        6: "T",
    }
    
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 4, 4
    
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        # print(type(img))
        plt.imshow(img, cmap="gray")

    plt.show()
    
