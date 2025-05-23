import torch
import matplotlib
import matplotlib.pyplot as plt


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
    
