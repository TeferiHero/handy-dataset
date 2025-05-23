import json
import os

import matplotlib.pyplot as plt

def graph_stats(stats, epochs_num, name=''):


    os.makedirs(f"./graphs/{name}", exist_ok=True)
    epochs = range(1, epochs_num + 1)
    # Plot each metric
    for metric_name, values in stats.items():
        plt.plot(epochs, values, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"Training Metrics: {metric_name} for {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./graphs/{name}/plot_{metric_name}.png")
        plt.show()


if __name__ == '__main__':
    name = 'MichCnn1_base_lr=0.01_momentum=0.9'

    with open(f"./stats/{name}_stats.json", "r") as f:
        stats = json.load(f)

    epochs_num = len(stats["accuracy"])
    if epochs_num == 0:
        print("No accuracy")
    graph_stats(stats, epochs_num, name)