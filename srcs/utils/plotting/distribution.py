import matplotlib.pyplot as plt
from typing import List
from utils.parsing.parse import Type


def plot_types(types: List[Type]) -> None:
    fig, ax = plt.subplots(len(types), 2)
    for i, type in enumerate(types):
        sizes = type.labels.values()
        labels = type.labels.keys()
        ax[i, 0].pie(sizes, labels=labels, autopct="%1.1f%%")
        ax[i, 0].set_title(type.name)
        ax[i, 1].bar(labels, sizes)
        ax[i, 1].set_title(type.name)
    fig.canvas.manager.set_window_title("Distribution")
    plt.show()
