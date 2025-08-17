import matplotlib.pyplot as plt
import argparse

from typing import List

from config import Config
from utils.parse import parse, Type


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program which will show distrubution of a dataset given."
    )
    parser.add_argument("path", type=str, help="The dataset path.")
    args = parser.parse_args()

    cfg: Config = Config()
    types: List[Type] = parse(args, cfg)
    plot_types(types)
