from utils import check_dir, split_data, load_datasets, build_model
from utils import train_model  # , plot_metrics
import argparse


def main(args):
    try:
        # --- Check directory validity ---
        check_dir(args.input_path)
        # --- Initial train/test split ---
        train_set, test_set = split_data(args.input_path)
        # --- Rafined split into train/val/test ---
        t_set, v_set, tt_set = load_datasets(train_set, test_set)
        # --- Build model ---
        model = build_model()
        # --- Compile and Train ---
        history = train_model(model, t_set, v_set, tt_set)
        print(history)  # linter for commit
        # --- Plot training metrics ---
        # plot_metrics(history)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the fully augmented and transformed dataset."
        )
    args = parser.parse_args()
    main(args)
