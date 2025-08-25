# from utils import split_data
from models import build_model
# from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    build_model(args.input_path)


if __name__ == "__main__":
    main()
