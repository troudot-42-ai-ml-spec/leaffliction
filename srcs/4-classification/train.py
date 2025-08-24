from utils import *
from models import *
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    build_model(args.input_path)


if __name__ == "__main__":
    main()
