import argparse
from pathlib import Path
from utils import __all__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        help="The path to the image you are trying to augment,\
             or directory you are trying to balance.",
    )
    args = parser.parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: The path '{input_path}' does not exist.")
        return
    try:
        if input_path.is_file():
            __all__.parse_file(input_path)
            __all__.process_file(input_path)
        elif input_path.is_dir():
            new_path = __all__.parse_dir(input_path)
            __all__.process_dir(new_path)
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
