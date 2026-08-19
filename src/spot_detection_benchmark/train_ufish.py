import argparse
from ufish.api import UFish


def main():
    parser = argparse.ArgumentParser(description='Train U-Fish model')
    parser.add_argument('--train_dir',
                        type=str,
                        help='Path to training data')
    parser.add_argument('--val_dir',
                        type=str,
                        help='Path to validation data')
    parser.add_argument('--output_dir',
                        type=str,
                        help='Path to save the trained model')

    args = parser.parse_args()

    ufish = UFish()
    ufish.train(
        args.train_dir,
        args.val_dir,
        model_save_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()