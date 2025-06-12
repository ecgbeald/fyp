# driver code for shallownet
import argparse
from shallow_net.train_bin import train as train_bin
from shallow_net.train_mult import train as train_mult

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/fyp_data",
        help="Path to the dataset directory",
    )
    args = parser.parse_args()
    train_bin(args.dataset_path, save_path="shallow_net/combined.pth")
    train_mult(args.dataset_path, save_path="shallow_net/mult.pth")
