import sys
from train.train import train


def main(argv):
    if len(argv)  == 1:
        env = argv[0]
    else:
        env = "cpu"
    assert env in ["cpu","gpu"]


if __name__ == "__main__":
    main(sys.argv[1:])