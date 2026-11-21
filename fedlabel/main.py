import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from argparse import ArgumentParser
from fedlabel.config import get_args
from fedlabel.runner import FedLabelRunner

# Pickles written by an older preprocess.py run as a script store their dataset class
# as __main__.<Name>. Importing the classes here (this file is __main__) lets those old
# pickles load without any notebook patch. New pickles do not need this.
from data.cifar import CIFARDataset  # noqa: F401
from data.mnist import MNISTDataset  # noqa: F401
from data.synthetic import SyntheticDataset  # noqa: F401
from data.emnist import EMNISTDataset  # noqa: F401
from data.cifar100 import CIFAR100Dataset  # noqa: F401
from data.organamnist import OrganAMNISTDataset  # noqa: F401
from data.bloodmnist import BloodMNISTDataset  # noqa: F401
from data.pathamnist import PathMNISTDataset  # noqa: F401

def main():
    parser = ArgumentParser(description="FedLabel Runner")
    args = get_args(parser)
    runner = FedLabelRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
