import numpy as np
import os
import pickle
import shutil
import torch
from path import Path
from argparse import ArgumentParser
from fedlab.utils.dataset.functional import hetero_dir_partition
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import Dataset

current_dir = Path(__file__).parent.absolute()


class CIFAR100Dataset(Dataset):
    def __init__(self, subset) -> None:
        self.data = torch.stack(list(map(lambda tup: tup[0], subset)))
        self.targets = torch.stack(list(map(lambda tup: torch.tensor(tup[1]), subset)))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


def preprocess(args):
    if os.path.isdir(current_dir / "pickles"):
        shutil.rmtree(current_dir / "pickles")
    cifar100_train = CIFAR100(
        current_dir, train=True, transform=transforms.ToTensor(), download=True
    )
    cifar100_test = CIFAR100(
        current_dir, transform=transforms.ToTensor(), train=False
    )

    num_classes = 100

    np.random.seed(args.seed)
    train_idxs = hetero_dir_partition(
        cifar100_train.targets, args.client_num_in_total, num_classes, 0.1
    )

    # Set random seed again is for making sure numpy split trainset and testset in the same way.
    np.random.seed(args.seed)
    test_idxs = hetero_dir_partition(
        cifar100_test.targets, args.client_num_in_total, num_classes, 0.1
    )
    # Now train_idxs[i] and test_idxs[i] have the same classes.

    all_trainsets = []
    all_testsets = []

    for train_indices, test_indices in zip(train_idxs.values(), test_idxs.values()):
        all_trainsets.append(CIFAR100Dataset([cifar100_train[i] for i in train_indices]))
        all_testsets.append(CIFAR100Dataset([cifar100_test[i] for i in test_indices]))
    os.mkdir(current_dir / "pickles")
    # Store clients local trainset and testset as pickles.
    for i in range(args.client_num_in_total):
        with open("{}/pickles/client_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump((all_trainsets[i], all_testsets[i]), file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    preprocess(args)
