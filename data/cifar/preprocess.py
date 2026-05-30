import numpy as np
import os
import pickle
import shutil
import torch
from path import Path
from argparse import ArgumentParser
from fedlab.utils.dataset.functional import hetero_dir_partition
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset

current_dir = Path(__file__).parent.absolute()


class CIFARDataset(Dataset):
    def __init__(self, subset) -> None:
        self.data = torch.stack(list(map(lambda tup: tup[0], subset)))
        self.targets = torch.stack(list(map(lambda tup: torch.tensor(tup[1]), subset)))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


def preprocess(args):
    pickle_dir = current_dir / f"pickles_alpha{args.alpha}_label{args.label_ratio}"
    if os.path.isdir(pickle_dir):
        shutil.rmtree(pickle_dir)

    cifar_train = CIFAR10(
        current_dir, train=True, transform=transforms.ToTensor(), download=True
    )
    cifar_test = CIFAR10(
        current_dir, train=False, transform=transforms.ToTensor()
    )

    np.random.seed(args.seed)
    train_idxs = hetero_dir_partition(
        cifar_train.targets, args.client_num_in_total, args.classes, args.alpha
    )
    np.random.seed(args.seed)
    test_idxs = hetero_dir_partition(
        cifar_test.targets, args.client_num_in_total, args.classes, args.alpha
    )

    all_trainsets = []
    all_testsets = []

    for train_indices, test_indices in zip(train_idxs.values(), test_idxs.values()):
        train_indices = np.array(train_indices)
        np.random.shuffle(train_indices)

        # Split into labeled and unlabeled
        n_labeled = max(1, int(len(train_indices) * args.label_ratio))
        labeled_indices = train_indices[:n_labeled].tolist()
        unlabeled_indices = train_indices[n_labeled:].tolist()

        labeled_set = CIFARDataset([cifar_train[i] for i in labeled_indices])
        unlabeled_set = CIFARDataset([cifar_train[i] for i in unlabeled_indices])
        test_set = CIFARDataset([cifar_test[i] for i in test_indices])

        all_trainsets.append((labeled_set, unlabeled_set))
        all_testsets.append(test_set)

    os.mkdir(pickle_dir)
    for i in range(args.client_num_in_total):
        with open("{}/client_{}.pkl".format(pickle_dir, i), "wb") as file:
            pickle.dump((all_trainsets[i][0],    # labeled
                         all_trainsets[i][1],    # unlabeled
                         all_testsets[i]), file) # test


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=100)
    parser.add_argument("--classes", type=int, default=10)          # fixed
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--label_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    preprocess(args)