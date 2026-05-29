import numpy as np
import os
import pickle
import shutil
import torch
from path import Path
from argparse import ArgumentParser
from fedlab.utils.dataset.functional import hetero_dir_partition
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import Dataset

current_dir = Path(__file__).parent.absolute()


class EMNISTDataset(Dataset):
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

    # EMNIST byclass: 62 classes (digits 0-9, uppercase A-Z, lowercase a-z)
    # Note: torchvision EMNIST images are transposed, so we apply a lambda to fix orientation
    emnist_transform = transforms.Compose([
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.ToTensor(),
    ])

    emnist_train = EMNIST(
        current_dir, split="byclass", train=True, transform=emnist_transform, download=True
    )
    emnist_test = EMNIST(
        current_dir, split="byclass", train=False, transform=emnist_transform
    )

    num_classes = 62

    np.random.seed(args.seed)
    train_idxs = hetero_dir_partition(
        emnist_train.targets.numpy().tolist(), args.client_num_in_total, num_classes, 0.1
    )

    # Set random seed again is for making sure numpy split trainset and testset in the same way.
    np.random.seed(args.seed)
    test_idxs = hetero_dir_partition(
        emnist_test.targets.numpy().tolist(), args.client_num_in_total, num_classes, 0.1
    )
    # Now train_idxs[i] and test_idxs[i] have the same classes.

    all_trainsets = []
    all_testsets = []

    for train_indices, test_indices in zip(train_idxs.values(), test_idxs.values()):
        # Simulate label scarcity: only keep label_ratio fraction of training data
        if args.label_ratio < 1.0:
            n_keep = max(1, int(len(train_indices) * args.label_ratio))
            train_indices = np.random.choice(train_indices, n_keep, replace=False).tolist()
        all_trainsets.append(EMNISTDataset([emnist_train[i] for i in train_indices]))
        all_testsets.append(EMNISTDataset([emnist_test[i] for i in test_indices]))
    os.mkdir(current_dir / "pickles")
    # Store clients local trainset and testset as pickles.
    for i in range(args.client_num_in_total):
        with open("{}/pickles/client_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump((all_trainsets[i], all_testsets[i]), file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=100)
    parser.add_argument("--label_ratio", type=float, default=0.2, help="Fraction of labeled training data per client (cross-device: 0.2)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    preprocess(args)
