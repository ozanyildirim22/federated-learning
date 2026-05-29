import numpy as np
import os
import pickle
import shutil
import torch
from path import Path
from argparse import ArgumentParser
from fedlab.utils.dataset.functional import hetero_dir_partition
from torchvision import transforms
from torch.utils.data import Dataset
import medmnist
from medmnist import OrganAMNIST

current_dir = Path(__file__).parent.absolute()


class OrganAMNISTDataset(Dataset):
    def __init__(self, subset) -> None:
        self.data = torch.stack(list(map(lambda tup: tup[0], subset)))
        self.targets = torch.stack(list(map(lambda tup: torch.tensor(tup[1]), subset))).squeeze()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


def preprocess(args):
    if os.path.isdir(current_dir / "pickles"):
        shutil.rmtree(current_dir / "pickles")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])

    # OrganAMNIST: 11 classes, 3-channel 28x28 (cross-silo)
    train_dataset = OrganAMNIST(
        root=current_dir, split="train", transform=transform, download=True
    )
    test_dataset = OrganAMNIST(
        root=current_dir, split="test", transform=transform, download=True
    )

    num_classes = 11

    # MedMNIST targets are shape (N, 1), squeeze to (N,)
    train_targets = train_dataset.labels.squeeze().tolist()
    test_targets = test_dataset.labels.squeeze().tolist()

    np.random.seed(args.seed)
    train_idxs = hetero_dir_partition(
        train_targets, args.client_num_in_total, num_classes, 0.1
    )

    np.random.seed(args.seed)
    test_idxs = hetero_dir_partition(
        test_targets, args.client_num_in_total, num_classes, 0.1
    )

    all_trainsets = []
    all_testsets = []

    for train_indices, test_indices in zip(train_idxs.values(), test_idxs.values()):
        # Simulate label scarcity: only keep label_ratio fraction of training data
        if args.label_ratio < 1.0:
            n_keep = max(1, int(len(train_indices) * args.label_ratio))
            train_indices = np.random.choice(train_indices, n_keep, replace=False).tolist()
        all_trainsets.append(OrganAMNISTDataset([train_dataset[i] for i in train_indices]))
        all_testsets.append(OrganAMNISTDataset([test_dataset[i] for i in test_indices]))
    os.mkdir(current_dir / "pickles")
    # Store clients local trainset and testset as pickles.
    for i in range(args.client_num_in_total):
        with open("{}/pickles/client_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump((all_trainsets[i], all_testsets[i]), file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=10)
    parser.add_argument("--label_ratio", type=float, default=0.05, help="Fraction of labeled training data per client (cross-silo: 0.05)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    preprocess(args)
