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


def pickle_dir_name(alpha, label_ratio, split="official", seed=0):
    """Directory holding one client_<i>.pkl per client.
    'official': CIFAR-10's own 50k/10k train/test split (original behaviour).
    'paper'   : FedLabel arXiv appendix split, 80% train / 5% val / 15% test of the
                pooled 60k images; the seed is in the name because the paper averages
                over 3 partition seeds."""
    if split == "paper":
        return f"pickles_paper_alpha{alpha}_label{label_ratio}_seed{seed}"
    return f"pickles_alpha{alpha}_label{label_ratio}"


def split_labeled(train_indices, label_ratio):
    """Uniformly random labeled/unlabeled split of one client's training indices."""
    train_indices = np.array(train_indices)
    np.random.shuffle(train_indices)
    n_labeled = max(1, int(len(train_indices) * label_ratio))
    return train_indices[:n_labeled].tolist(), train_indices[n_labeled:].tolist()


def preprocess_paper(args):
    """
    FedLabel (arXiv 2307.08809, Appendix A): "80% is for training partitioned across
    clients and the 5%, 15% of the data is for the validation and test respectively."
    CIFAR-10 train+test are pooled (60k), shuffled with --seed, and cut into
    48k train / 3k val / 9k test. Train is Dirichlet-partitioned across clients and
    split into labeled/unlabeled per client. The 9k test set is also Dirichlet-
    partitioned so every client keeps a local test split; the union of those splits
    is exactly the 9k global test set the runner scores on. Val is saved separately.
    """
    pickle_dir = current_dir / pickle_dir_name(args.alpha, args.label_ratio, "paper", args.seed)
    if os.path.isdir(pickle_dir):
        shutil.rmtree(pickle_dir)

    cifar_train = CIFAR10(current_dir, train=True, transform=transforms.ToTensor(), download=True)
    cifar_test = CIFAR10(current_dir, train=False, transform=transforms.ToTensor())
    n_train_orig = len(cifar_train)
    pooled_targets = np.array(list(cifar_train.targets) + list(cifar_test.targets))

    def get_item(i):
        return cifar_train[i] if i < n_train_orig else cifar_test[i - n_train_orig]

    np.random.seed(args.seed)
    perm = np.random.permutation(len(pooled_targets))
    n_total = len(perm)
    n_tr = int(0.80 * n_total)
    n_val = int(0.05 * n_total)
    train_pool, val_pool, test_pool = perm[:n_tr], perm[n_tr:n_tr + n_val], perm[n_tr + n_val:]
    print(f"Pooled split: train {len(train_pool)} / val {len(val_pool)} / test {len(test_pool)}")

    # hetero_dir_partition returns positions into the list it is given; map them back.
    np.random.seed(args.seed)
    train_parts = hetero_dir_partition(
        pooled_targets[train_pool].tolist(), args.client_num_in_total, args.classes, args.alpha
    )
    np.random.seed(args.seed)
    test_parts = hetero_dir_partition(
        pooled_targets[test_pool].tolist(), args.client_num_in_total, args.classes, args.alpha
    )

    os.mkdir(pickle_dir)
    for cid in range(args.client_num_in_total):
        client_train = train_pool[np.array(train_parts[cid], dtype=int)]
        client_test = test_pool[np.array(test_parts[cid], dtype=int)]
        labeled_idx, unlabeled_idx = split_labeled(client_train, args.label_ratio)

        labeled_set = CIFARDataset([get_item(i) for i in labeled_idx])
        unlabeled_set = CIFARDataset([get_item(i) for i in unlabeled_idx])
        test_set = CIFARDataset([get_item(i) for i in client_test])
        with open("{}/client_{}.pkl".format(pickle_dir, cid), "wb") as file:
            pickle.dump((labeled_set, unlabeled_set, test_set), file)

    # Validation set lives next to (not inside) the client dir, so client counting stays correct.
    val_set = CIFARDataset([get_item(i) for i in val_pool])
    with open(current_dir / f"{pickle_dir_name(args.alpha, args.label_ratio, 'paper', args.seed)}_val.pkl", "wb") as file:
        pickle.dump(val_set, file)


def preprocess(args):
    if getattr(args, "split", "official") == "paper":
        return preprocess_paper(args)

    pickle_dir = current_dir / pickle_dir_name(args.alpha, args.label_ratio)
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
    parser.add_argument("--split", type=str, default="official", choices=["official", "paper"],
                        help="official: CIFAR-10 50k/10k split; paper: pooled 80/5/15 split (FedLabel appendix)")
    args = parser.parse_args()
    preprocess(args)