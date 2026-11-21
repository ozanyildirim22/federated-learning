import numpy as np
import os
import pickle
import shutil
import sys
import torch
from path import Path
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import Dataset

current_dir = Path(__file__).parent.absolute()


class OrganAMNISTDataset(Dataset):
    def __init__(self, subset) -> None:
        self.data = torch.stack(list(map(lambda tup: tup[0], subset)))
        self.targets = torch.stack(list(map(lambda tup: torch.tensor(tup[1]), subset))).flatten()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


def pickle_dir_name(alpha, label_ratio, split="official", seed=0):
    """Directory holding one client_<i>.pkl per client.
    'official': MedMNIST's own train/test split (train partitioned, full test kept).
    'paper'   : FedLabel arXiv appendix split, 80% train / 5% val / 15% test of the
                pooled train+val+test images; the seed is in the name because the paper
                averages over 3 partition seeds."""
    if split == "paper":
        return f"pickles_paper_alpha{alpha}_label{label_ratio}_seed{seed}"
    return f"pickles_alpha{alpha}_label{label_ratio}"


def _transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])


def _load(split):
    from medmnist import OrganAMNIST
    return OrganAMNIST(root=current_dir, split=split, transform=_transform(), download=True)


def split_labeled(train_indices, label_ratio):
    """Uniformly random labeled/unlabeled split of one client's training indices."""
    train_indices = np.array(train_indices)
    np.random.shuffle(train_indices)
    n_labeled = max(1, int(len(train_indices) * label_ratio))
    return train_indices[:n_labeled].tolist(), train_indices[n_labeled:].tolist()


def _write_clients(pickle_dir, args, train_pool, test_pool, targets, get_item):
    """Dirichlet-partition train_pool (and test_pool, so every client keeps a local test
    split whose union is exactly test_pool) and write client_<i>.pkl files."""
    from fedlab.utils.dataset.functional import hetero_dir_partition

    np.random.seed(args.seed)
    train_parts = hetero_dir_partition(
        targets[train_pool].tolist(), args.client_num_in_total, args.classes, args.alpha
    )
    np.random.seed(args.seed)
    test_parts = hetero_dir_partition(
        targets[test_pool].tolist(), args.client_num_in_total, args.classes, args.alpha
    )

    os.mkdir(pickle_dir)
    for cid in range(args.client_num_in_total):
        client_train = train_pool[np.array(train_parts[cid], dtype=int)]
        client_test = test_pool[np.array(test_parts[cid], dtype=int)]
        labeled_idx, unlabeled_idx = split_labeled(client_train, args.label_ratio)

        labeled_set = OrganAMNISTDataset([get_item(i) for i in labeled_idx])
        unlabeled_set = OrganAMNISTDataset([get_item(i) for i in unlabeled_idx])
        test_set = OrganAMNISTDataset([get_item(i) for i in client_test])
        print(f"client {cid}: labeled {len(labeled_set)} / unlabeled {len(unlabeled_set)} / test {len(test_set)}")
        with open("{}/client_{}.pkl".format(pickle_dir, cid), "wb") as file:
            pickle.dump((labeled_set, unlabeled_set, test_set), file)


def preprocess_paper(args):
    """
    FedLabel (arXiv 2307.08809, Appendix A): "For all datasets, 80% is for training
    partitioned across clients and the 5%, 15% of the data is for the validation and
    test respectively." OrganAMNIST train+val+test are pooled (58,830 images), shuffled
    with --seed and cut into 80% train / 5% val / 15% test. Train is Dirichlet-
    partitioned across the clients and split into labeled/unlabeled per client.
    """
    name = pickle_dir_name(args.alpha, args.label_ratio, "paper", args.seed)
    pickle_dir = current_dir / name
    if os.path.isdir(pickle_dir):
        shutil.rmtree(pickle_dir)

    parts = [_load("train"), _load("val"), _load("test")]
    offsets = np.cumsum([0] + [len(p) for p in parts])
    targets = np.concatenate([p.labels.flatten() for p in parts]).astype(int)

    def get_item(i):
        k = int(np.searchsorted(offsets, i, side="right") - 1)
        return parts[k][i - offsets[k]]

    np.random.seed(args.seed)
    perm = np.random.permutation(len(targets))
    n_tr = int(0.80 * len(perm))
    n_val = int(0.05 * len(perm))
    train_pool, val_pool, test_pool = perm[:n_tr], perm[n_tr:n_tr + n_val], perm[n_tr + n_val:]
    print(f"Pooled split: train {len(train_pool)} / val {len(val_pool)} / test {len(test_pool)}")

    _write_clients(pickle_dir, args, train_pool, test_pool, targets, get_item)

    # Validation set lives next to (not inside) the client dir, so client counting stays correct.
    val_set = OrganAMNISTDataset([get_item(i) for i in val_pool])
    with open(current_dir / f"{name}_val.pkl", "wb") as file:
        pickle.dump(val_set, file)


def preprocess(args):
    if getattr(args, "split", "official") == "paper":
        return preprocess_paper(args)

    pickle_dir = current_dir / pickle_dir_name(args.alpha, args.label_ratio)
    if os.path.isdir(pickle_dir):
        shutil.rmtree(pickle_dir)

    train_dataset, test_dataset = _load("train"), _load("test")
    n_train = len(train_dataset)
    targets = np.concatenate([train_dataset.labels.flatten(), test_dataset.labels.flatten()]).astype(int)

    def get_item(i):
        return train_dataset[i] if i < n_train else test_dataset[i - n_train]

    train_pool = np.arange(n_train)
    test_pool = np.arange(n_train, len(targets))
    _write_clients(pickle_dir, args, train_pool, test_pool, targets, get_item)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=10)
    parser.add_argument("--classes", type=int, default=11)          # fixed
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--label_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="official", choices=["official", "paper"],
                        help="official = MedMNIST train/test; paper = pooled 80/5/15 (FedLabel appendix)")
    return parser.parse_args()


if __name__ == "__main__":
    # Run through the package module (not __main__) so the pickled dataset class is
    # data.organamnist.preprocess.OrganAMNISTDataset and loads from any entry point.
    sys.path.insert(0, str(current_dir.parent.parent))
    from data.organamnist import preprocess as _module
    _module.preprocess(get_args())
