import os
import pickle
from path import Path
from torch.utils.data import DataLoader
from .preprocess import OrganAMNISTDataset, pickle_dir_name

current_dir = Path(__file__).parent.absolute()


def organamnist_pickle_dir(alpha=0.1, label_ratio=0.05, split="official", seed=0):
    return current_dir / pickle_dir_name(alpha, label_ratio, split, seed)


def get_organamnist(client_id, batch_size, alpha=0.1, label_ratio=0.05, split="official", seed=0):
    pickle_dir = organamnist_pickle_dir(alpha, label_ratio, split, seed)
    if os.path.isdir(pickle_dir) is False:
        raise RuntimeError(
            "Please run data/organamnist/preprocess.py with matching --alpha, --label_ratio, --split"
            " (and --seed for the paper split) to generate data first.\n"
            f"Expected directory: {pickle_dir}"
        )
    with open("{}/client_{}.pkl".format(pickle_dir, client_id), "rb") as file:
        labeled_set, unlabeled_set, testset = pickle.load(file)

    labeled_loader = DataLoader(labeled_set, batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size, shuffle=True)

    return labeled_loader, unlabeled_loader, testloader
