import os
import pickle
from path import Path
from torch.utils.data import DataLoader
from .preprocess import CIFARDataset

current_dir = Path(__file__).parent.absolute()


def get_cifar(client_id, batch_size, alpha=0.1, label_ratio=0.2):
    pickle_dir = current_dir / f"pickles_alpha{alpha}_label{label_ratio}"
    if os.path.isdir(pickle_dir) is False:
        raise RuntimeError(
            "Please run data/cifar/preprocess.py with matching --alpha and --label_ratio to generate data first.\n"
            f"Expected directory: {pickle_dir}"
        )
    with open("{}/client_{}.pkl".format(pickle_dir, client_id), "rb") as file:
        labeled_set, unlabeled_set, testset = pickle.load(file)

    labeled_loader = DataLoader(labeled_set, batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size, shuffle=True)

    return labeled_loader, unlabeled_loader, testloader
