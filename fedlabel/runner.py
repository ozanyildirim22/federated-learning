import os
import random
from os import listdir
from tqdm import trange
import torch

from utils import get_model, evaluate
from data import get_dataloader
from fedlabel.server import Server
from fedlabel.client import Client

# Dataset imports to ensure pickle deserialization succeeds
from data.cifar import CIFARDataset
from data.mnist import MNISTDataset
from data.synthetic import SyntheticDataset
from data.emnist import EMNISTDataset
from data.cifar100 import CIFAR100Dataset
from data.organamnist import OrganAMNISTDataset
from data.bloodmnist import BloodMNISTDataset
from data.pathamnist import PathMNISTDataset


class FedLabelRunner:
    """
    Coordinates end-to-end execution of the FedLabel algorithm (ICCV 2023).
    Manages client initialization, server aggregation, and evaluation.
    """
    def __init__(self, config):
        self.config = config
        
        # Setup device & seeds
        use_cuda = config.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if use_cuda:
            torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        
        # Global model initialization
        self.global_model = get_model((config.model, config.dataset)).to(self.device)
        self.server = Server(self.global_model)
        
        # Locate preprocessed partitioned client pickles
        if config.dataset == "synthetic":
            self.pickle_dir = "data/{}/pickles_label{}".format(config.dataset, config.label_ratio)
        else:
            self.pickle_dir = "data/{}/pickles_alpha{}_label{}".format(config.dataset, config.alpha, config.label_ratio)
            
        if not os.path.isdir(self.pickle_dir):
            raise FileNotFoundError(
                f"Dataset directory '{self.pickle_dir}' not found. "
                f"Please run 'python data/{config.dataset}/preprocess.py --alpha {config.alpha} --label_ratio {config.label_ratio}' first."
            )
            
        self.client_num_in_total = len(listdir(self.pickle_dir))
        self.client_indices = list(range(self.client_num_in_total))
        
        self.clients = []
        self._setup_clients()

    def _setup_clients(self):
        """
        Loads client dataloaders and constructs Client instances.
        """
        print(f"\033[1;32mUsing device: {self.device}\033[0m")
        print(f"Loading data for {self.client_num_in_total} clients from {self.pickle_dir}...")
        
        for client_id in range(self.client_num_in_total):
            loaders = get_dataloader(
                client_id, 
                self.config.dataset, 
                self.config.batch_size, 
                alpha=self.config.alpha, 
                label_ratio=self.config.label_ratio
            )
            if len(loaders) == 3:
                labeled_loader, unlabeled_loader, valloader = loaders
            else:
                labeled_loader, valloader = loaders
                unlabeled_loader = None
                
            client = Client(
                client_id=client_id,
                config=self.config,
                device=self.device,
                labeled_loader=labeled_loader,
                unlabeled_loader=unlabeled_loader,
                valloader=valloader
            )
            self.clients.append(client)

    def train_round(self, round_idx):
        """
        Executes a single communication round of FedLabel:
          1. Select m clients uniformly at random.
          2. Each client performs supervised & semi-supervised updates.
          3. Server aggregates weight updates based on sample weights r_k.
        """
        selected_client_ids = self.server.select_clients(
            self.client_indices, self.config.client_num_per_round
        )
        print(f"\033[1;34mselected clients in round [{round_idx}]: {selected_client_ids}\033[0m")
        
        client_updates = []
        for cid in selected_client_ids:
            client = self.clients[cid]
            delta_w, r_k = client.update(self.server.global_model)
            client_updates.append((delta_w, r_k))
            
        self.server.aggregate(client_updates)

    def evaluate(self):
        """
        Evaluates the global model across test/validation loaders of selected clients.
        """
        criterion = torch.nn.CrossEntropyLoss()
        avg_loss = 0.0
        avg_acc = 0.0
        
        for _ in trange(self.config.test_round, desc="\033[1;36mevaluating epoch\033[0m"):
            selected_client_ids = self.server.select_clients(
                self.client_indices, self.config.client_num_per_round
            )
            for cid in selected_client_ids:
                client = self.clients[cid]
                loss, acc = evaluate(self.server.global_model, client.valloader, criterion, self.device)
                avg_loss += float(loss)
                avg_acc += float(acc)
                
        total_evals = self.config.client_num_per_round * self.config.test_round
        avg_loss /= total_evals
        avg_acc /= total_evals
        
        return avg_loss, avg_acc

    def run(self):
        """
        Runs full training loop and outputs final performance metrics.
        """
        print(f"\033[1;33mStarting FedLabel Training for {self.config.comms_round} rounds...\033[0m")
        for r in trange(self.config.comms_round, desc="\033[1;33mtraining epoch\033[0m"):
            self.train_round(r)
            
        print("\033[1;36mStarting Final Evaluation...\033[0m")
        avg_loss, avg_acc = self.evaluate()
        
        print("\033[1;32m---------------------- RESULTS ----------------------\033[0m")
        print("\033[1;33m Global FedLabel loss: {:.4f}\033[0m".format(avg_loss))
        print("\033[1;33m Global FedLabel accuracy: {:.2f}%\033[0m".format(avg_acc))
        return avg_loss, avg_acc
