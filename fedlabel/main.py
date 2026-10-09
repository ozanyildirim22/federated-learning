import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
from argparse import ArgumentParser
from tqdm import trange

from utils import get_model, evaluate
from data import get_dataloader
from fedlabel.config import get_args
from fedlabel.server import Server
from fedlabel.client import Client

# Required for unpickling locally generated datasets in PyTorch
from data.cifar import CIFARDataset
from data.mnist import MNISTDataset
from data.synthetic import SyntheticDataset
from data.emnist import EMNISTDataset
from data.cifar100 import CIFAR100Dataset
from data.organamnist import OrganAMNISTDataset
from data.bloodmnist import BloodMNISTDataset
from data.pathamnist import PathMNISTDataset
from os import listdir

if __name__ == "__main__":
    parser = ArgumentParser()
    args = get_args(parser)
    
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    print(f"\033[1;32mUsing device: {device}\033[0m")
    
    global_model = get_model((args.model, args.dataset))
    global_model = global_model.to(device)
    
    # Determine number of clients based on pickles directory
    if args.dataset == "synthetic":
        pickle_dir = "data/{}/pickles_label{}".format(args.dataset, args.label_ratio)
    else:
        pickle_dir = "data/{}/pickles_alpha{}_label{}".format(args.dataset, args.alpha, args.label_ratio)
    
    try:
        client_num_in_total = len(listdir(pickle_dir))
    except FileNotFoundError:
        print(f"\033[1;31mDataset not found. Please run preprocess script for {args.dataset}.\033[0m")
        sys.exit(1)
        
    client_indices = list(range(client_num_in_total))
    
    print("Loading data for all clients...")
    clients = []
    for client_id in range(client_num_in_total):
        loaders = get_dataloader(client_id, args.dataset, args.batch_size, alpha=args.alpha, label_ratio=args.label_ratio)
        if len(loaders) == 3:
            labeled_loader, unlabeled_loader, valloader = loaders
        else:
            labeled_loader, valloader = loaders
            unlabeled_loader = None
            
        client = Client(client_id, args, device, labeled_loader, unlabeled_loader, valloader)
        clients.append(client)
        
    server = Server(global_model)
    
    for r in trange(args.comms_round, desc="\033[1;33mtraining epoch\033[0m"):
        selected_clients_ids = server.select_clients(client_indices, args.client_num_per_round)
        print(f"\033[1;34mselected clients in round [{r}]: {selected_clients_ids}\033[0m")
        
        client_updates = []
        for client_id in selected_clients_ids:
            client = clients[client_id]
            delta_w, r_k = client.update(server.global_model)
            client_updates.append((delta_w, r_k))
            
        # Aggregate the weight updates based on the FedLabel aggregation scheme
        server.aggregate(client_updates)
        
    # Evaluate
    criterion = torch.nn.CrossEntropyLoss()
    avg_loss_g, avg_acc_g = 0.0, 0.0
    
    for r in trange(args.test_round, desc="\033[1;36mevaluating epoch\033[0m"):
        selected_clients_ids = server.select_clients(client_indices, args.client_num_per_round)
        for client_id in selected_clients_ids:
            client = clients[client_id]
            loss, acc = evaluate(server.global_model, client.valloader, criterion, device)
            avg_loss_g += loss.item()
            avg_acc_g += acc.item()
            
    avg_loss_g /= (args.client_num_per_round * args.test_round)
    avg_acc_g /= (args.client_num_per_round * args.test_round)
    
    print("\033[1;32m---------------------- RESULTS ----------------------\033[0m")
    print("\033[1;33m Global FedLabel loss: {:.4f}\033[0m".format(avg_loss_g))
    print("\033[1;33m Global FedLabel accuracy: {:.2f}%\033[0m".format(avg_acc_g))
