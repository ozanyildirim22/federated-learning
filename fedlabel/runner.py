import os
import random
from os import listdir
from tqdm import trange
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

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
        self.global_model = get_model(
            (config.model, config.dataset), cifar_stem=getattr(config, "cifar_stem", True)
        ).to(self.device)
        self.server = Server(self.global_model)

        # Locate preprocessed partitioned client pickles
        self.split = getattr(config, "split", "official")
        if config.dataset == "synthetic":
            self.pickle_dir = "data/{}/pickles_label{}".format(config.dataset, config.label_ratio)
        elif config.dataset == "cifar":
            from data.cifar import cifar_pickle_dir
            self.pickle_dir = str(cifar_pickle_dir(config.alpha, config.label_ratio, self.split, config.seed))
        elif config.dataset == "organamnist":
            from data.organamnist import organamnist_pickle_dir
            self.pickle_dir = str(organamnist_pickle_dir(config.alpha, config.label_ratio, self.split, config.seed))
        else:
            if self.split == "paper":
                raise NotImplementedError(f"--split paper is only implemented for cifar and organamnist, not {config.dataset}")
            self.pickle_dir = "data/{}/pickles_alpha{}_label{}".format(config.dataset, config.alpha, config.label_ratio)

        if not os.path.isdir(self.pickle_dir):
            raise FileNotFoundError(
                f"Dataset directory '{self.pickle_dir}' not found. "
                f"Please run 'python data/{config.dataset}/preprocess.py --alpha {config.alpha} --label_ratio {config.label_ratio}"
                + (f" --split paper --seed {config.seed}" if self.split == "paper" else "")
                + "' first."
            )

        self.client_num_in_total = len(listdir(self.pickle_dir))
        self.client_indices = list(range(self.client_num_in_total))

        self.clients = []
        self._setup_clients()

        # Global test set: CIFAR-10 official split -> the official 10000 test images;
        # CIFAR-10 paper split -> the pooled 15% (9000) test set; other datasets -> union of
        # all clients' test splits (OrganAMNIST official -> 17778, the full official test set;
        # OrganAMNIST paper split -> the pooled 15% (8825) test set).
        self.global_testloader = None
        if getattr(config, "global_test", True):
            self.global_testloader = DataLoader(
                self._build_global_testset(), batch_size=500, shuffle=False
            )
            print(f"\033[1;35mGlobal test set size: {len(self.global_testloader.dataset)}\033[0m")
        else:
            print("\033[1;35mglobal_test is OFF: evaluating on random clients' val loaders.\033[0m")

        self.history = []

    def _build_global_testset(self):
        """
        Full, un-partitioned test set used to score the global model (paper protocol).
        CIFAR-10 with the official split loads the official 10k test split directly, with
        the same ToTensor() transform the preprocess uses. With --split paper (and for
        other datasets) it is the union of all clients' test splits, which for the paper
        split is exactly the pooled 15% (9k) test set.
        """
        if self.config.dataset == "cifar" and self.split == "official":
            from torchvision.datasets import CIFAR10
            from torchvision import transforms
            root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cifar")
            testset = CIFAR10(root, train=False, transform=transforms.ToTensor(), download=True)
            assert len(testset) == 10000, f"Expected 10000 CIFAR-10 test images, got {len(testset)}"
            return testset
        return ConcatDataset([c.valloader.dataset for c in self.clients])

    def _setup_clients(self):
        """
        Loads client dataloaders and constructs Client instances.
        """
        print(f"\033[1;32mUsing device: {self.device}\033[0m")
        print(f"Loading data for {self.client_num_in_total} clients from {self.pickle_dir}...")

        extra = (
            {"split": self.split, "seed": self.config.seed}
            if self.config.dataset in ("cifar", "organamnist") else {}
        )
        print(f"\033[1;32mAlgorithm: {getattr(self.config, 'algorithm', 'fedlabel')}\033[0m")
        for client_id in range(self.client_num_in_total):
            loaders = get_dataloader(
                client_id,
                self.config.dataset,
                self.config.batch_size,
                alpha=self.config.alpha,
                label_ratio=self.config.label_ratio,
                **extra
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

        algorithm = getattr(self.config, "algorithm", "fedlabel")
        client_updates = []
        for cid in selected_client_ids:
            client = self.clients[cid]
            if algorithm == "fedlabel":
                delta_w, r_k = client.update(self.server.global_model)
            else:
                # Supervised FedAvg baselines of the paper's Table 3/4, trained and scored
                # with exactly the same data, client sampling and global test set.
                delta_w, r_k = client.update_fedavg(
                    self.server.global_model, fully_labeled=(algorithm == "fedavg_full")
                )
            client_updates.append((delta_w, r_k))

        self.server.aggregate(client_updates)

    @torch.no_grad()
    def evaluate_global(self):
        """
        Evaluates the global model on the union of all clients' test splits.
        Self-contained (does not depend on utils.evaluate), so the loss is
        a true per-sample cross-entropy and accuracy is in percent.
        Also stores macro precision / recall / F1 (percent) in self.last_metrics:
        per-class scores averaged over every class that occurs in the labels or the
        predictions, with 0 for an undefined ratio (sklearn's average='macro',
        zero_division=0). Macro F1 weights rare classes equally, which matters for the
        class-imbalanced OrganAMNIST test set.
        """
        model = self.server.global_model
        model.eval()
        total_loss, correct, n = 0.0, 0, 0
        max_abs_logit = 0.0
        tp = fp = fn = None

        for x, y in self.global_testloader:
            x = x.to(self.device)
            y = y.to(self.device).long().view(-1)
            out = model(x)
            pred = out.argmax(dim=1)
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
            correct += (pred == y).sum().item()
            n += y.size(0)
            max_abs_logit = max(max_abs_logit, out.abs().max().item())

            c = out.size(1)
            if tp is None:
                tp, fp, fn = (torch.zeros(c, dtype=torch.long, device=self.device) for _ in range(3))
            hit = pred == y
            tp += torch.bincount(y[hit], minlength=c)
            fp += torch.bincount(pred[~hit], minlength=c)
            fn += torch.bincount(y[~hit], minlength=c)

        tp, fp, fn = tp.double(), fp.double(), fn.double()
        present = (tp + fn + fp) > 0
        precision = torch.where(tp + fp > 0, tp / (tp + fp).clamp(min=1), torch.zeros_like(tp))
        recall = torch.where(tp + fn > 0, tp / (tp + fn).clamp(min=1), torch.zeros_like(tp))
        f1 = torch.where(precision + recall > 0,
                         2 * precision * recall / (precision + recall).clamp(min=1e-12),
                         torch.zeros_like(tp))
        self.last_metrics = {
            "precision": 100.0 * precision[present].mean().item(),
            "recall": 100.0 * recall[present].mean().item(),
            "f1": 100.0 * f1[present].mean().item(),
        }
        self.last_max_abs_logit = max_abs_logit
        return total_loss / n, 100.0 * correct / n

    def evaluate(self):
        """
        Fallback: evaluates the global model on val loaders of randomly selected clients.
        Only used when global_test is disabled.
        """
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_acc = 0.0
        total_evals = 0

        for _ in trange(self.config.test_round, desc="\033[1;36mevaluating epoch\033[0m"):
            selected_client_ids = self.server.select_clients(
                self.client_indices, self.config.client_num_per_round
            )
            for cid in selected_client_ids:
                client = self.clients[cid]
                loss, acc = evaluate(self.server.global_model, client.valloader, criterion, self.device)
                total_loss += float(loss) / len(client.valloader.dataset)
                total_acc += float(acc)
                total_evals += 1

        total_evals = max(total_evals, 1)
        return total_loss / total_evals, total_acc / total_evals

    def run(self):
        """
        Runs full training loop and outputs final performance metrics.
        """
        name = {"fedlabel": "FedLabel", "fedavg": "FedAvg", "fedavg_full": "FedAvg (100%)"}[getattr(self.config, "algorithm", "fedlabel")]
        print(f"\033[1;33mStarting {name} Training for {self.config.comms_round} rounds...\033[0m")
        eval_every = getattr(self.config, "eval_every", 0)

        for r in trange(self.config.comms_round, desc="\033[1;33mtraining epoch\033[0m"):
            self.train_round(r)
            if self.global_testloader is not None and eval_every > 0 and (r + 1) % eval_every == 0:
                loss, acc = self.evaluate_global()
                self.history.append((r + 1, loss, acc, self.last_metrics["f1"]))
                print(
                    f"\n\033[1;36m[round {r + 1}] global test loss {loss:.4f} | "
                    f"acc {acc:.2f}% | f1 {self.last_metrics['f1']:.2f}% | "
                    f"max|logit| {self.last_max_abs_logit:.1f}\033[0m"
                )

        print("\033[1;36mStarting Final Evaluation...\033[0m")
        if self.global_testloader is not None:
            avg_loss, avg_acc = self.evaluate_global()
        else:
            avg_loss, avg_acc = self.evaluate()

        print("\033[1;32m---------------------- RESULTS ----------------------\033[0m")
        print("\033[1;33m Global {} loss: {:.4f}\033[0m".format(name, avg_loss))
        print("\033[1;33m Global {} accuracy: {:.2f}%\033[0m".format(name, avg_acc))
        if self.global_testloader is not None:
            m = self.last_metrics
            print("\033[1;33m Global {} macro-precision: {:.2f}%\033[0m".format(name, m["precision"]))
            print("\033[1;33m Global {} macro-recall: {:.2f}%\033[0m".format(name, m["recall"]))
            print("\033[1;33m Global {} macro-F1: {:.2f}%\033[0m".format(name, m["f1"]))
        return avg_loss, avg_acc
