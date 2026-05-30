import torch
from argparse import ArgumentParser, ArgumentTypeError
from models import *


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "t", "1", "yes", "y"):
        return True
    if value in ("false", "f", "0", "no", "n"):
        return False
    raise ArgumentTypeError("Boolean value expected for --cuda (true/false).")


def get_args(parser: ArgumentParser):
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--comms_round", type=int, default=40)
    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--test_round", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--global_lr", type=float, default=1.0)
    parser.add_argument("--local_lr", type=float, default=5e-2)
    parser.add_argument(
        "--cuda",
        type=_str2bool,
        default=True,
        help="Use CUDA if available (true/false).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet distribution parameter alpha (used to locate preprocessed data)")
    parser.add_argument("--label_ratio", type=float, default=0.2, help="Fraction of labeled training data per client (used to locate preprocessed data)")
    return parser.parse_args()


def get_model(model_info):
    _struct, _dataset = model_info
    if _dataset == "mnist":
        if _struct == "mlp":
            return MLP_MNIST()
        elif _struct == "cnn":
            return CNN_MNIST()
        else:
            raise ValueError
    elif _dataset == "cifar":
        if _struct == "cnn":
            return CNN_CIFAR()
        elif _struct == "resnet18":
            return ResNet18_CIFAR()
        elif _struct == "resnet34":
            return ResNet34_CIFAR()
        else:
            raise NotImplementedError
    elif _dataset == "femnist":
        if _struct == "mlp":
            return MLP_FEMNIST()
        elif _struct == "cnn":
            return CNN_FEMNIST()
        else:
            raise ValueError
    elif _dataset == "emnist":
        if _struct == "mlp":
            return MLP_EMNIST()
        elif _struct == "cnn":
            return CNN_EMNIST()
        elif _struct == "resnet18":
            return ResNet18_EMNIST()
        else:
            raise ValueError
    elif _dataset == "synthetic":
        if _struct == "mlp":
            return MLP_SYNTHETIC()
        else:
            raise NotImplementedError
    elif _dataset == "cifar100":
        if _struct == "resnet50":
            return ResNet50_CIFAR100()
        else:
            raise NotImplementedError
    elif _dataset == "organamnist":
        if _struct == "resnet18":
            return ResNet18_OrganAMNIST()
        else:
            raise NotImplementedError
    elif _dataset == "bloodmnist":
        if _struct == "resnet18":
            return ResNet18_BloodMNIST()
        else:
            raise NotImplementedError


@torch.no_grad()
def evaluate(model, testloader, criterion, gpu=None):
    model.eval()
    correct = 0
    loss = 0
    if gpu is not None:
        model = model.to(gpu)
    for x, y in testloader:
        if gpu is not None:
            x, y = x.to(gpu), y.to(gpu)

        logit = model(x)
        loss += criterion(logit, y)

        pred_y = torch.softmax(logit, -1).argmax(-1)
        correct += torch.eq(pred_y, y).int().sum()

    acc = 100.0 * (correct / len(testloader.dataset))
    return loss, acc
