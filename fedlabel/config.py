import argparse

def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "t", "1", "yes", "y"):
        return True
    if value in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def get_args(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser(description="FedLabel: Selective Knowledge Assimilation for Federated Learning with Limited Labels")
        
    # Baseline & Federated Learning Setup
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name: mnist, cifar, organamnist, bloodmnist, etc.")
    parser.add_argument("--model", type=str, default="cnn", help="Model architecture: cnn, resnet18, resnet34, resnet50, mlp")
    parser.add_argument("--comms_round", type=int, default=40, help="Total communication rounds")
    parser.add_argument("--client_num_per_round", type=int, default=10, help="Number of clients selected per round (m)")
    parser.add_argument("--test_round", type=int, default=1, help="Number of evaluation rounds at completion")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs argument for baseline compatibility")
    parser.add_argument("--batch_size", type=int, default=20, help="Local batch size")
    parser.add_argument("--global_lr", type=float, default=1.0, help="Server aggregation learning rate")
    parser.add_argument("--local_lr", type=float, default=5e-2, help="Client local learning rate (eta)")
    parser.add_argument(
        "--cuda",
        type=_str2bool,
        default=True,
        help="Use CUDA if available (true/false).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet heterogeneity parameter alpha")
    parser.add_argument("--label_ratio", type=float, default=0.2, help="Fraction of labeled training data per client (e.g. 0.2 or 0.05)")
    
    # FedLabel Specific Hyperparameters (ICCV 2023)
    parser.add_argument("--beta", type=float, default=0.95, help="Confidence threshold beta for pseudo-labeling (eq. 4)")
    parser.add_argument("--lambda_0", type=float, default=1.0, help="Upper bound weight factor lambda_0 for KL consistency loss (eq. 7)")
    parser.add_argument("--tau", type=int, default=5, help="Labeled steps (mini-batch iterations) per client round")
    parser.add_argument("--tau_prime", type=int, default=5, help="Unlabeled steps (mini-batch iterations) per client round")
    parser.add_argument("--confidence_metric", type=str, default="variance", choices=["variance", "entropy", "max_prob"], help="Confidence scoring function h(.)")
    parser.add_argument("--use_epochs", type=_str2bool, default=False, help="If true, tau and tau_prime represent full epochs over loaders instead of mini-batch steps")
    
    return parser.parse_args()
