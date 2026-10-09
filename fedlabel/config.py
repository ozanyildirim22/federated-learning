import argparse

def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "t", "1", "yes", "y"):
        return True
    if value in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected for --cuda (true/false).")

def get_args(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--comms_round", type=int, default=40)
    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--test_round", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5, help="Standard epochs for backward compatibility, not used by FedLabel directly.")
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
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet distribution parameter alpha")
    parser.add_argument("--label_ratio", type=float, default=0.2, help="Fraction of labeled training data per client")
    
    # FedLabel hyperparameters
    parser.add_argument("--beta", type=float, default=0.95, help="Confidence threshold")
    parser.add_argument("--lambda_0", type=float, default=1.0, help="KL Divergence adaptive weight scale")
    parser.add_argument("--tau", type=int, default=5, help="Labeled steps/epochs")
    parser.add_argument("--tau_prime", type=int, default=5, help="Unlabeled steps/epochs")
    
    return parser.parse_args()
