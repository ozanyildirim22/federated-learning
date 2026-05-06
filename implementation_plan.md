# Replicate ICCV 2023 Paper Experimental Setup for CIFAR-10

Based on the paper "Local or Global: Selective Knowledge Assimilation for Federated Learning with Limited Labels" (Cho et al., ICCV 2023), the experimental setup for evaluating FedAvg on the CIFAR-10 dataset requires specific data partitioning and model architecture that differ from the current simple setup in your codebase.

Currently, your setup uses:
* **Model:** A simple CNN (`CNN_CIFAR`).
* **Data Partitioning:** Pathological non-IID splitting where each client only gets exactly 2 classes.
* **Labeling:** Full supervision (although the paper evaluates both fully-supervised baselines and semi-supervised settings, we need the exact data distribution first).

The paper uses:
* **Model:** ResNet-34.
* **Data Partitioning:** Dirichlet non-IID partitioning with $\alpha = 0.1$ across 100 clients.
* **Selection:** 10 clients selected per round.

To allow you to compare your FedAvg results fairly with the results in the paper (specifically the `FedAvg (100%)` baseline or `FedAvg` with partially labeled data), we need to update your code to match their environment.

## User Review Required

> [!WARNING]
> This plan will overwrite your current `data/cifar/pickles` folder. The new pickles will be partitioned using the Dirichlet distribution ($\alpha=0.1$) instead of the current 2-class pathological split. Let me know if you want me to backup the old pickles first!

## Proposed Changes

### `data/cifar/preprocess.py`
We will modify the CIFAR-10 preprocessing script to use the Dirichlet distribution partitioning provided by the `fedlab` library.
* [MODIFY] `data/cifar/preprocess.py`: Replace `noniid_slicing` with `hetero_dir_partition(targets, num_clients=100, num_classes=10, dir_alpha=0.1)` to accurately replicate the data heterogeneity parameter ($\alpha=0.1$) used in the paper.

### `models.py`
We will add the ResNet-34 architecture.
* [MODIFY] `models.py`: Import `torchvision.models.resnet34` and wrap it in a new class `ResNet34_CIFAR` that properly maps to the `("resnet34", "cifar")` model info. The ResNet-34 will be adjusted slightly for CIFAR-10 (e.g., modifying the first convolutional layer and removing the maxpool, or using a standard implementation) to handle 32x32 images instead of 224x224.

### `utils.py`
We will update the model selection logic so you can choose ResNet-34 via the command line.
* [MODIFY] `utils.py`: Add `resnet34` to the `get_model` function so that `--model resnet34` properly initializes the new `ResNet34_CIFAR` model.

## Verification Plan

1. Execute the updated `data/cifar/preprocess.py` script to generate the new Dirichlet partitioned pickles.
2. Run a short trial of `main.py` using `--dataset cifar --model resnet34 --client_num_per_round 10 --comms_round 5` to verify that the ResNet-34 model trains correctly on the new data without crashing.
3. Provide you with the exact command to run the full training loop so you can compare the ﬁnal accuracy against the paper's FedAvg baseline results (Table 3 & 4 in the paper).
