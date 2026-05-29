from torch import nn
from torchvision.models import resnet34, resnet18, resnet50


class CNN_MNIST(nn.Module):
    def __init__(self) -> None:
        super(CNN_MNIST, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("cnn", "mnist")


class CNN_EMNIST(nn.Module):
    def __init__(self) -> None:
        super(CNN_EMNIST, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 62),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("cnn", "emnist")


class MLP_EMNIST(nn.Module):
    def __init__(self) -> None:
        super(MLP_EMNIST, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 200),
            nn.ReLU(True),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, 62),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("mlp", "emnist")


class ResNet18_EMNIST(nn.Module):
    def __init__(self) -> None:
        super(ResNet18_EMNIST, self).__init__()
        self.net = resnet18(weights=None)
        # Adapt ResNet-18 for EMNIST (1-channel 28x28 instead of 3-channel 224x224)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 62)

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("resnet18", "emnist")


class MLP_MNIST(nn.Module):
    def __init__(self) -> None:
        super(MLP_MNIST, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 200),
            nn.ReLU(True),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, 10),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("mlp", "mnist")


class MLP_SYNTHETIC(nn.Module):
    def __init__(self) -> None:
        super(MLP_SYNTHETIC, self).__init__()
        self.net = nn.Linear(60, 10)

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("mlp", "synthetic")


class MLP_FEMNIST(nn.Module):
    def __init__(self) -> None:
        super(MLP_FEMNIST, self).__init__()
        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("mlp", "femnist")


class CNN_FEMNIST(nn.Module):
    def __init__(self) -> None:
        super(CNN_FEMNIST, self).__init__()
        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("cnn", "femnist")


class CNN_CIFAR(nn.Module):
    def __init__(self) -> None:
        super(CNN_CIFAR, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("cnn", "cifar")


class ResNet34_CIFAR(nn.Module):
    def __init__(self) -> None:
        super(ResNet34_CIFAR, self).__init__()
        self.net = resnet34(weights=None)
        # Adapt ResNet-34 for CIFAR-10 (32x32 instead of 224x224)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("resnet34", "cifar")


class ResNet18_CIFAR(nn.Module):
    def __init__(self) -> None:
        super(ResNet18_CIFAR, self).__init__()
        self.net = resnet18(weights=None)
        # Adapt ResNet-18 for CIFAR-10 (32x32 instead of 224x224)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("resnet18", "cifar")


class ResNet50_CIFAR100(nn.Module):
    def __init__(self) -> None:
        super(ResNet50_CIFAR100, self).__init__()
        self.net = resnet50(weights=None)
        # Adapt ResNet-50 for CIFAR-100 (32x32 instead of 224x224)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 100)

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("resnet50", "cifar100")


class ResNet18_OrganAMNIST(nn.Module):
    def __init__(self) -> None:
        super(ResNet18_OrganAMNIST, self).__init__()
        self.net = resnet18(weights=None)
        # Use standard ResNet-18 (which downsamples 28x28 images heavily, lowering accuracy to ~71%)
        self.net.fc = nn.Linear(self.net.fc.in_features, 11)

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("resnet18", "organamnist")


class ResNet18_BloodMNIST(nn.Module):
    def __init__(self) -> None:
        super(ResNet18_BloodMNIST, self).__init__()
        self.net = resnet18(weights=None)
        # Adapt ResNet-18 for BloodMNIST (3-channel 28x28)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 8)

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("resnet18", "bloodmnist")
