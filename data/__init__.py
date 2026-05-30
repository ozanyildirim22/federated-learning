from .mnist import get_mnist
from .synthetic import get_synthetic
from .cifar import get_cifar
from .emnist import get_emnist
from .cifar100 import get_cifar100
from .organamnist import get_organamnist
from .bloodmnist import get_bloodmnist


def get_dataloader(client_id, dataset, batch_size, **kwargs):
    if dataset == "mnist":
        return get_mnist(client_id, batch_size)
    elif dataset == "cifar":
        return get_cifar(client_id, batch_size, **kwargs)
    elif dataset == "emnist":
        return get_emnist(client_id, batch_size)
    elif dataset == "cifar100":
        return get_cifar100(client_id, batch_size)
    elif dataset == "organamnist":
        return get_organamnist(client_id, batch_size)
    elif dataset == "bloodmnist":
        return get_bloodmnist(client_id, batch_size)
    elif dataset == "synthetic":
        return get_synthetic(client_id, batch_size)
    else:
        raise NotImplementedError(
            'Dataset "{}" is not supported.'.format(dataset)
        )


