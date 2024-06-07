import matplotlib.pyplot as _plt
from torch_geometric.datasets import MoleculeNet as _MoleculeNet
from sklearn.model_selection import train_test_split as _train_test_split
import torch as _torch
from torch_geometric.loader import DataLoader as _Dataloader, ImbalancedSampler as _ImbalancedSampler

def plot_roc_curve(fpr, tpr, roc_auc, save_dir = None):
    """
    Function to plot the Receiver Operating Characteristic (ROC) curve, and its corresponding Area Under Curve (AOC)

    Parameters:
    
    fpr:
        NumPy array to store the false positive rates
    tpr:
        NumPy array to store the true positive rates
    roc_auc (int):
        Area under the curve
    save_dir (Dir, default = None):
        Specify the directory to save the figure if not None
    
    Returns: None
    """
    _plt.figure()
    _plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    _plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    _plt.xlim([0.0, 1.0])
    _plt.ylim([0.0, 1.05])
    _plt.xlabel('False Positive Rate')
    _plt.ylabel('True Positive Rate')
    _plt.title('Receiver Operating Characteristic')
    _plt.legend(loc="lower right")
    if save_dir is not None:
        _plt.savefig(save_dir)
    _plt.show()
    return None


def load_dataloader(root = "../../datasets", name = "HIV", imbalance_weight = True, ratio = 0.8):
    """
    Function to load the MoleculeNet dataset and return the dataloader for the train and test dataset.

    Parameters:

    root (str, default: "../../datasets"):
        Root of the full MoleculeNet
    name (str, default: "HIV"):
        Name of the dataset inside MoleculeNet
    imbalance_weight (bool, default: True)
        Whether to apply imbalance weights in the dataloader
    ratio (float, default: 0.8)
        Percentage of the whole dataset as the train dataset. Must take a number between 0 and 1.

    Returns:

    train_dataloader:
        DataLoader of the training dataset
    test_dataloader:
        DataLoader of the test dataset
    """
    dataset = _MoleculeNet(root = root, name = name)
    dataset.y = dataset.y.to(_torch.int64)
    label = dataset.y.numpy()
    train_indices, test_indices = _train_test_split(
        range(len(label)), test_size=1 - ratio, stratify=label, random_state=42
    )

    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]
    train_dataset.y = train_dataset.y[train_indices]
    test_dataset.y = test_dataset.y[train_indices]

    train_dataloader, test_dataloader = None, None
    if imbalance_weight:
        sampler = _ImbalancedSampler(train_dataset)
        train_dataloader = _Dataloader(train_dataset, batch_size = 64, num_workers = 8, sampler = sampler)
        test_dataloader = _Dataloader(test_dataset, batch_size = len(test_dataset), num_workers = 8)
    else:
        train_dataloader = _Dataloader(train_dataset, batch_size = 64, num_workers = 8, shuffle = True)
        test_dataloader = _Dataloader(test_dataset, batch_size = len(test_dataset), num_workers = 8)

    return train_dataloader, test_dataloader