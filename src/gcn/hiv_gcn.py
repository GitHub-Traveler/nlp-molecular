import torch.nn as _nn
import torch as _torch
import torch_geometric.nn as _geo_nn
from tqdm import tqdm as _tqdm
from sklearn.metrics import roc_curve as _roc_curve, auc as _auc

__all__  = ['GraphConvNet', 'train', 'eval']

class GraphConvNet(_nn.Module):
    """
    Graph Neural Network class for predicting molecule properties.
    """
    class ConvBlock(_nn.Module):
        def __init__(self, in_channel, out_channel) -> None:
            super().__init__()

            self.conv = _geo_nn.GCNConv(in_channel, out_channel)
            self.relu = _nn.ReLU()
        
        def forward(self, x, edge_list):
            hidden = self.conv(x, edge_list)
            return self.relu(hidden)

    def __init__(self, num_features, num_classes) -> None:
        super().__init__()
        self.num_channel = num_features
        self.conv1 = self.ConvBlock(self.num_channel, 16)
        self.conv2 = self.ConvBlock(16, 32)
        self.conv3 = self.ConvBlock(32, 64)
        self.conv4 = self.ConvBlock(64, 128)
        self.linear_layers = _nn.Sequential(
            _nn.Linear(128, 64),
            _nn.ReLU(),
            _nn.Linear(64, num_classes)
        )
        self.out = _nn.Sigmoid()

    def forward(self, x, edge_list, batch):
        hidden = self.conv1(x, edge_list)
        hidden = self.conv2(hidden, edge_list)
        hidden = self.conv3(hidden, edge_list)
        hidden = self.conv4(hidden, edge_list)
        hidden = _geo_nn.global_mean_pool(hidden, batch)
        return self.linear_layers(hidden)

def eval(net, test_dataloader, device = 'cpu'):
    """
    Evaluate the performance of the Graph Neural Network based on a test dataset.

    Return the TPR, FPR and ROC-AUC of the test.

    Parameters:

    net (GraphConvNet):
        nn.Module-based Graph Neural Network.
    test_dataloader (torch.utils.data.Dataloader): 
        Dataloader of the test dataset.
    device (torch.device): 
        Device to run the evaluation on. Default: 'cpu'.

    Returns:
    
    TPR: 
        NumPy True Positive Rate Array when calculating ROC
    FPR: 
        NumPy False Positive Rate Array when calculating ROC
    AUC: 
        Area Under Curve when calculating ROC.

    """
    net.eval()
    labels = []
    prediction_probabilities = []
    for batch in test_dataloader:
        input_x = batch.x.to(_torch.float32).to(device)
        input_batch_edge_index = batch.edge_index.to(device)
        input_batch = batch.batch.to(device)
        batch_y = batch.y.squeeze().to(_torch.int64).to(device)
        output = net(input_x, input_batch_edge_index, input_batch)
        labels.extend(list(batch_y.detach().cpu().numpy()))
        prediction_probabilities.extend(list(_torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()))
    fpr, tpr, _ = _roc_curve(labels, prediction_probabilities)
    roc_auc = _auc(fpr, tpr)
    return fpr, tpr, roc_auc

def train(net, loss_fn, optimizer, train_dataloader, test_dataloader, device = 'cpu', num_epoch = 300):
    """
    Train function for Graph Neural Network.

    Parameters:
    net (GraphConvNet):
        nn.Module-based Graph Neural Network.
    loss_fn:
        PyTorch Loss Function
    optimizer:
        PyTorch Optimizer
    test_dataloader (torch.utils.data.Dataloader): 
        Dataloader of the train dataset.
    test_dataloader (torch.utils.data.Dataloader): 
        Dataloader of the test dataset.
    device (torch.device): 
        Device to run the evaluation on. Default: 'cpu'.
    num_epoch (int):
        Number of epoches to train on.

    Returns: None
    """
    for i in _tqdm(range(num_epoch)):
        net.train()
        total_loss = 0
        total_correct = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_x = batch.x.to(_torch.float32).to(device)
            input_batch_edge_index = batch.edge_index.to(device)
            input_batch = batch.batch.to(device)
            batch_y = batch.y.squeeze().to(_torch.int64).to(device)

            output = net(input_x, input_batch_edge_index, input_batch)
            loss = loss_fn(output, batch_y)
            total_loss += loss.item()
            total_correct += (batch_y == _torch.argmax(output, dim = 1)).sum()
            loss.backward()
            optimizer.step()

        fpr, tpr, roc_auc = eval(net, test_dataloader, device)
        print(f'Epoch {i}. Loss: {total_loss}. Train Accuracy: {total_correct/len(test_dataloader.dataset)}. Test ROC: {roc_auc}')