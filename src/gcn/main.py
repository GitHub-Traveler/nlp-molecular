# %%
import torch as _torch
import torch.nn as _nn
from torch.cuda import is_available as _cuda_available
import hiv_gcn, utils

if __name__ == "__main__":
    device = 'cuda' if _cuda_available() else 'cpu'
    num_epoch = 10

    train_dataloader, test_dataloader = utils.load_dataloader()
    net = hiv_gcn.GraphConvNet(train_dataloader.dataset.num_features, 2)
    net.to(device)
    optimizer = _torch.optim.Adam(net.parameters(), lr = 2e-4, weight_decay=1e-6)
    
    loss_fn = _nn.CrossEntropyLoss()
    hiv_gcn.train(net, loss_fn, optimizer, train_dataloader, test_dataloader, device, num_epoch)
    fpr, tpr, auc = hiv_gcn.eval(net, test_dataloader, device)
    utils.plot_roc_curve(fpr, tpr, auc, "test.png")

