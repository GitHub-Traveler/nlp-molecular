import torch.nn as nn
import torch_geometric.nn as geo_nn

class GraphConvNet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channel, out_channel) -> None:
            super().__init__()

            self.conv = geo_nn.GCNConv(in_channel, out_channel)
            self.relu = nn.ReLU()
        
        def forward(self, x, edge_list):
            hidden = self.conv(x, edge_list)
            return self.relu(hidden)

    def __init__(self, num_features, num_classes) -> None:
        super().__init__()
        self.num_channel = num_features
        self.conv1 = self.ConvBlock(self.num_channel, 16)
        self.conv2 = self.ConvBlock(16, 32)
        self.conv3 = self.ConvBlock(32, 64)
        self.linear_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.out = nn.Sigmoid()

    def forward(self, x, edge_list, batch):
        hidden = self.conv1(x, edge_list)
        hidden = self.conv2(hidden, edge_list)
        hidden = self.conv3(hidden, edge_list)
        hidden = geo_nn.global_mean_pool(hidden, batch)
        return self.linear_layers(hidden)