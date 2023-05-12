import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


class STGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(STGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.lstm = torch.nn.LSTM(64, 32, batch_first=True)
        self.linear = Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Spatial modeling with GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Temporal modeling with LSTM
        x = x.unsqueeze(0)  # Add batch dimension
        x, _ = self.lstm(x)
        x = x.squeeze(0)  # Remove batch dimension

        # Readout function
        x = global_mean_pool(x, data.batch)

        # Final classification layer
        x = self.linear(x)
        
        return F.log_softmax(x, dim=1)
