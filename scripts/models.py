import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear
from torch_geometric.nn import GCNConv, global_mean_pool


class STGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(STGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, 256)
        self.lstm = torch.nn.LSTM(256, 128, batch_first=True)
        self.attention = torch.nn.MultiheadAttention(128, num_heads=8)
        self.linear1 = Linear(128, 64)
        self.linear2 = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Spatial modeling with GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)

        # Temporal modeling with LSTM
        x = x.unsqueeze(0)  # Add batch dimension
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)  # Remove batch dimension

        # Readout function
        x = global_mean_pool(x, data.batch)

        # Final classification layer
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.linear2(x)
        
        return F.log_softmax(x, dim=1)
