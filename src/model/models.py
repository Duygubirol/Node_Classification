import torch.nn as nn
import torch.nn.functional as F
from src.model.layers import GraphConvolution


class GCN(nn.Module):
    '''
    Graph Convolutional Network (GCN) model for node classification.

    Args:
        nfeat (int): Number of input features.
        nhid (int): Number of hidden units.
        nclass (int): Number of output classes.
        dropout (float): Dropout probability.

    Attributes:
        gc1 (GraphConvolution): First graph convolutional layer.
        gc2 (GraphConvolution): Second graph convolutional layer.
        dropout (float): Dropout probability.
    '''

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        '''
        Forward pass of the GCN model.

        Args:
            x (torch.Tensor): Input feature matrix.
            adj (torch.sparse.FloatTensor): Sparse adjacency matrix.

        Returns:
            torch.Tensor: Log softmax output tensor.
        '''
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
