from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

dataset = Planetoid(root='Cora', name='Cora')
cora = dataset[0]

print(len(dataset))
print(dataset.num_classes)
print(dataset.num_node_features)

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
num_features = dataset.num_features

print(f'Number of classes: {dataset.num_classes}')

print(f'Number of nodes: {cora.num_nodes}')
print(f'Number of edges: {cora.num_edges}')
print(f'Average node degree: {cora.num_edges / cora.num_nodes:.2f}')
print(f'Number of training nodes: {cora.train_mask.sum()}')
print(f'Training node label rate: {int(cora.train_mask.sum()) / cora.num_nodes:.2f}')
print(f'Contains isolated nodes: {cora.contains_isolated_nodes()}')
print(f'Contains self-loops: {cora.contains_self_loops()}')
print(f'Is undirected: {cora.is_undirected()}')

# # Convert the edge index to an adjacency matrix
# adj_matrix = nx.to_numpy_array(nx.from_edgelist(cora.edge_index.t().tolist()))
# # Create a NetworkX graph from the adjacency matrix
# G = nx.from_numpy_array(adj_matrix)
# # Set the node color based on the true class label
# node_colors = [cora.y[i] for i in range(len(cora.y))]
# # Draw the graph
# pos = nx.spring_layout(G)
# nx.draw_networkx(G, pos=pos, node_color=node_colors, cmap=plt.cm.tab10, with_labels=True)
# plt.show()



x = cora.x
edge_index = cora.edge_index
data = torch_geometric.data.Data(x=x, edge_index=edge_index)
g = torch_geometric.utils.to_networkx(data, to_undirected=False)
nx.draw_networkx(g, with_labels=True)
plt.show()


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
