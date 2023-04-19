import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)


data = torch_geometric.data.Data(x=x, edge_index=edge_index)
g = torch_geometric.utils.to_networkx(data, to_undirected=False)
nx.draw_networkx(g, with_labels=True)
plt.show()


data = Data(x=x, edge_index=edge_index)

print(data)

data.validate(raise_on_error=True)

print(data.keys)


from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

print(len(dataset))

print(dataset.num_classes)

print(dataset[0])
