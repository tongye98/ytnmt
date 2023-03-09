import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import  scatter_max
import re 

def test1():
    class SmallNet(torch.nn.Module):
        def __init__(self):
            super(SmallNet, self).__init__()
            self.conv1 = GCNConv(2, 4)
            self.linear1 = torch.nn.Linear(4,3)

        def forward(self, data):
            x, edge_index, token = data.node, data.node_edge, data.token
            print("x = {}".format(x))
            print("edge = {}".format(edge_index))
            print("token = {} shape = {}".format(token, token.shape))
            assert False
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x, _ = scatter_max(x, data.batch, dim=0)
            x = self.linear1(x)
            return x

    class Datap(Data):
        def __init__(self, node=None, node_edge=None, token=None, label=None):
            super().__init__()
            self.node = node
            self.token = token
            self.node_edge = node_edge
            self.label = label 
            # self.__num_nodes__ = node.size(0)
        
        # 自定义拼接步长
        def __inc__(self, key, value, *args, **kwargs):  # 增量，针对edge_index
            if key == 'node_edge':
                return self.node.size(0)
            else:
                return 0

        def __cat_dim__(self, key, value, *args, **kwargs):
            if bool(re.search('node_edge', key)):
                return -1
            elif bool(re.search('node',key)):
                return 0
            elif bool(re.search('token', key)):
                return None 
            else:
                super().__cat_dim__(key, value, *args, **kwargs)


    def init_data():
        labels=np.array([0,1,2],dtype=int)
        a=labels[0]
        data_list = []
        
        #定义第一个节点的信息
        x = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ])
        x = torch.tensor(x, dtype=torch.float)
        edge = np.array([
            [0, 0, 2],
            [1, 2, 0]
        ])
        edge = torch.tensor(edge, dtype=torch.long)
        token = torch.tensor([11,12,13,14], dtype=torch.long)
        data_list.append(Datap(node=x, node_edge=edge.contiguous(), token=token, label=int(labels[0])))

        #定义第二个节点的信息
        x = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ])
        x = torch.tensor(x, dtype=torch.float)
        edge = np.array([
            [0, 1],
            [1, 2]
        ])
        edge = torch.tensor(edge, dtype=torch.long)
        token = torch.tensor([21,22,23,24], dtype=torch.long)
        data_list.append(Datap(node=x, node_edge=edge.contiguous(), token=token,label=int(labels[1])))

        #定义第三个节点的信息
        x = np.array([
            [0, 0],
        [1, 1],
            [2, 2]
        ])
        x = torch.tensor(x, dtype=torch.float)
        edge = np.array([
            [0, 1, 2],
            [2, 2, 0]
        ])
        edge = torch.tensor(edge, dtype=torch.long)
        token = torch.tensor([31,32,33,34], dtype=torch.long)
        data_list.append(Datap(node=x, node_edge=edge.contiguous(), token=token, label=int(labels[2])))
        return data_list

    epoch_num=10000
    batch_size=2
    trainset=init_data()
    #NOTE 
    item0 = trainset[0]
    print(item0)
    print('edge_attr' in item0)
    print(item0.num_nodes)
    print(item0.num_node_features)
    print("is directed = {}".format(item0.is_direceted()))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    device = torch.device('cpu')
    model = SmallNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        train_loss = 0.0
        for i, batch in enumerate(trainloader):
            #print("label = {}".format(batch.label))
            batch = batch.to("cpu")
            optimizer.zero_grad()
            outputs = model(batch)
            #print(outputs)
            #print(batch.label)
            loss = criterion(outputs, batch.label)
            #print("loss = {}".format(loss))
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            # print('epoch: {:d} loss: {:.3f}'
            #       .format(epoch + 1, loss.cpu().item()))
        print('epoch: {:d} loss: {:.3f}'
            .format(epoch + 1, train_loss / batch_size))

def test2():
    scores = torch.tensor([1,2,3,4])
    mask = torch.BoolTensor([1,1,0,0])
    scores = scores.masked_fill(~mask, value=-100)
    print(scores)
    return None

def test3():
    size = 4
    ones = torch.ones(size,size, dtype=torch.bool)
    subsequence_mask = torch.tril(ones, out=ones).unsqueeze(0)
    print(subsequence_mask)

    trg_mask = torch.tensor([[1,1,1,0],[1,0,0,0],[1,1,0,0]]).view(3,1,4)

    mask = subsequence_mask & trg_mask
    print(mask)


from torch_geometric.utils import to_dense_batch
def test4():
    x = torch.arange(18).view(6,3)
    print("x={}".format(x))

    batch = torch.tensor([0,0,1,2,2,2])
    out, mask = to_dense_batch(x, batch, max_num_nodes=3)
    print("out = {}".format(out))
    print("mask = {}".format(mask))

if __name__ == "__main__":
    test3()