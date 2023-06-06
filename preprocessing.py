import numpy as np
import scipy.sparse as sp
from torch_geometric.data import InMemoryDataset, Batch
import torch
from torch_geometric import data as DATA

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


class PFPDataset(InMemoryDataset):
    def __init__(self, dir=None, train_data_X=None,train_data_Y=None,transform=None,pre_transform=None):

        super(PFPDataset, self).__init__( transform, pre_transform)
        self.dir=dir
        self.X_data_list = train_data_X
        self.Y_data_list = train_data_Y

    def __len__(self):
        return int(self.X_data_list.shape[0])

    def __getitem__(self, idx):
        embedding = self.X_data_list[idx]
        label = self.Y_data_list[idx]
        embedding = torch.Tensor([embedding])
        label = torch.Tensor([label])
        # pro_name = dict_item.get('target')
        # ligand = dict_item.get('ligand')
        # label = dict_item.get('label')

        # for i in range(site_3D.shape[0]):
        #     for j in range(site_3D.shape[1]):
        #         for k in range(site_3D.shape[2]):
        #             if site_3D[i][j][k][1] > 0:
        #                 print(site_3D[i][j][k][1])
        #                 print('finish!')

        # GCNData_mol = DATA.Data(x=torch.Tensor(features),
        #                         edge_index=torch.LongTensor(edge_index).transpose(1, 0),
        #                         y=torch.FloatTensor([label]))
        # GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
        # GCNData_mol.__setitem__('smiles', ligand)
        # GCNData_mol.__setitem__('pro', pro_name)


        # site_3D_Features=torch.Tensor([site_3D])
        # seq_coding_features=torch.Tensor([seq_coding])

        return embedding,label

def collate(data_list):
    embedding = [data[0] for data in data_list]
    label = [data[1] for data in data_list]
    embedding_list = torch.stack(embedding).squeeze(dim=1)
    label_list = torch.stack(label).squeeze(dim=1)
    return embedding_list,label_list
    # return data_list
    # mol_data_list=[data[0] for data in data_list]
    # batchA = Batch.from_data_list(mol_data_list)
    # site_3D_list=[data[1] for data in data_list]
    # site_3D_list_tensor=torch.stack(site_3D_list).squeeze(dim=1)
    # Seq_1D_list=[data[2].long() for data in data_list]
    # Seq_1D_list_tensor=torch.stack(Seq_1D_list).squeeze(dim=1)
    # return batchA, site_3D_list_tensor,Seq_1D_list_tensor