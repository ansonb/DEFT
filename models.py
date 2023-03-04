import torch
import utils as u
from argparse import Namespace
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn
import math

import dgl

from gwnn_layer import SparseGraphWaveletLayer, DenseGraphWaveletLayer
"""GWNN data reading utils."""
import json
import pygsp
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable
from sklearn.preprocessing import normalize

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to create an NX graph object.
    :param path: Path to the edge list csv.
    :return graph: NetworkX graph.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def feature_reader(path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param feature_path: Path to the JSON file.
    :return features: Feature sparse COO matrix.
    """
    features = json.load(open(path))
    index_1 = [int(k) for k, v in features.items() for fet in v]
    index_2 = [int(fet) for k, v in features.items() for fet in v]
    values = [1.0]*len(index_1)
    nodes = [int(k) for k, v in features.items()]
    node_count = max(nodes)+1
    feature_count = max(index_2)+1
    features = sparse.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    return features

def target_reader(path):
    """
    Reading thetarget vector to a numpy column vector.
    :param path: Path to the target csv.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"])
    return target

def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    with open(args.log_path, "w") as f:
        json.dump(logs, f)

class WaveletSparsifier(object):
    """
    Object to sparsify the wavelet coefficients for a graph.
    """
    def __init__(self, graph, scale, approximation_order, tolerance):
        """
        :param graph: NetworkX graph object.
        :param scale: Kernel scale length parameter.
        :param approximation_order: Chebyshev polynomial order.
        :param tolerance: Tolerance for sparsification.
        """
        self.graph = graph
        self.pygsp_graph = pygsp.graphs.Graph(nx.adjacency_matrix(self.graph))
        self.pygsp_graph.estimate_lmax()
        self.scales = [-scale, scale]
        self.approximation_order = approximation_order
        self.tolerance = tolerance
        self.phi_matrices = []

    def calculate_wavelet(self):
        """
        Creating sparse wavelets.
        :return remaining_waves: Sparse matrix of attenuated wavelets.
        """
        impulse = np.eye(self.graph.number_of_nodes(), dtype=int)
        wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.pygsp_graph,self.chebyshev,impulse)
        wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
        ind_1, ind_2 = wavelet_coefficients.nonzero()
        n_count = self.graph.number_of_nodes()
        remaining_waves = sparse.csr_matrix((wavelet_coefficients[ind_1, ind_2], (ind_1, ind_2)),shape=(n_count, n_count),dtype=np.float32)
        return remaining_waves

    def normalize_matrices(self):
        """
        Normalizing the wavelet and inverse wavelet matrices.
        """
        print("\nNormalizing the sparsified wavelets.\n")
        for i, phi_matrix in enumerate(self.phi_matrices):
            self.phi_matrices[i] = normalize(self.phi_matrices[i], norm='l1', axis=1)

    def calculate_density(self):
        """
        Calculating the density of the sparsified wavelet matrices.
        """
        wavelet_density = len(self.phi_matrices[0].nonzero()[0])/(self.graph.number_of_nodes()**2)
        wavelet_density = str(round(100*wavelet_density, 2))
        inverse_wavelet_density = len(self.phi_matrices[1].nonzero()[0])/(self.graph.number_of_nodes()**2)
        inverse_wavelet_density = str(round(100*inverse_wavelet_density, 2))
        print("Density of wavelets: "+wavelet_density+"%.")
        print("Density of inverse wavelets: "+inverse_wavelet_density+"%.\n")

    def calculate_all_wavelets(self):
        """
        Graph wavelet coefficient calculation.
        """
        print("\nWavelet calculation and sparsification started.\n")
        for i, scale in enumerate(self.scales):
            self.heat_filter = pygsp.filters.Heat(self.pygsp_graph,
                                                  tau=[scale])
            self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter,m=self.approximation_order)
            sparsified_wavelets = self.calculate_wavelet()          
            self.phi_matrices.append(sparsified_wavelets)
        self.normalize_matrices()
        self.calculate_density()

class GWNN(torch.nn.Module):
    """
    Graph Wavelet Neural Network class.
    For details see: Graph Wavelet Neural Network.
    Bingbing Xu, Huawei Shen, Qi Cao, Yunqi Qiu, Xueqi Cheng. ICLR, 2019
    :param args: Arguments object.
    :param ncount: Number of nodes.
    :param feature_number: Number of features.
    :param class_number: Number of classes.
    :param device: Device used for training.
    """
    def __init__(self, args, device):
        super(GWNN, self).__init__()
        self.args = args
        self.ncount = args.ncount
        self.feature_number = args.feats_per_node
        self.class_number = args.class_number
        self.device = device
        self.setup_layers()

    def setup_layers(self):
        """
        Setting up a sparse and a dense layer.
        """
        self.convolution_1 = SparseGraphWaveletLayer(self.feature_number,self.args.filters,self.ncount,self.device)

        # self.convolution_2 = DenseGraphWaveletLayer(self.args.filters,self.class_number,self.ncount,self.device)
        self.convolution_2 = DenseGraphWaveletLayer(self.args.filters,self.args.filters,self.ncount,self.device)

    def setup_features(self, sparsifier, features):
        """
        Defining PyTorch tensors for sparse matrix multiplications.
        """
        self.ncount = sparsifier.phi_matrices[0].shape[0]
        self.feature_number = features.shape[1]
        # self.class_number = max(target)+1
        # self.target = torch.LongTensor(self.target).to(self.device)
        # self.feature_indices = torch.LongTensor([features.row, features.col])
        self.feature_indices = features.coalesce().indices()
        # adj['idx'].t(), adj['vals'].type(torch.float)
        self.feature_indices = self.feature_indices.to(self.device)
        # self.feature_values = torch.FloatTensor(features.data).view(-1).to(self.device)
        self.feature_values = features.coalesce().values().view(-1).to(self.device)
        self.phi_indices = torch.LongTensor(sparsifier.phi_matrices[0].nonzero()).to(self.device)
        self.phi_values = torch.FloatTensor(sparsifier.phi_matrices[0][sparsifier.phi_matrices[0].nonzero()])
        self.phi_values = self.phi_values.view(-1).to(self.device)
        self.phi_inverse_indices = torch.LongTensor(sparsifier.phi_matrices[1].nonzero()).to(self.device)
        self.phi_inverse_values = torch.FloatTensor(sparsifier.phi_matrices[1][sparsifier.phi_matrices[1].nonzero()])
        self.phi_inverse_values = self.phi_inverse_values.view(-1).to(self.device)

        return self.phi_indices, self.phi_values, self.phi_inverse_indices, self.phi_inverse_values, self.feature_indices, self.feature_values

    def forward(self, phi_indices, phi_values, phi_inverse_indices,
                phi_inverse_values, feature_indices, feature_values):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param predictions: Predicted node label vector.
        """
        deep_features_1 = self.convolution_1(phi_indices,
                                             phi_values,
                                             phi_inverse_indices,
                                             phi_inverse_values,
                                             feature_indices,
                                             feature_values,
                                             self.args.dropout)

        deep_features_2 = self.convolution_2(phi_indices,
                                             phi_values,
                                             phi_inverse_indices,
                                             phi_inverse_values,
                                             deep_features_1)

        return deep_features_2
        # predictions = torch.nn.functional.log_softmax(deep_features_2, dim=1)
        # return predictions

class Sp_GCN(torch.nn.Module):
    def __init__(self,args,activation):
        super().__init__()
        self.activation = activation
        self.num_layers = args.num_layers

        self.w_list = nn.ParameterList()
        for i in range(self.num_layers):
            if i==0:
                w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                u.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                u.reset_param(w_i)
            self.w_list.append(w_i)


    def forward(self,A_list, Nodes_list, nodes_mask_list):
        node_feats = Nodes_list[-1]
        #A_list: T, each element sparse tensor
        #take only last adj matrix in time
        Ahat = A_list[-1]
        #Ahat: NxN ~ 30k
        #sparse multiplication

        # Ahat NxN
        # self.node_embs = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        return last_l


class Sp_Skip_GCN(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.W_feat = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))

    def forward(self,A_list, Nodes_list = None):
        node_feats = Nodes_list[-1]
        #A_list: T, each element sparse tensor
        #take only last adj matrix in time
        Ahat = A_list[-1]
        #Ahat: NxN ~ 30k
        #sparse multiplication

        # Ahat NxN
        # self.node_feats = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        l1 = self.activation(Ahat.matmul(node_feats.matmul(self.W1)))
        l2 = self.activation(Ahat.matmul(l1.matmul(self.W2)) + (node_feats.matmul(self.W3)))

        return l2

class Sp_Skip_NodeFeats_GCN(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)

    def forward(self,A_list, Nodes_list = None):
        node_feats = Nodes_list[-1]
        Ahat = A_list[-1]
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        skip_last_l = torch.cat((last_l,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input
        return skip_last_l

class Sp_GCN_LSTM_A(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

    def forward(self,A_list, Nodes_list = None, nodes_mask_list = None):
        last_l_seq=[]
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            #A_list: T, each element sparse tensor
            #note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)

        last_l_seq = torch.stack(last_l_seq)

        out, _ = self.rnn(last_l_seq, None)
        return out[-1]


class Sp_GCN_GRU_A(Sp_GCN_LSTM_A):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn = nn.GRU(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

class Sp_GCN_LSTM_B(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        assert args.num_layers == 2, 'GCN-LSTM and GCN-GRU requires 2 conv layers.'
        self.rnn_l1 = nn.LSTM(
                input_size=args.layer_1_feats,
                hidden_size=args.lstm_l1_feats,
                num_layers=args.lstm_l1_layers
                )

        self.rnn_l2 = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )
        self.W2 = Parameter(torch.Tensor(args.lstm_l1_feats, args.layer_2_feats))
        u.reset_param(self.W2)

    def forward(self,A_list, Nodes_list = None, nodes_mask_list = None):
        l1_seq=[]
        l2_seq=[]
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            l1 = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            l1_seq.append(l1)

        l1_seq = torch.stack(l1_seq)

        out_l1, _ = self.rnn_l1(l1_seq, None)

        for i in range(len(A_list)):
            Ahat = A_list[i]
            out_t_l1 = out_l1[i]
            #A_list: T, each element sparse tensor
            l2 = self.activation(Ahat.matmul(out_t_l1).matmul(self.w_list[1]))
            l2_seq.append(l2)

        l2_seq = torch.stack(l2_seq)

        out, _ = self.rnn_l2(l2_seq, None)
        return out[-1]


class Sp_GCN_GRU_B(Sp_GCN_LSTM_B):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn_l1 = nn.GRU(
                input_size=args.layer_1_feats,
                hidden_size=args.lstm_l1_feats,
                num_layers=args.lstm_l1_layers
               )

        self.rnn_l2 = nn.GRU(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

class Classifier(torch.nn.Module):
    def __init__(self,args,out_features=2, in_features = None):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()

        if in_features is not None:
            num_feats = in_features
        elif args.experiment_type in ['sp_lstm_A_trainer', 'sp_lstm_B_trainer',
                                    'sp_weighted_lstm_A', 'sp_weighted_lstm_B', 'spatial_transformer'] :
            num_feats = args.gcn_parameters['lstm_l2_feats'] * 2
        else:
            num_feats = args.gcn_parameters['layer_2_feats'] * 2
        print ('CLS num_feats',num_feats)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = num_feats,
                                                       out_features =args.gcn_parameters['cls_feats']),
                                       activation,
                                       torch.nn.Linear(in_features = args.gcn_parameters['cls_feats'],
                                                       out_features = out_features))

    def forward(self,x):
        return self.mlp(x)


class TimePredicter(torch.nn.Module):
    def __init__(self, args, in_features = None):
        super(TimePredicter, self).__init__()
        self.activation = torch.nn.ReLU(inplace=True)
        if in_features is not None:
            num_feats = in_features
        print ('ADAPT num_feats',num_feats)
        out_features = 1

        self.vars = nn.ParameterList()

        w = Parameter(torch.ones((num_feats, num_feats)))
        nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(num_feats)))

        w2 = Parameter(torch.ones((out_features, num_feats)))
        nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(out_features)))

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        # print(F.linear(x, vars[0]).size())
        # print(vars[0].size(), vars[1].size(), x.size())
        x = F.linear(x, vars[0], vars[1])
        x = self.activation(x)
        x = F.linear(x, vars[2], vars[3])
        return x

class Adapter(torch.nn.Module):
    def __init__(self, args, in_features = None):
        super(Adapter, self).__init__()
        self.activation = torch.nn.ReLU(inplace=True)
        if in_features is not None:
            num_feats = in_features
        print ('ADAPT num_feats',num_feats)
        out_features = num_feats

        self.vars = nn.ParameterList()

        w = Parameter(torch.ones((num_feats, num_feats)))
        nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(num_feats)))

        w2 = Parameter(torch.ones((out_features, num_feats)))
        nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(num_feats)))

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        x = self.activation(x)
        x = F.linear(x, vars[2], vars[3])
        return torch.sigmoid(x)


class StaticPNA(nn.Module):
    def __init__(self,gcn_args,activation, device='cpu', lappe=True, timepe=False):
        super().__init__()
        
        n_hid = gcn_args.hidden_dim
        n_out = gcn_args.out_dim
        self.use_2_hot_node_feats = gcn_args.use_2_hot_node_feats
        self.use_1_hot_node_feats = gcn_args.use_1_hot_node_feats
        n_inp = n_hid

        if self.use_2_hot_node_feats or self.use_1_hot_node_feats:
            self.embedding_h = nn.Embedding(gcn_args.feats_per_node, n_hid) # node feat is an integer
        else:
            self.embedding_h = nn.Linear(gcn_args.feats_per_node, n_hid)

        aggregators =  ['mean', 'max', 'sum']
        scalers = ['identity', 'amplification']
        delta = 2.5
        self.pna_layer_1 = dgl.nn.pytorch.conv.PNAConv(n_hid, n_hid, aggregators, scalers, delta, dropout=0.0, num_towers=1, edge_feat_size=0, residual=True)
        self.pna_layer_2 = dgl.nn.pytorch.conv.PNAConv(n_hid, n_hid, aggregators, scalers, delta, dropout=0.0, num_towers=1, edge_feat_size=0, residual=True)

        self.device = device

    def forward(self,A_list, Nodes_list, nodes_mask_list, graph_list=[], pos_enc_list=[]):

        if self.use_2_hot_node_feats or self.use_1_hot_node_feats:
            node_feats = Nodes_list[-1]._indices()[1,:]
        else:
            node_feats = Nodes_list[-1]
        last_l = self.embedding_h(node_feats)

        # Consider the graph containing edges from latest batch of timesteps
        g = graph_list[-1]

        last_l = self.pna_layer_1(g, last_l)
        last_l = self.pna_layer_2(g, last_l)

        return last_l


class StaticSAGE(nn.Module):
    def __init__(self,gcn_args,activation, device='cpu', lappe=True, timepe=False):
        super().__init__()
        
        n_hid = gcn_args.hidden_dim
        n_out = gcn_args.out_dim
        self.use_2_hot_node_feats = gcn_args.use_2_hot_node_feats
        self.use_1_hot_node_feats = gcn_args.use_1_hot_node_feats
        n_inp = n_hid

        if self.use_2_hot_node_feats or self.use_1_hot_node_feats:
            self.embedding_h = nn.Embedding(gcn_args.feats_per_node, n_hid) # node feat is an integer
        else:
            self.embedding_h = nn.Linear(gcn_args.feats_per_node, n_hid)
            
        aggregator_type = 'pool'
        # aggregator_type = 'mean'
        self.sage_layer_1 = dgl.nn.pytorch.conv.SAGEConv(n_hid, n_hid, aggregator_type, feat_drop=0.0, bias=True, norm=None, activation=None)
        self.sage_layer_2 = dgl.nn.pytorch.conv.SAGEConv(n_hid, n_hid, aggregator_type, feat_drop=0.0, bias=True, norm=None, activation=None)

        self.device = device

    def forward(self,A_list, Nodes_list, nodes_mask_list, graph_list=[], pos_enc_list=[]):

        if self.use_2_hot_node_feats or self.use_1_hot_node_feats:
            node_feats = Nodes_list[-1]._indices()[1,:]
        else:
            node_feats = Nodes_list[-1]
        last_l = self.embedding_h(node_feats)

        # Consider the graph containing edges from latest batch of timesteps
        g = graph_list[-1]

        last_l = self.sage_layer_1(g, last_l)
        last_l = self.sage_layer_2(g, last_l)

        return last_l