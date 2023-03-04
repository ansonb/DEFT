import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import dgl
import math
from graph_transformer_layer import GraphTransformerLayer
from mlp_readout_layer import MLPReadout
from models import Sp_GCN
from tst import Transformer as TemporalTransformer

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params, device='cpu'):
        super().__init__()

        # in_dim_node = net_params.in_dim # node_dim (feat is an integer)
        in_dim_node = net_params.feats_per_node
        hidden_dim = net_params.hidden_dim
        out_dim = net_params.out_dim
        n_classes = net_params.n_classes
        num_heads = net_params.n_heads
        in_feat_dropout = net_params.in_feat_dropout
        dropout = net_params.dropout
        n_layers = net_params.L

        self.readout = net_params.readout
        self.layer_norm = net_params.layer_norm
        self.batch_norm = net_params.batch_norm
        self.residual = net_params.residual
        self.dropout = dropout
        self.n_classes = n_classes
        # self.device = net_params.device
        self.device = device
        self.lap_pos_enc = net_params.lap_pos_enc
        self.wl_pos_enc = net_params.wl_pos_enc
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params.pos_enc_dim
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.use_2_hot_node_feats = net_params.use_2_hot_node_feats
        self.use_1_hot_node_feats = net_params.use_1_hot_node_feats
        if net_params.use_2_hot_node_feats or net_params.use_1_hot_node_feats:
            self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        else:
            self.embedding_h = nn.Linear(in_dim_node, hidden_dim) # node feat is an integer
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                              dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual))
        # self.MLP_layer = MLPReadout(out_dim, n_classes)


    def forward(self, g, h, e=None, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)
        
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
            
        # output
        # h_out = self.MLP_layer(h)

        return h_out
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss




class Transformer_LSTM(GraphTransformerNet):
    def __init__(self,transformer_args,gcn_args,activation, device='cpu', lappe=True, timepe=False):
        super().__init__(transformer_args,device)
        # TODO: Define net params
        # conv = GraphTransformerNet(args)
        self.rnn = nn.LSTM(
                # input_size=gcn_args.layer_2_feats,
                input_size=transformer_args.out_dim,
                hidden_size=gcn_args.lstm_l2_feats,
                num_layers=gcn_args.lstm_l2_layers
                )

    # TODO: handle if Nodes_list is None
    # TODO: check A_list (and Ahat)
    # def forward(self,A_list, Nodes_list = None, nodes_mask_list = None):
    def forward(self,A_list, Nodes_list, nodes_mask_list = None, graph_list=[], pos_enc_list=[]):
        last_l_seq=[]
        for t,Ahat in enumerate(A_list):
            # if isinstance(Nodes_list[t],torch.Tensor):
            #     node_feats = Nodes_list[t]
            # else:
            #     node_feats = Nodes_list[t]._indices()[1,:]
            if self.use_2_hot_node_feats or self.use_1_hot_node_feats:
                node_feats = Nodes_list[t]._indices()[1,:]
            else:
                node_feats = Nodes_list[t]
            # A_list: T, each element sparse tensor
            last_l = self.embedding_h(node_feats)
            if self.lap_pos_enc:
                pos_enc = pos_enc_list[t]
                pos_enc_emb = self.embedding_lap_pos_enc(pos_enc)
                last_l = last_l + pos_enc_emb
            # GraphTransformer Layers
            # TODO: make dgl graph from Ahat?
            if len(graph_list)==0:
                g = Ahat
            else:
                g = graph_list[t]
            for conv in self.layers:
                last_l = conv(g, last_l)
            last_l_seq.append(last_l)

        last_l_seq = torch.stack(last_l_seq)

        out, _ = self.rnn(last_l_seq, None)
        return out[-1]

class GCN_Transformer(Sp_GCN):
    def __init__(self,transformer_args,gcn_args,activation, device='cpu', lappe=False, timepe=True):
        super().__init__(gcn_args,activation)

        # TODO: handle custom time position encoding
        self.time_transformer = TemporalTransformer(gcn_args.layer_2_feats, transformer_args.tst_l2_feats, transformer_args.tst_l2_feats, transformer_args.tst_l2_feats, transformer_args.tst_l2_feats, transformer_args.tst_l2_feats, transformer_args.tst_l2_layers, chunk_mode=None, pe='original', pe_period=None, use_decoder=transformer_args.tst_use_decoder)
        # , pe_period=transformer_args.pe_period)

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

        # TODO: check/test
        last_l_seq = last_l_seq.permute((1,0,2))
        out = self.time_transformer(last_l_seq)
        out = out.permute((1,0,2))
        return out[-1]

class Spatio_Temporal_Transformer(GraphTransformerNet):
    def __init__(self, transformer_args, gcn_args, activation, device='cpu', lappe=True, timepe=True):
        # super().__init__()
        super().__init__(transformer_args,device)
        self.time_transformer = TemporalTransformer(transformer_args.out_dim, transformer_args.tst_l2_feats, transformer_args.tst_l2_feats, transformer_args.tst_l2_feats, transformer_args.tst_l2_feats, transformer_args.tst_l2_feats, transformer_args.tst_l2_layers, chunk_mode=None, pe='original', pe_period=None, use_decoder=transformer_args.tst_use_decoder)


    # def parameters(self):
    #     return self._parameters

    def forward(self,A_list, Nodes_list, nodes_mask_list = None, graph_list=[], pos_enc_list=[]):
        last_l_seq=[]
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]._indices()[1,:]
            # A_list: T, each element sparse tensor
            last_l = self.embedding_h(node_feats)
            if self.lap_pos_enc:
                pos_enc = pos_enc_list[t]
                pos_enc_emb = self.embedding_lap_pos_enc(pos_enc)
                last_l = last_l + pos_enc_emb
            # GraphTransformer Layers
            # TODO: make dgl graph from Ahat?
            if len(graph_list)==0:
                g = Ahat
            else:
                g = graph_list[t]
            for conv in self.layers:
                last_l = conv(g, last_l)
            last_l_seq.append(last_l)

        last_l_seq = torch.stack(last_l_seq)

        last_l_seq = last_l_seq.permute((1,0,2))
        out = self.time_transformer(last_l_seq)
        out = out.permute((1,0,2))
        return out[-1]

class StaticGraphTransformer(GraphTransformerNet):
    def __init__(self,transformer_args,gcn_args,activation, device='cpu', lappe=True, timepe=False):
        super().__init__(transformer_args, device)


    def forward(self,A_list, Nodes_list, nodes_mask_list, graph_list=[], pos_enc_list=[]):

        if self.use_2_hot_node_feats or self.use_1_hot_node_feats:
            node_feats = Nodes_list[-1]._indices()[1,:]
        else:
            node_feats = Nodes_list[-1]
        # A_list: T, each element sparse tensor
        last_l = self.embedding_h(node_feats)
        if self.lap_pos_enc:
            pos_enc = pos_enc_list[-1]
            pos_enc_emb = self.embedding_lap_pos_enc(pos_enc)
            last_l = last_l + pos_enc_emb
        # GraphTransformer Layers
        # TODO: make dgl graph from Ahat?
        if len(graph_list)==0:
            g = Ahat
        else:
            g = graph_list[-1]
        for conv in self.layers:
            last_l = conv(g, last_l)


        return last_l








