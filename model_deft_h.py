import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from sgnn_dynamic import SGNNDynamic
# from sgnn_dynamic_dgl import SGNNDynamicDgl

import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

class DEFT(torch.nn.Module):
    def __init__(self, transformer_args, gcn_args, activation, device='cpu', skipfeats=False, data=''):
        super().__init__()
        RT_args = u.Namespace({})

        feats = [gcn_args.feats_per_node,
                 gcn_args.layer_1_feats,
                 gcn_args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.RT_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            RT_args = u.Namespace({
                'in_feats' : feats[i-1],
                'out_feats1': feats[i],
                'sgnn_in_feats': feats[i-1] if i==1 else transformer_args.out_dim,
                'activation': activation,
                'filter_order': transformer_args.filter_order,
                'in_channels_sgnn': feats[i] if i==1 else transformer_args.out_dim,
                'out_channels_sgnn': transformer_args.out_channels_sgnn,
                'fc1_dim': transformer_args.fc1_dim,
                'pe_dim': transformer_args.pe_dim,
                'out_feats': transformer_args.out_dim,
                'num_heads': transformer_args.num_heads,
                'layer_norm': transformer_args.layer_norm,
                'batch_norm': transformer_args.batch_norm,
                'is_recurrent': transformer_args.is_recurrent,
                'sgwt_scales': transformer_args.sgwt_scales,
                'device': device,
                'use_transformer': transformer_args.use_transformer,
                'concat_in_skipfeat': transformer_args.concat_in_skipfeat,
                'rt_residual': transformer_args.rt_residual,
                'skip_in_feat': transformer_args.skip_in_feat,
                'use_spatial_feat_in_lpe': transformer_args.use_spatial_feat_in_lpe,
                'use_spectral_in_lpe': transformer_args.use_spectral_in_lpe,   
                'num_filter_subspaces': transformer_args.num_filter_subspaces,  
                'use_spatial_feat_in_rgt_ip': transformer_args.use_spatial_feat_in_rgt_ip,  
                'skip_rgt_in_feat': transformer_args.skip_rgt_in_feat,  
                'device': device,   
                'aggregator': transformer_args.aggregator,  
                'use_static_spectral_wavelets': transformer_args.use_static_spectral_wavelets,  
                'use_sgnn_dgl':  transformer_args.use_sgnn_dgl, 
                'data': data
            })

            rt_i = DEFTLayer(RT_args)
            self.RT_layers.append(rt_i.to(self.device))
            self._parameters.extend(list(self.RT_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list, graph_list, A_list_u, pos_enc_list=[]):
        assert len(graph_list)==len(A_list), "DEFT model needs a list of DGL graphs of the same size of A_list to be provided."

        node_feats= Nodes_list[-1]

        for unit in self.RT_layers:
            Nodes_list = unit(A_list,A_list_u,graph_list,Nodes_list,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   
        return out


class DEFTLayer(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

        cell_args = u.Namespace({})
        cell_args.rows = args.sgnn_in_feats
        cell_args.cols = args.out_feats1

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation

        self.SGNN_init_weights = Parameter(torch.Tensor(self.args.sgnn_in_feats,self.args.out_feats1))
        if self.args.use_static_spectral_wavelets:  
            self.filter_coeff_list = nn.ParameterList([Parameter(torch.Tensor(self.args.filter_order,)) for _ in range(self.args.num_filter_subspaces)])    
        else:   
            self.SGNN_proj_weights_fc_list = nn.ParameterList([Parameter(torch.Tensor(self.args.out_feats1,self.args.filter_order)) for _ in range(self.args.num_filter_subspaces)])


        if self.args.use_spectral_in_lpe:  
            if self.args.use_spatial_feat_in_lpe and not self.args.use_spatial_feat_in_rgt_ip:
                self.FFN_pe = nn.Linear(args.out_feats+args.out_feats1+1, args.pe_dim)
            else:
               self.FFN_pe = nn.Linear(args.out_feats+1, args.pe_dim) 
        else:
            if self.args.use_spatial_feat_in_lpe:
                self.FFN_pe = nn.Linear(args.out_feats1+1, args.pe_dim)

        self.activation_pe = torch.nn.RReLU()
        self.FFN_pe2 = nn.Linear(args.pe_dim*2, args.out_feats1)
        self.FFN_pe3 = nn.Linear(args.out_feats1, args.out_feats1)

        in_channels = args.in_channels_sgnn
        out_channels = args.out_channels_sgnn
        filter_order = args.filter_order
        if self.args.use_sgnn_dgl:  
            self.sgnn_list = nn.ModuleList([SGNNDynamicDgl(in_channels, out_channels, filter_order, device=self.args.device) for _ in range(self.args.num_filter_subspaces)])   
        else:   
            self.sgnn_list = nn.ModuleList([SGNNDynamic(in_channels, out_channels, filter_order) for _ in range(self.args.num_filter_subspaces)])

        self.fc_pool = ['sum','mean'][1]
        self.fc1 = nn.Linear(self.args.filter_order, self.args.fc1_dim)
        self.fc2 = nn.Linear(self.args.fc1_dim, self.args.filter_order)
        self.reset_param(self.SGNN_init_weights)
        if self.args.use_static_spectral_wavelets:  
            for filter_coeff in self.filter_coeff_list: 
                self.reset_param_fc(filter_coeff)   
        else:   
            for SGNN_proj_weights_fc in self.SGNN_proj_weights_fc_list: 
                self.reset_param(SGNN_proj_weights_fc)


        in_dim_rgt = self.args.out_feats
        out_dim_rgt = self.args.out_feats
        num_heads = self.args.num_heads
        layer_norm = self.args.layer_norm
        batch_norm = self.args.batch_norm
        is_recurrent = self.args.is_recurrent
        self.FFN_rtg_in1 = nn.Linear(self.args.out_feats1+self.args.pe_dim, in_dim_rgt)
        if self.args.use_transformer:
            self.rgt_layer = RecGraphTransformerLayer(in_dim_rgt, out_dim_rgt, num_heads, dropout=0.0, layer_norm=layer_norm, batch_norm=batch_norm, residual=True, use_bias=False, use_state_vectors=is_recurrent, tied_weights=True)
        elif self.args.aggregator is not '':    
            if self.args.aggregator=='GAT': 
                assert out_dim_rgt%num_heads==0, "[GAT] Expected out dimension must be equal to num_heads*dim_per_head" 
                out_dim_gat = out_dim_rgt//num_heads    
                self.rgt_layer = dgl.nn.pytorch.conv.GATConv(in_dim_rgt, out_dim_gat, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True)  
            elif self.args.aggregator=='GATv2': 
                assert out_dim_rgt%num_heads==0, "[GATv2] Expected out dimension must be equal to num_heads*dim_per_head"   
                out_dim_gat = out_dim_rgt//num_heads    
                self.rgt_layer = dgl.nn.pytorch.conv.GATv2Conv(in_dim_rgt, out_dim_gat, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True, share_weights=False)   
            elif self.args.aggregator=='PNA':   
                if self.args.data.lower()=='reddit':    
                    aggregators =  ['max']  
                    scalers = ['identity']  
                else:   
                    aggregators =  ['mean', 'max', 'sum']   
                    scalers = ['identity', 'amplification'] 
                delta = 2.5 
                self.rgt_layer = dgl.nn.pytorch.conv.PNAConv(in_dim_rgt, out_dim_rgt, aggregators, scalers, delta, dropout=0.0, num_towers=1, edge_feat_size=0, residual=True)  
            elif self.args.aggregator=='GraphSAGE': 
                aggregator_type = 'pool'    
                self.rgt_layer = dgl.nn.pytorch.conv.SAGEConv(in_dim_rgt, out_dim_rgt, aggregator_type, feat_drop=0.0, bias=True, norm=None, activation=None)   
            elif self.args.aggregator=='GIN':   
                lin = torch.nn.Linear(in_dim_rgt, out_dim_rgt)  
                activation = torch.nn.functional.relu   
                self.rgt_layer = dgl.nn.pytorch.conv.GINConv(apply_func=lin, aggregator_type='sum', init_eps=0, learn_eps=False, activation=activation) 
            elif self.args.aggregator=='HGT':   
                assert out_dim_rgt%num_heads==0, "[HGT] Expected out dimension must be equal to num_heads*dim_per_head" 
                head_size = out_dim_rgt//num_heads  
                num_ntypes = 1  
                num_etypes = 1  
                self.rgt_layer = dgl.nn.pytorch.conv.HGTConv(in_dim_rgt, head_size, num_heads, num_ntypes, num_etypes, dropout=0.2, use_norm=False)
        if self.args.concat_in_skipfeat:
            self.proj_inp_rgt = nn.Linear(self.args.out_feats1, self.args.out_feats)
            self.FFN_skipcat1 = nn.Linear(self.args.out_feats1+self.args.out_feats, self.args.out_feats)
            self.FFN_skipcat2 = nn.Linear(self.args.out_feats1+self.args.out_feats, self.args.out_feats)
        self.rt_residual = self.args.rt_residual

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def reset_param_fc(self,t): 
        # TODO: try initializing with all pass filter (all ones)?
        #Initialize based on the number of columns  
        stdv = 1. / math.sqrt(t.size(0))    
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,A_list_u,G_list,node_embs_list,mask_list,residual=True):
        residual = self.rt_residual

        SGNN_weights = self.SGNN_init_weights
        out_seq = []
        seq_len = len(A_list)
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            if len(A_list_u[t]['idx'].shape)==3:
                edge_index = A_list_u[t]['idx'][0].T.to(self.args.device)
            elif len(A_list_u[t]['idx'].shape)==2:
                edge_index = A_list_u[t]['idx'].T.to(self.args.device)
            else:
                raise Exception("Unable to get edge index")
            graph = G_list[t] 

            #first evolve the weights from the initial and use the new weights with the node_embs
            if self.args.use_static_spectral_wavelets:  
                pass    
            else:   
                SGNN_weights = self.evolve_weights(SGNN_weights,node_embs,mask_list[t])
            gnn_out0 = self.activation(Ahat.matmul(node_embs.matmul(SGNN_weights)))

            node_embs_in1 = node_embs.matmul(SGNN_weights)

            if self.args.use_spectral_in_lpe:
                if self.args.use_static_spectral_wavelets:  
                    filter_coeff_list = self.filter_coeff_list  
                else:   
                    filter_coeff_list = []  
                    for SGNN_proj_weights_fc in self.SGNN_proj_weights_fc_list: 
                        gnn_out1 = self.activation(Ahat.matmul(node_embs.matmul(SGNN_weights.matmul(SGNN_proj_weights_fc))))    
                        if self.fc_pool=='mean':    
                            filter_coeff = torch.mean(gnn_out1, dim=0)  
                        elif self.fc_pool=='sum':   
                            filter_coeff = torch.sum(gnn_out1, dim=0)   
                        filter_coeff = self.activation(self.fc1(filter_coeff))  
                        filter_coeff = self.fc2(filter_coeff)   
                        filter_coeff_list.append(filter_coeff)
                node_embs = node_embs.matmul(SGNN_weights)

                all_node_embs = []
                for fc_idx, sgnn in enumerate(self.sgnn_list):  
                    filter_coeff = filter_coeff_list[fc_idx]    
                    for scale in self.args.sgwt_scales: 
                        if self.args.use_sgnn_dgl:  
                            graph.ndata['node_embs_sgnn_in'] = node_embs    
                            node_embs_cur_scale = sgnn(graph, filter_coeff, feature_name='node_embs_sgnn_in', scale=scale) * (1./max(1,scale)**(self.args.filter_order))    
                        else:   
                            node_embs_cur_scale = sgnn(node_embs, edge_index, filter_coeff, scale=scale) * (1./max(1,scale)**(self.args.filter_order))  
                        all_node_embs.append(node_embs_cur_scale)
                all_node_embs = torch.cat([ne.unsqueeze(1) for ne in all_node_embs], dim=1)
                node_embs = torch.sum(all_node_embs, dim=1)
                if self.args.use_spatial_feat_in_rgt_ip:    
                    node_embs = node_embs + gnn_out0

            if t==0:
                state_vectors = gnn_out0 
            time_pe = (1./seq_len)*torch.ones((node_embs.shape[0],1)).to(self.args.device)
            if not self.args.use_spectral_in_lpe and not self.args.use_spatial_feat_in_lpe:
                node_embs_in_rgt = node_embs_in1
            else:
                if self.args.use_spectral_in_lpe:
                    if self.args.use_spatial_feat_in_lpe and not self.args.use_spatial_feat_in_rgt_ip:
                        pe = torch.cat((node_embs,gnn_out0,time_pe), dim=1)
                    else:
                        pe = torch.cat((node_embs,time_pe), dim=1)
                else:
                    if self.args.use_spatial_feat_in_lpe:
                        pe = torch.cat((gnn_out0,time_pe), dim=1)
                pe = self.FFN_pe(pe)
                pe_cos = torch.cos(pe)
                pe_sin = torch.sin(pe)
                pe = torch.cat((pe_cos,pe_sin), dim=1)
                pe = self.activation_pe(self.FFN_pe2(pe))
                pe = self.FFN_pe3(pe)
                if self.args.use_spectral_in_lpe:
                    if self.args.concat_in_skipfeat:
                        node_embs = torch.cat((node_embs_in1,node_embs), dim=-1)
                        node_embs_in_rgt = self.FFN_skipcat1(node_embs)
                    else:
                        node_embs_in_rgt = node_embs_in1 + node_embs
                else:
                    node_embs_in_rgt = node_embs_in1
                node_embs_in_rgt = torch.cat((node_embs_in_rgt,pe), dim=1)
                node_embs_in_rgt = self.FFN_rtg_in1(node_embs_in_rgt)
            if self.args.use_transformer:
                node_embs, state_vectors = self.rgt_layer(graph, node_embs_in_rgt, state_vectors)
            elif self.args.aggregator!='':  
                if self.args.aggregator=='GAT': 
                    node_embs = self.rgt_layer(graph, node_embs_in_rgt) 
                    node_embs = node_embs.reshape((node_embs.shape[0],-1))  
                elif self.args.aggregator=='GATv2': 
                    node_embs = self.rgt_layer(graph, node_embs_in_rgt) 
                    node_embs = node_embs.reshape((node_embs.shape[0],-1))  
                elif self.args.aggregator=='PNA':   
                    node_embs = self.rgt_layer(graph, node_embs_in_rgt) 
                elif self.args.aggregator=='GraphSAGE': 
                    node_embs = self.rgt_layer(graph, node_embs_in_rgt) 
                elif self.args.aggregator=='GIN':   
                    node_embs = self.rgt_layer(graph, node_embs_in_rgt) 
                elif self.args.aggregator=='HGT':   
                    ntype = torch.zeros((graph.num_nodes(),),dtype=int).to(self.args.device)    
                    etype = torch.zeros((graph.num_edges(),),dtype=int).to(self.args.device)    
                    node_embs = self.rgt_layer(graph, node_embs_in_rgt, ntype, etype)
            else:
                node_embs = node_embs_in_rgt

            if residual:
                if self.args.concat_in_skipfeat:
                    node_embs = torch.cat((node_embs,gnn_out0), dim=-1)
                    node_embs = self.FFN_skipcat2(node_embs)
                elif self.args.skip_in_feat:    
                    node_embs = node_embs + node_embs_in1
                elif self.args.skip_rgt_in_feat:    
                    node_embs = node_embs + node_embs_in_rgt
                else:
                    node_embs = node_embs + gnn_out0

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q,prev_Z,mask):
        z_topk = self.choose_topk(prev_Z,mask)

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()


"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
    # TODO: handle weighted edges if needed
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, query, key, value):
        
        Q_h = self.Q(query)
        K_h = self.K(key)
        V_h = self.V(value)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        # TODO: check if same key names give issue during backprop (hopefully should not)
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        head_out = g.ndata['wV']/g.ndata['z']
        
        return head_out
    

class RecGraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False, use_state_vectors=True, tied_weights=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        self.use_state_vectors = use_state_vectors
        self.tied_weights = tied_weights

        self.self_attention_v = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        if use_state_vectors:
            if self.tied_weights:
                self.cross_attention_v = self.self_attention_v
                self.self_attention_h = self.self_attention_v
                self.cross_attention_h = self.self_attention_v
            else:
                self.cross_attention_v = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
                self.self_attention_h = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
                self.cross_attention_h = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
            
        if use_state_vectors:
            self.O = nn.Linear(out_dim*2, out_dim)
        else:
            self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

        if use_state_vectors:
            if self.layer_norm:
                self.layer_norm1_h = nn.LayerNorm(out_dim)
            if self.batch_norm:
                self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            if self.layer_norm:
                self.layer_norm2_h = nn.LayerNorm(out_dim)
            if self.batch_norm:
                self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.O_h = nn.Linear(out_dim*2, out_dim)
            self.FFN_layer1_h = nn.Linear(out_dim, out_dim*2)
            self.FFN_layer2_h = nn.Linear(out_dim*2, out_dim)
            self.FFN_gate1 = nn.Linear(in_dim, out_dim, bias=True)
            self.FFN_gate2 = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, g, h, state_vectors=None):
        
        # Usual Vertical flow
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        attn_out_self = self.self_attention_v(g, h, h, h)
        if self.use_state_vectors:
            # TODO: add state ids/pos embeddings to state vectors?
            attn_out_cross = self.cross_attention_v(g, h, state_vectors, state_vectors)
            attn_out_self = attn_out_self.view(-1, self.out_channels)
            attn_out_cross = attn_out_cross.view(-1, self.out_channels)
            h = torch.cat((attn_out_self,attn_out_cross), dim=-1)
        else:
            attn_out = attn_out_self
            h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       


        if self.use_state_vectors:
            # Horizontal flow
            sv_in1 = state_vectors # for first residual connection
            
            # multi-head attention out
            attn_out_self_h = self.self_attention_h(g, state_vectors, state_vectors, state_vectors)
            # TODO: add state ids/pos embeddings to state vectors?
            attn_out_cross_h = self.cross_attention_v(g, state_vectors, h_in1, h_in1)
            attn_out_self_h = attn_out_self_h.view(-1, self.out_channels)
            attn_out_cross_h = attn_out_cross_h.view(-1, self.out_channels)
            sv = torch.cat((attn_out_self_h,attn_out_cross_h), dim=-1)
            
            sv = F.dropout(sv, self.dropout, training=self.training)
            
            sv = self.O_h(sv)
            
            if self.residual:
                # TODO: try with lstm gate?
                g1 = self.FFN_gate1(h_in1)
                g1 = nn.Sigmoid()(g1)
                sv = (1-g1)*sv_in1 + g1*sv # residual connection with fixed gate
            
            if self.layer_norm:
                sv = self.layer_norm1_h(sv)
                
            if self.batch_norm:
                sv = self.batch_norm1_h(sv)
            
            sv_in2 = sv # for second residual connection
            
            # FFN
            sv = self.FFN_layer1_h(sv)
            sv = F.relu(sv)
            sv = F.dropout(sv, self.dropout, training=self.training)
            sv = self.FFN_layer2_h(sv)

            if self.residual:
                g2 = self.FFN_gate2(h_in1)
                g2 = nn.Sigmoid()(g2)
                sv = (1-g2)*sv_in2 + g2*sv # residual connection with fixed gate
            
            if self.layer_norm:
                sv = self.layer_norm2_h(sv)
                
            if self.batch_norm:
                sv = self.batch_norm2_h(sv)     

        if self.use_state_vectors:
            return h, sv
        else:
            return h, None
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)