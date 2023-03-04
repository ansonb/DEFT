import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import dgl
import dgl.function as fn
import numpy as np
from scipy import sparse as sp
import math


###### ChebNet based filter ############
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
def cheb_msg(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges._edge_data[dst_field])}
    return func
def cheb_msg2(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges._edge_data[dst_field].unsqueeze(1))}
    return func
def cheb_reduce(src_field, out_field):
    # def func(msg=src_field, out=out_field):
    #     return fn.sum(msg=src_field, out=out_field)
    # return func
    return fn.sum(msg=src_field, out=out_field)
# use laplacian function from chebnet
def get_laplacian(g, norm='sym', device='cuda'):
    g = dgl.remove_self_loop(g)
    num_nodes = g.number_of_nodes()
    if g.edata.get('ew') is None:
        edge_index = g.get_edges()
        g.edata['ew'] = torch.ones(edge_index.size(1), device=device)
    row, col = g.get_edges()
    deg = g.in_degree() + g.out_degree()
    if norm is None:
        adj = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        # g = dgl.add_self_loop(g)
        # laplacian = sparse.eye(num_nodes) - adj
        laplacian = deg*sparse.eye(num_nodes) - adj
    elif norm=='sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        adj = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        ew = deg_inv_sqrt[row] * adj * deg_inv_sqrt[col]
        laplacian = sparse.eye(num_nodes) - ew
    elif norm=='rw':
        deg_inv = deg.pow_(-1.)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        adj = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        ew = deg_inv[row] * adj
        laplacian = sparse.eye(num_nodes) - ew

    return laplacian

def get_laplacian_sp(g, norm='sym'):
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    # A = g.adjacency_matrix(transpose=False, scipy_fmt="csr")
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    return L

class SGNNDynamicDgl(nn.Module):
    def __init__(self, in_feats, out_feats, K, 
        normalization='sym', bias=True, device='cuda', **kwargs):
        super(SGNNDynamicDgl, self).__init__()
        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_feats
        self.out_channels = out_feats
        self.normalization = normalization
        self.weight = Parameter(torch.Tensor(K, in_feats, out_feats))

        if bias:
            self.bias = Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.device = device

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, g, filter_coeff, batch=None, lambda_max=None, feature_name='node_embs_sgnn_in', scale=1.0):


        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        if lambda_max is None:
            lambda_max = 2.0


        if batch is not None:
            filter_coeff = torch.repeat_interleave(filter_coeff, batch, dim=1)
        weight = self.weight

        laplacian_mat = get_laplacian_sp(g)
        laplacian_mat = 2.*laplacian_mat/lambda_max - sp.eye(g.number_of_nodes())
        laplacian_mat = laplacian_mat.astype(np.float32)
        l_g = dgl.from_scipy(laplacian_mat, eweight_name='w', device=self.device)
        if torch.isnan(l_g.edata['w']).any():
            import pdb; pdb.set_trace()
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'Tx_is'` ndata below) are automatically popped out
        # when the scope exits.
        with l_g.local_scope():
            l_g.ndata[feature_name] = g.ndata[feature_name]
            l_g.ndata['Tx_0'] = g.ndata[feature_name]
            l_g.ndata['Tx_1'] = g.ndata[feature_name]  # Dummy.
            out = torch.matmul(filter_coeff[0]*l_g.ndata['Tx_0'], weight[0])

            # propagate_type: (x: Tensor, norm: Tensor)
            if weight.size(0) > 1:
                l_g.update_all(cheb_msg2(feature_name,'w','Tx_1_p'), cheb_reduce('Tx_1_p','Tx_1'))
                # g.edata['w'] stores the edge weights
                l_g.ndata['Tx_1'] = l_g.ndata['Tx_1'] * scale
                out = out + torch.matmul(filter_coeff[1]*l_g.ndata['Tx_1'], weight[1])

            for k in range(2, weight.size(0)):
                # Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
                l_g.update_all(cheb_msg2('Tx_1','w','Tx_2_p'), cheb_reduce('Tx_2_p','Tx_2'))
                # g.edata['w'] stores the edge weights
                l_g.ndata['Tx_2'] = l_g.ndata['Tx_2'] * scale
                l_g.ndata['Tx_2'] = 2. * l_g.ndata['Tx_2'] - l_g.ndata['Tx_0']
                out = out + torch.matmul(filter_coeff[k]*l_g.ndata['Tx_2'], weight[k])
                l_g.ndata['Tx_0'] = l_g.ndata['Tx_1']
                l_g.ndata['Tx_1'] = l_g.ndata['Tx_2'] 

            if self.bias is not None:
                out += self.bias

        return out