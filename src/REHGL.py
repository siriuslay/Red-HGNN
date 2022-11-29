import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from memory_profiler import profile
from util_funcs import cos_sim


class REHGL(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, cf, g):
        super(REHGL, self).__init__()
        self.__dict__.update(cf.get_model_conf())
        # ! Init variables
        self.dev = cf.dev
        self.ti, self.ri, self.types, self.ud_rels = g.t_info, g.r_info, g.types, g.undirected_relations
        self.target_type, self.target_nodes_num = cf.target_type, g.target_nodes_num
        feat_dim, mp_emb_dim = g.features.shape[1], list(g.mp_emb_dict.values())[0].shape[1]
        self.non_linear = nn.ReLU()
        # ! Graph Structure Learning
        MD = nn.ModuleDict
        self.fgg_hidden, self.fgg_origin, self.fgg_topo, self.sg_agg, self.f_agg, self.fp_origin, self.fp_feat, self.sgg_gen, self.topo_encoder = \
            MD({}), MD({}), MD({}), MD({}), MD({}), MD({}), MD({}), MD({}), MD({})

        # Feature encoder
        self.encoder = MD(dict(zip(g.types, [nn.Linear(g.features.shape[1], cf.com_feat_dim) for _ in g.types])))

        # ! Feature Graph Generator
        self.fgg_hidden = GraphGenerator(cf.com_feat_dim, cf.num_head, cf.fgh_th, self.dev)  # dim = 16
        self.fgg_origin = GraphGenerator(feat_dim, cf.num_head, cf.fgh_th, self.dev)  # dim = 82
        self.fp_origin = GraphGenerator(feat_dim, cf.num_head, cf.fgp_th, self.dev)  # dim = 82

        # ! Semantic Graph Generator
        self.sgg_gen = MD(dict(
            zip(cf.mp_list, [GraphGenerator(mp_emb_dim, cf.num_head, cf.sem_th, self.dev) for _ in cf.mp_list])))
        self.sg_agg = GraphChannelAttLayer(len(cf.mp_list))
        # Neibor's embedding graph for target nodes
        self.f_agg_f = GraphChannelAttLayer(len(g.undirected_relations))
        # Topological graph for target nodes
        self.f_agg_s = GraphChannelAttLayer(len(g.undirected_relations))
        # Final Graph for target nodes
        self.f_agg = GraphChannelAttLayer(4)

        for r in g.undirected_relations:
            # Topological embedding
            self.topo_encoder[r] = nn.Linear(self.ti[r[-1]]['cnt'], cf.com_feat_dim)
            # Topological Graph Generator
            self.fgg_topo[r] = GraphGenerator(cf.com_feat_dim, cf.num_head, cf.fgh_th, self.dev)  # dim=r_node_num

        # ! Graph Convolution
        if cf.conv_method == 'gcn':
            # self.GCN = GCN(cf.com_feat_dim, cf.emb_dim, g.n_class, cf.dropout)
            self.GCN = GCN(g.n_feat, cf.emb_dim, g.n_class, cf.dropout)
        self.norm_order = cf.adj_norm_order

    def forward(self, features, adj_ori, mp_emb):
        def get_rel_mat(mat, r):
            return mat[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]]

        def get_type_rows(mat, type):
            return mat[self.ti[type]['ind'], :]

        def gen_g_via_feat(graph_gen_func, mat, r):
            return graph_gen_func(get_type_rows(mat, r[0]), get_type_rows(mat, r[0]))

        # ! Heterogeneous Feature Mapping
        com_feat_mat = torch.cat([self.non_linear(
            self.encoder[t](features[self.ti[t]['ind']])) for t in self.types])   # 3913 * 16

        # ! Heterogeneous Graph Generation
        new_adj = torch.zeros((self.target_nodes_num, self.target_nodes_num)).to(self.dev)

        h, f_h, s_h = [], [], []

        # generate feature graph
        fmat_targ_fea = features[self.ti[self.target_type]['ind']]
        g_targ_fea = self.fgg_origin(fmat_targ_fea, fmat_targ_fea)
        h.append(g_targ_fea)  # 原始特征构造图

        sem_g_list = [gen_g_via_feat(self.sgg_gen[mp], mp_emb[mp], self.target_type) for mp in mp_emb]
        sem_g = self.sg_agg(sem_g_list)
        h.append(sem_g)  # 元路径特征构造图

        for r in self.ud_rels:
            ori_g = get_rel_mat(adj_ori, r)  # b*c
            fmat_r = features[self.ti[r[-1]]['ind']]
            sim_r = self.fgg_origin(fmat_r, fmat_r)

            fmat_targ_topo = ori_g.mm(sim_r)  # b*c
            feat_prop = ori_g.mm(fmat_r)
            fp_g = self.fp_origin(feat_prop, feat_prop)

            fmat_topo_hid = self.topo_encoder[r](fmat_targ_topo)
            sim_targ_topo = self.fgg_topo[r](fmat_topo_hid, fmat_topo_hid)
            f_h.append(fp_g)  # 异类邻居特征构造图
            s_h.append(sim_targ_topo)  # 拓扑特征构造图

        # !overall graph
        fp_g = self.f_agg_f(f_h)
        sim_targ_topo = self.f_agg_f(s_h)
        h.append(fp_g)
        h.append(sim_targ_topo)
        new_adj = self.f_agg(h)
        new_adj_temp = new_adj.clone()
        new_adj += new_adj_temp.t()  # sysmetric
        # ! Aggregate
        new_adj = F.normalize(new_adj, dim=0, p=self.norm_order)

        #add
        logits = self.GCN(fmat_targ_fea, new_adj)
        return logits, new_adj


class FeaturePropagation(nn.Module):
    def __init__(self, nr):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(nr, nr))    # 1*16
        nn.init.xavier_uniform_(self.weight)   # 初始化

    def forward(self, Ar, X):
        return Ar.mm(self.weight).mm(X)

class FeaturePropagation2(nn.Module):
    def __init__(self, nl, nr):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(nl, nr))    # 1*16
        nn.init.xavier_uniform_(self.weight)   # 初始化

    def forward(self, Ar, X):
        return Ar * self.weight * X


class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))    # 1*16
        nn.init.xavier_uniform_(self.weight)   # 初始化

    def forward(self, h):
        return h * self.weight


class GraphGenerator(nn.Module):
    """
    Generate graph using similarity.
    """

    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left_h, right_h):
        """

        Args:
            left_h: left_node_num * hidden_dim/feat_dim
            right_h: right_node_num * hidden_dim/feat_dim
        Returns:

        """
        if torch.sum(left_h) == 0 or torch.sum(right_h) == 0:
            return torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        s = torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)

        zero_lines = torch.nonzero(torch.sum(left_h, 1) == 0)
        # The ReLU function will generate zero lines, which lead to the nan (devided by zero) problem.
        if len(zero_lines) > 0:
            left_h[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](left_h)
            weighted_right_h = self.metric_layer[i](right_h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
        # return x


class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight
        # if weights != None:
        #     # self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
        #     with torch.no_grad():
        #         w = torch.Tensor(weights).reshape(self.weight.shape)
        #         self.weight.copy_(w)

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        adj_list = F.normalize(adj_list, dim=1, p=1)
        # Hadamard product + summation -> Conv
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)

class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output
