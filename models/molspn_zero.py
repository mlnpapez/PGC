import os
import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from models.utils import ohe2cat, cat2ohe


class MolSPNZeroCore(nn.Module):
    def __init__(self,
                 nc,
                 nd_n,
                 nd_e,
                 nk_n,
                 nk_e,
                 ns_n,
                 ns_e,
                 ni_n,
                 ni_e,
                 graph_nodes,
                 graph_edges,
                 device,
                 regime,
                 dc_n=0,
                 dc_e=0
                 ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e
        self.regime = regime

        if regime == 'cat':
            ef_dist_n = ExponentialFamilyArray.CategoricalArray
            ef_dist_e = ExponentialFamilyArray.CategoricalArray
            ef_args_n = {'K': nk_n}
            ef_args_e = {'K': nk_e}
            num_dim_n = 1
            num_dim_e = 1
        elif regime == 'bin':
            ef_dist_n = ExponentialFamilyArray.BinomialArray
            ef_dist_e = ExponentialFamilyArray.BinomialArray
            ef_args_n = {'N': nk_n-1}
            ef_args_e = {'N': nk_e-1}
            num_dim_n = 1
            num_dim_e = 1
        elif regime == 'deq':
            ef_dist_n = ExponentialFamilyArray.NormalArray
            ef_dist_e = ExponentialFamilyArray.NormalArray
            ef_args_n = {'min_var': 1e-3, 'max_var': 1e-1}
            ef_args_e = {'min_var': 1e-3, 'max_var': 1e-1}
            num_dim_n = nk_n
            num_dim_e = nk_e
            self.dc_n = dc_n
            self.dc_e = dc_e
        else:
            os.error('Unsupported \'regime\'.')

        args_nodes = EinsumNetwork.Args(
            num_var=nd_n,
            num_dims=num_dim_n,
            num_input_distributions=ni_n,
            num_sums=ns_n,
            num_classes=nc,
            exponential_family=ef_dist_n,
            exponential_family_args=ef_args_n,
            use_em=False)
        args_edges = EinsumNetwork.Args(
            num_var=nd_e,
            num_dims=num_dim_e,
            num_input_distributions=ni_e,
            num_sums=ns_e,
            num_classes=nc,
            exponential_family=ef_dist_e,
            exponential_family_args=ef_args_e,
            use_em=False)

        self.network_nodes = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
        self.network_edges = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
        self.network_nodes.initialize()
        self.network_edges.initialize()

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, x, a):
        pass

    def forward(self, x, a):
        if   self.regime == 'cat' or self.regime == 'bin':
            _x, _a = x, a
        elif self.regime == 'deq':
            x, a = cat2ohe(x, a, self.nk_nodes, self.nk_edges)
            _x = x + self.dc_n*torch.rand(x.size(), device=self.device)
            _a = a + self.dc_e*torch.rand(a.size(), device=self.device)
        else:
            os.error('Unknown regime')
        return self._forward(_x, _a)

    def logpdf(self, x, a):
        return self(x, a).mean()

    @abstractmethod
    def sample(self, num_samples):
        pass


class MolSPNZeroSort(MolSPNZeroCore):
    def __init__(self,
                 nc,
                 nd_n,
                 nk_n,
                 nk_e,
                 nl_n,
                 nl_e,
                 nr_n,
                 nr_e,
                 ns_n,
                 ns_e,
                 ni_n,
                 ni_e,
                 regime,
                 dc_n=0.6,
                 dc_e=0.6,
                 device='cuda'
                 ):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(nc, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, regime, dc_n, dc_e)

        self.logits_w = nn.Parameter(torch.ones(1, nc, device=self.device), requires_grad=True)
        self.m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool, device=self.device), diagonal=-1)

    def _forward(self, x, a):
        ll_nodes = self.network_nodes(x)
        
        if   self.regime == 'cat' or self.regime == 'bin':
            ll_edges = self.network_edges(a[:, self.m].view(-1, self.nd_edges))
        elif self.regime == 'deq':
            ll_edges = self.network_edges(a[:, self.m, :].view(-1, self.nd_edges, self.nk_edges))
        else:
            os.error('Unknown regime')

        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.logits_w, dim=1), dim=1)

    def sample(self, num_samples):
        dist_w = torch.distributions.Categorical(logits=self.logits_w)
        samp_w = dist_w.sample((num_samples, )).squeeze()

        x = self.network_nodes.sample(num_samples, class_idxs=samp_w)
        l = self.network_edges.sample(num_samples, class_idxs=samp_w)

        if   self.regime == 'cat' or self.regime == 'bin':
            a = torch.full((num_samples, self.nd_nodes, self.nd_nodes), self.nk_edges - 1, dtype=torch.float, device=self.device)
            a[:, self.m] = l
            a.transpose(1, 2)[:, self.m] = l
        elif self.regime == 'deq':
            a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, self.nk_edges, device=self.device)
            a[:, self.m, :] = l
            x, a = ohe2cat(x, a)
        else:
            os.error('Unknown regime')

        return x.cpu(), a.cpu()

MODELS = {
    'molspn_zero_sort': MolSPNZeroSort,
}
