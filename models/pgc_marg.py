import torch
import torch.nn as nn

from models.backend import backend_selector
from typing import Optional


class PGCMargSort(nn.Module):
    def __init__(self,
                 loader_trn,
                 hpars
                 ):
        super().__init__()
        x = torch.stack([b['x'] for b in loader_trn.dataset])
        a = torch.stack([b['a'] for b in loader_trn.dataset])

        network_x, nd_x, nk_x, network_a, nd_a, nk_a = backend_selector(x, a, hpars, nk_x_offset=True)

        self.nd_x = nd_x
        self.nd_a = nd_a
        self.nk_x = nk_x
        self.nk_a = nk_a

        self.nc = hpars['nc']
        self.device = hpars['device']

        self.network_x = network_x
        self.network_a = network_a

        self.logits_n = nn.Parameter(torch.randn(nd_x,    device=self.device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.randn(self.nc, device=self.device), requires_grad=True)
        self.m = torch.tril(torch.ones(nd_x, nd_x, dtype=torch.bool, device=self.device), diagonal=-1)

        self.to(self.device)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        x -= 1
        mask_xn = (x > -1)
        mask_an = (mask_xn.unsqueeze(2) * mask_xn.unsqueeze(1))[:, self.m].view(-1, self.nd_a)
        self.network_x.set_marginalization_mask(mask_xn)
        self.network_a.set_marginalization_mask(mask_an)

        n = mask_xn.sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)

        logs_c = dist_n.log_prob(n)
        logs_x = self.network_x(x)
        logs_a = self.network_a(a[:, self.m].view(-1, self.nd_a))
        logs_w = torch.log_softmax(self.logits_w, dim=0).unsqueeze(0)

        return logs_c + torch.logsumexp(logs_x + logs_a + logs_w, dim=1)

    def logpdf(self, x, a):
        return self(x, a).mean()

    @torch.no_grad
    def _sample(self, num_samples: int=1, cond_x: Optional[torch.Tensor]=None, cond_a: Optional[torch.Tensor]=None):
        if cond_x is not None and cond_a is not None:
            if len(cond_x) == len(cond_a):
                num_samples = len(cond_x)
            else:
                raise 'len(cond_x) and len(cond_a) are not equal.'

        if cond_x is not None:
            cond_x -= 1
            mask_x = (cond_x > -1)
            logs_n = self.logits_n.unsqueeze(0).expand(num_samples, -1).masked_fill_(mask_x, -torch.inf)

            self.network_x.set_marginalization_mask(mask_x)
            logs_x = self.network_x(cond_x)
        else:
            mask_x = torch.zeros(num_samples, self.nd_x, device=self.device, dtype=torch.bool)
            logs_x = torch.zeros(num_samples, self.nc,   device=self.device)
            logs_n = self.logits_n.unsqueeze(0).expand(num_samples, -1)

        if cond_a is not None:
            mask_a = (cond_a > -1)
            mask_a = mask_a[:, self.m].view(-1, self.nd_a)
            cond_a = cond_a[:, self.m].view(-1, self.nd_a)

            self.network_a.set_marginalization_mask(mask_a)
            logs_a = self.network_a(cond_a)
        else:
            mask_a = torch.zeros(num_samples, self.nd_a, device=self.device, dtype=torch.bool)
            logs_a = torch.zeros(num_samples, self.nc,   device=self.device)

        logs_w = logs_x + logs_a + self.logits_w.unsqueeze(0)

        samp_n = torch.distributions.Categorical(logits=logs_n).sample()
        samp_w = torch.distributions.Categorical(logits=logs_w).sample()

        mask_xn = torch.arange(self.nd_x, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        mask_an = (mask_xn.unsqueeze(2) * mask_xn.unsqueeze(1))[:, self.m].view(-1, self.nd_a)
        mask_xx = mask_x + mask_xn
        mask_aa = mask_a + mask_an

        x = self.network_x.sample(num_samples, class_idxs=samp_w, x=cond_x)
        l = self.network_a.sample(num_samples, class_idxs=samp_w, x=cond_a)

        x[~mask_xx] = -1
        l[~mask_aa] = 0

        a = torch.zeros((num_samples, self.nd_x, self.nd_x), device=self.device)
        a[:, self.m] = l
        a.transpose(1, 2)[:, self.m] = l

        x += 1

        return x.to(device='cpu', dtype=torch.int), a.to(device='cpu', dtype=torch.int)

    @torch.no_grad
    def sample(self, num_samples: int=1, cond_x: Optional[torch.Tensor]=None, cond_a: Optional[torch.Tensor]=None, chunk_size: int=2000):
        x_sam = []
        a_sam = []

        if cond_x is not None and cond_a is not None:
            if len(cond_x) == len(cond_a):
                if len(cond_x) > chunk_size:
                    for chunk_cond_x, chunk_cond_a in zip(cond_x.chunk(chunk_size), cond_a.chunk(chunk_size)):
                        x, a = self._sample(cond_x=chunk_cond_x, cond_a=chunk_cond_a)
                        x_sam.append(x)
                        a_sam.append(a)
                    x_sam, a_sam = torch.cat(x_sam), torch.cat(a_sam)
                else:
                    x_sam, a_sam = self._sample(cond_x=cond_x, cond_a=cond_a)
            else:
                raise 'len(cond_x) and len(cond_a) are not equal.'
        else:
            if num_samples > chunk_size:
                chunks = num_samples // chunk_size*[chunk_size] + ([num_samples % chunk_size] if num_samples % chunk_size > 0 else [])
                for n in chunks:
                    x, a = self._sample(n)
                    x_sam.append(x)
                    a_sam.append(a)
                x_sam, a_sam = torch.cat(x_sam), torch.cat(a_sam)
            else:
                x_sam, a_sam = self._sample(num_samples)  

        return x_sam, a_sam


MODELS = {
    'marg_sort': PGCMargSort,
}
