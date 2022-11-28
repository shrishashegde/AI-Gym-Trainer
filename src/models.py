# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn

import numpy as np
import torch
from scipy.special import iv as bessel

def get_device(model):
    p = next(model.parameters(), None)
    default = "cuda" if torch.cuda.is_available() else "cpu"
    return p.device if p is not None else default

def project_to_probability_simplex(v):
    device = v.device
    nn, n = v.size()
    mu = torch.sort(v, dim=1, descending=True)[0].float()
    ns = torch.arange(1, n + 1).view(1, -1).float()
    ns = ns.to(device)
    cumsum = torch.cumsum(mu, dim=1).float() - 1
    arg = mu - cumsum / ns

    idx = [i for i in range(arg.size()[1] - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    idx = idx.to(device)
    arg = arg.index_select(1, idx)

    rho = n - 1 - torch.max(arg > 0, dim=1)[1].view(-1, 1)
    theta = 1 / (rho.float() + 1) * cumsum.gather(1, rho)
    out = torch.max(v - theta, 0 * v).clamp_(10e-8, 1)
    return out

def tmk(ts, xs, a, ms, Ts):
    block_size = 500
    ts = ts.unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, len_ts, d)
    a_xp = torch.cat([a, a], dim=1).unsqueeze(2).unsqueeze(0)  # (1, T, 2m, 1)
    ms = (ms.unsqueeze(0) / Ts.unsqueeze(1)).unsqueeze(0).unsqueeze(3)  # (1, Ts, m, 1)

    for t in range(0, xs.size()[1], block_size):
        args = ms * xs[:, t : t + block_size].unsqueeze(1).unsqueeze(1)  # (b_s, Ts, m, len_ts)
        sin_cos = a_xp * torch.cat([torch.sin(args), torch.cos(args)], dim=2)  # (b_s, Ts, 2m, len_ts)
        sin_cos = sin_cos.unsqueeze(4)  # (b_s, Ts, 2m, len_ts, 1)
        this_fv = torch.sum(sin_cos * ts[:, :, :, t : t + block_size], dim=3)  # (b_s, Ts, 2m, d)
        if t == 0:
            fv = this_fv
        else:
            fv += this_fv

    return fv

class TemporalMatchKernel(nn.Module):
    def __init__(
        self, T, normalization, beta=32, m=32, init="von_mises", normalize_l1=False
    ):
        """
        Temporal Match Kernel Layer
        :param T: Periods (list)
        :param beta: beta param of the modified Bessel function of the first kind
        :param m: number of Fourier coefficients per period
        :param init: type of initialization ('von_mises' or 'uniform': random from uniform distribution)
        :param normalize_l1: Whether to L1 normalize Fourier coefficents before each forward pass
        """
        super(TemporalMatchKernel, self).__init__()
        self.T = T
        self.beta = beta
        self.m = m
        self.normalize_l1 = normalize_l1
        self.normalization = normalization

        # Initialization
        if init == "von_mises":
            np_a = [
                (bessel(0, self.beta) - np.exp(-self.beta)) / (2 * np.sinh(self.beta))
            ] + [bessel(i, self.beta) / np.sinh(self.beta) for i in range(1, self.m)]
            np_a = np.asarray(np_a).reshape(1, -1)
            np_a = np.repeat(np_a, len(T), 0)
        elif init == "uniform":
            np_a = np.random.uniform(0, 1, (len(T), self.m))
        else:
            raise NotImplementedError

        self.a = nn.Parameter(torch.from_numpy(np_a).float())  # (T, m)
        self.ms = 2 * np.pi * torch.arange(0, self.m).float()
        self.Ts = torch.tensor(self.T, dtype=torch.float32, requires_grad=False)

    def single_fv(self, ts, xs):
        device = get_device(self)
        self.ms = self.ms.to(device)
        self.Ts = self.Ts.to(device)
        self.a.data.clamp_(min=10e-8)

        if ts.dim() == 3:
            return tmk(ts, xs, torch.sqrt(self.a), self.ms, self.Ts)
        elif ts.dim() == 4:
            out = []
            for i in range(self.Ts.shape[0]):
                outi = tmk(
                    ts[:, :, i],
                    xs,
                    torch.sqrt(self.a[i : i + 1]),
                    self.ms,
                    self.Ts[i : i + 1],
                )
                out.append(outi)
            return torch.cat(out, 1)

    def merge(self, fv_a, fv_b, offsets):
        device = get_device(self)
        eps = 1e-8
        if "feat" in self.normalization:
            a_xp = self.a.unsqueeze(0).unsqueeze(-1)
            a_xp = torch.cat([a_xp, a_xp], dim=2)
            fv_a_0 = fv_a / torch.sqrt(a_xp)
            fv_b_0 = fv_b / torch.sqrt(a_xp)
            norm_a = torch.sqrt(torch.sum(fv_a_0 ** 2, dim=3, keepdim=True) + eps) + eps
            norm_b = torch.sqrt(torch.sum(fv_b_0 ** 2, dim=3, keepdim=True) + eps) + eps
            fv_a = fv_a / norm_a
            fv_b = fv_b / norm_b

        if "freq" in self.normalization:
            norm_a = (
                torch.sqrt(torch.sum(fv_a ** 2, dim=2, keepdim=True) / self.m + eps)
                + eps
            )
            norm_b = (
                torch.sqrt(torch.sum(fv_b ** 2, dim=2, keepdim=True) / self.m + eps)
                + eps
            )
            fv_a = fv_a / norm_a
            fv_b = fv_b / norm_b

        elif self.normalization == "matrix":
            norm_a = (
                torch.sqrt(
                    torch.sum(torch.sum(fv_a ** 2, dim=-1, keepdim=True), dim=2) + eps
                )
                + eps
            )  # (b_s, T, 1)
            norm_b = (
                torch.sqrt(
                    torch.sum(torch.sum(fv_b ** 2, dim=-1, keepdim=True), dim=2) + eps
                )
                + eps
            )  # (b_s, T, 1)

        fv_a_sin = fv_a[:, :, : self.m]  # (b_s, T, m, d)
        fv_a_cos = fv_a[:, :, self.m :]  # (b_s, T, m, d)
        fv_b_sin = fv_b[:, :, : self.m]  # (b_s, T, m, d)
        fv_b_cos = fv_b[:, :, self.m :]  # (b_s, T, m, d)

        self.ms = self.ms.to(device)

        xs = offsets.float()
        ms = self.ms.unsqueeze(1)  # (m, 1)

        dot_sin_sin = torch.sum(
            fv_a_sin * fv_b_sin, dim=3, keepdim=True
        )  # (b_s, T, m, 1)
        dot_sin_cos = torch.sum(
            fv_a_sin * fv_b_cos, dim=3, keepdim=True
        )  # (b_s, T, m, 1)
        dot_cos_cos = torch.sum(
            fv_a_cos * fv_b_cos, dim=3, keepdim=True
        )  # (b_s, T, m, 1)
        dot_cos_sin = torch.sum(
            fv_a_cos * fv_b_sin, dim=3, keepdim=True
        )  # (b_s, T, m, 1)

        T = torch.tensor(self.T, dtype=torch.float32, requires_grad=False)
        T = T.to(device)
        T = T.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        cos_delta = torch.cos(
            ms.unsqueeze(0).unsqueeze(0) * xs.unsqueeze(1).unsqueeze(1) / T
        )  # (b_s, T, m, delta)
        sin_delta = torch.sin(
            ms.unsqueeze(0).unsqueeze(0) * xs.unsqueeze(1).unsqueeze(1) / T
        )  # (b_s, T, m, delta)

        dots = (
            dot_sin_sin * cos_delta
            + dot_sin_cos * sin_delta
            + dot_cos_cos * cos_delta
            - dot_cos_sin * sin_delta
        )  # (b_s, T, m, delta)
        dots = torch.sum(dots, dim=2)  # (b_s, T, delta)

        if self.normalization == "matrix":
            dots = dots / (norm_a * norm_b)
        elif self.normalization == "freq":
            dots = dots / self.m
        elif self.normalization in ["feat", "feat_freq"]:
            dots = dots / 512
        dots = torch.mean(dots, dim=1)
        return dots

    def shift_fv(self, fv, offset):
        device = get_device(self)
        fv_sin = fv[:, :, : self.m]  # (b_s, T, m, d)
        fv_cos = fv[:, :, self.m :]  # (b_s, T, m, d)

        ms = self.ms.unsqueeze(1)  # (m, 1)
        T = torch.tensor(self.T, dtype=torch.float32, requires_grad=False)
        T = T.unsqueeze(0).unsqueeze(2).unsqueeze(2)

        self.ms = self.ms.to(device)
        T = T.to(device)

        sin_delta = torch.sin(
            ms.unsqueeze(0).unsqueeze(0) * offset / T
        )  # (b_s, T, m, 1)
        cos_delta = torch.cos(
            ms.unsqueeze(0).unsqueeze(0) * offset / T
        )  # (b_s, T, m, 1)

        fv_sin_shifted = fv_sin * cos_delta + fv_cos * sin_delta
        fv_cos_shifted = fv_cos * cos_delta - fv_sin * sin_delta
        fv_shifted = torch.cat([fv_sin_shifted, fv_cos_shifted], 2)
        return fv_shifted

    def forward(self, ts_a, ts_b, xs_a, xs_b, offsets):
        """
        Computes the TMK scores over two batch of sequences with the same length
        :param ts_a: First time series (b_s, length, d)
        :param ts_b: Second time series (b_s, length, d)
        :param xs_a: Timestamps of first series (b_s, length)
        :param xs_b: Timestamps of second series (b_s, length)
        :param offsets: Offsets for which the kernel score is computed (b_s, n_offsets)
        :return: Kernel scores for every temporal offset (b_s, 2*length)
        """
        fv_a = self.single_fv(ts_a, xs_a)
        fv_b = self.single_fv(ts_b, xs_b)
        return self.merge(fv_a, fv_b, offsets)

class Model(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Model, self).__init__()

    def single_fv(self, ts, xs):
        raise NotImplementedError

    def shift_fv(self, fv, offset):
        raise NotImplementedError

    def score_pair(self, fv_a, fv_b, offsets=None):
        raise NotImplementedError

    def forward_pair(self, ts_a, ts_b, xs_a, xs_b, offsets=None):
        if offsets is None:
            length = ts_a.size()[1]
            offsets = torch.arange(-length, length).unsqueeze(0)
            offsets = offsets.to(ts_a.device)
        fv_a = self.single_fv(ts_a, xs_a)
        fv_b = self.single_fv(ts_b, xs_b)
        return self.score_pair(fv_a, fv_b, offsets)

    def forward(self, *args, **kwargs):
        return self.forward_pair(*args, **kwargs)

    def score(self, *args, **kwargs):
        return self.score_pair(*args, **kwargs)


class SumAggregation(Model):
    def __init__(self, args):
        super(SumAggregation, self).__init__(args)

    def single_fv(self, ts, xs):
        fv = torch.sum(ts, 1)
        fv = fv / (torch.sqrt(torch.sum(fv ** 2, 1, keepdim=True) + 10e-8))
        return fv

    def score_pair(self, fv_a, fv_b, offsets=None):
        return torch.sum(fv_a * fv_b, 1).view(-1, 1)


class CTE(Model):
    def __init__(self, args):
        super(CTE, self).__init__(args)
        self.cte = CirculantTemporalEncoding(m=self.args.m, lmbda=0.1)

    def single_fv(self, ts, xs):
        return self.cte.single_fv(ts.data)

    def score_pair(self, fv_a, fv_b, offsets=None, max_len=None):
        return self.cte.merge(fv_a.data, fv_b.data, offsets, max_len)


class TMK_Poullot(Model):
    def __init__(self, args):
        super(TMK_Poullot, self).__init__(args)
        self.tmk = TemporalMatchKernel(
            [2731, 4391, 9767, 14653],
            m=self.args.m,
            init="von_mises",
            normalize_l1=False,
            normalization="freq",
        )
        for p in self.tmk.parameters():
            p.requires_grad = False

    def single_fv(self, ts, xs):
        fv = self.tmk.single_fv(ts, xs)
        return fv

    def shift_fv(self, fv, offset):
        return self.tmk.shift_fv(fv, offset)

    def score_pair(self, fv_a, fv_b, offsets=None):
        return self.tmk.merge(fv_a, fv_b, offsets)


class TMK(Model):
    def __init__(self, args):
        super(TMK, self).__init__(args)
        self.tmk = TemporalMatchKernel(
            self.args.T,
            m=self.args.m,
            init="von_mises",
            normalize_l1=False,
            normalization=self.args.norm,
        )
        for p in self.tmk.parameters():
            p.requires_grad = False

        if self.args.use_pca:
            self.mean = torch.load(self.args.pca_mean)
            self.DVt = torch.load(self.args.pca_DVt)

    def single_fv(self, ts, xs):
        if self.args.use_pca:
            b_s, T, d = ts.size()
            ts = ts.view(-1, d)
            ts = ts - self.mean.expand_as(ts)
            ts = torch.mm(self.DVt, ts.transpose(0, 1)).transpose(0, 1)
            ts = ts / torch.sqrt(torch.sum(ts ** 2, dim=1, keepdim=True))
            ts = ts.view(b_s, T, d)

        fv = self.tmk.single_fv(ts, xs)
        return fv

    def shift_fv(self, fv, offset):
        return self.tmk.shift_fv(fv, offset)

    def score_pair(self, fv_a, fv_b, offsets=None):
        return self.tmk.merge(fv_a, fv_b, offsets)
