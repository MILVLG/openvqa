import math
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from openvqa.ops.gelu import GeLU


def get_gaussian_keys(n_keys, dim, normalized, seed):
    """
    Generate random Gaussian keys.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_keys, dim)
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def get_uniform_keys(n_keys, dim, normalized, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    X = rng.uniform(-bound, bound, (n_keys, dim))
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)

class QueryMLP(nn.Module):
    def __init__(self, __C, act='ReLU'):
        super().__init__()
        self.__C = __C
        layers = []
        if __C.INPUT_DROPOUT > 0:
            layers.append(nn.Dropout(__C.INPUT_DROPOUT))
        layers.append(nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE // 2))
        # layers.append(getattr(nn, act)())
        layers.append(nn.ELU())
        layers.append(nn.Linear(__C.HIDDEN_SIZE // 2, __C.HIDDEN_SIZE * 2))
        
        self.mlp = nn.Sequential(*layers)
        self.bs = nn.BatchNorm1d(num)

    def forward(self, x):
        n_batches = x.size(0)
        x = self.mlp(x)
        x = self.bs(x)
        return x.view(
            n_batches,
            -1,
            self.__C.MEM_HEAD,
            self.__C.HIDDEN_SIZE // 2
        )

class Memory(nn.Module):

    VALUES = None
    _ids = itertools.count(0)

    def __init__(self, __C):

        super().__init__()
        self.__C = __C
        self.id = next(self._ids)

        # initialize keys
        self.init_keys()

        values = nn.EmbeddingBag(
            __C.MEM_SIZE, __C.HIDDEN_SIZE, mode='sum', sparse=__C.MEM_SPARSE)
        
        # values initialization
        nn.init.normal_(values.weight, mean=0,
                        std=__C.HIDDEN_SIZE ** -0.5)
        self.values = values

        # optionally use the same values for all memories
        if __C.MEM_SHARE_VALUES:
            if HashingMemory.VALUES is None:
                HashingMemory.VALUES = self.values.weight
            else:
                self.values.weight = HashingMemory.VALUES
        
        # for different lr
        if 'value' not in __C.SPECIAL_W:
            __C.SPECIAL_W['value'] = []
            __C.SPECIAL_LR['value'] = __C.VALUE_LR_TIMES
        __C.SPECIAL_W['value'].append(self.values.weight)
        
        # # no query network
        # if len(params.mem_query_layer_sizes) == 0:
        #     assert self.heads == 1 or self.use_different_keys or self.shuffle_query
        #     assert self.input_dim == self.k_dim
        #     self.query_proj = QueryIdentity(self.input_dim, self.heads, self.shuffle_query)

        # query network
        self.query_proj_l = QueryMLP(__C, 14)
        self.query_proj_v = QueryMLP(__C, 100)


        # # shuffle indices for different heads
        # if self.shuffle_indices:
        #     head_permutations = [torch.randperm(self.n_indices).unsqueeze(0) for i in range(self.heads)]
        #     self.register_buffer('head_permutations', torch.cat(head_permutations, 0))

        # # do not learn the query network
        # if self.query_net_learn is False:
        #     for p in self.query_proj.parameters():
        #         p.requires_grad = False

    def forward(self, input, mod):
        """
        Read from the memory.
        """
        # # detach input
        # if self.query_detach_input:
        #     input = input.detach()

        prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)

        # (bs * heads, k_dim)
        if mod == 'lang':
            query = self.query_proj_l(input)
        else:
            query = self.query_proj_v(input)

        # get indices  (bs * heads, knn) ** 2
        scores, indices = self.get_indices(query, mod)
        # scores = F.softmax(scores.float(), dim=-1).type_as(scores)            # (bs * heads, knn)
        scores = F.softmax(scores, dim=-1)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.__C.MEM_HEAD * self.__C.KNN)              # (bs, heads * knn)
        scores = scores.view(bs, self.__C.MEM_HEAD * self.__C.KNN)                # (bs, heads * knn)

        # weighted sum of values
        output = self.values(
            indices,
            per_sample_weights=scores.to(self.values.weight.data)
        ).to(scores)                                                              # (bs, v_dim)

        # reshape output
        # (..., v_dim)
        output = output.view(prefix_shape + (self.__C.HIDDEN_SIZE,))

        return output

    def create_keys(self):
        """
        This function creates keys and returns them.
        I guess you could see that from the name of the function and the fact that is has a return statement.
        """
        half = self.__C.K_DIM // 2
        n_keys = self.__C.SUB_SIZE

        # random keys from Gaussian or uniform distributions
        if self.__C.KEYS_TYPE in ['gaussian', 'uniform']:
            init = get_gaussian_keys if self.__C.KEYS_TYPE == 'gaussian' else get_uniform_keys
            keys = torch.from_numpy(np.array([
                init(n_keys, half, self.__C.KEYS_NORMALIZED_INIT, seed=(2 * i + j))
                for i in range(self.__C.MEM_HEAD)
                for j in range(2)
            ])).view(self.__C.MEM_HEAD, 2, n_keys, half)

        return keys

    def init_keys(self):
        """
        Initialize keys.
        """
        keys_v = self.create_keys()
        self.keys_v = nn.Parameter(keys_v)
        keys_l = self.create_keys()
        self.keys_l = nn.Parameter(keys_l)

    def get_indices(self, query, mod):
        """
        Generate scores and indices given unnormalized queries.
        """
        query = query.view(-1, self.__C.MEM_HEAD, self.__C.K_DIM)
        if mod == 'lang':
            keys = self.keys_l
        else:
            keys = self.keys_v
        outputs = [
            self._get_indices(query[:, i], self.__C.KNN,
                              keys[i][0], keys[i][1])
            for i in range(self.__C.MEM_HEAD)
        ]
        scores = torch.cat([s.unsqueeze(1)
                            for s, _ in outputs], 1).view(-1, self.__C.KNN)
        indices = torch.cat([idx.unsqueeze(1)
                             for _, idx in outputs], 1).view(-1, self.__C.KNN)
        return scores, indices

    def _get_indices(self, query, knn, keys1, keys2, mod):
        """
        Generate scores and indices given keys and unnormalized queries.
        """
        bs = query.size(0)
        half = self.__C.K_DIM // 2

        # split query for product quantization
        q1 = query[:, :half]                                                                                          # (bs, half)
        q2 = query[:, half:]                                                                                          # (bs, half)

        # compute indices with associated scores
        scores1 = F.linear(q1, keys1, bias=None)                                                                     # (bs, n_keys ** 0.5)
        scores2 = F.linear(q2, keys2, bias=None)                                                                      # (bs, n_keys ** 0.5)
        
        if mod == 'lang':
            self.s1 = F.softmax(scores1.sum(dim=0) / 2.0,
                                dim=-1).unsqueeze(0).detach() + 1.0
            self.s2 = F.softmax(scores2.sum(dim=0) / 2.0,
                                dim=-1).unsqueeze(0).detach() + 1.0
            scores1, indices1 = scores1.topk(knn, dim=1, largest=True, sorted=True)                                       # (bs, knn) ** 2
            scores2, indices2 = scores2.topk(knn, dim=1, largest=True, sorted=True)
        else:
            # (bs, knn) ** 2
            scores1 = torch.mul(scores1, self.s1)
            scores2 = torch.mul(scores2, self.s2)
            # (bs, knn) ** 2
            scores1, indices1 = scores1.topk(
                knn, dim=1, largest=True, sorted=True)
            scores2, indices2 = scores2.topk(
                knn, dim=1, largest=True, sorted=True)

        # scores1, indices1 = get_knn_faiss(keys1, q1.contiguous(), knn, distance='dot_product')                        # (bs, knn) ** 2
        # scores2, indices2 = get_knn_faiss(keys2, q2.contiguous(), knn, distance='dot_product')                        # (bs, knn) ** 2

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, knn, 1).expand(bs, knn, knn) +
            scores2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                                                                # (bs, knn ** 2)
        all_indices = (
            indices1.view(bs, knn, 1).expand(bs, knn, knn) * self.__C.SUB_SIZE +
            indices2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                                                                # (bs, knn ** 2)

        # select overall best scores and indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=1, largest=True, sorted=True)                        # (bs, knn)
        indices = all_indices.gather(1, best_indices)                                                                 # (bs, knn)

        # code below: debug instant retrieval speed
        # scores = torch.zeros(bs, knn, dtype=query.dtype, device=query.device)
        # indices = torch.arange(knn, dtype=torch.int64, device=query.device).view(1, knn).expand(bs, knn)

        # return scores with indices
        return scores, indices
