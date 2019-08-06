# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.optim as Optim


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size, warmup_epoch):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        self.warmup_epoch = warmup_epoch
        self.lr_list = []
        for p in self.optimizer.param_groups:
            self.lr_list.append(p['lr'])


    def step(self):
        self._step += 1

        rate = self.rate()
        i = 0
        for p in self.optimizer.param_groups:
            p['lr'] = rate * self.lr_list
            i += 1
        self._rate = self.lr_base * rate

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.25):
            r = 1/(self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.5):
            r = 2/(self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.75):
            r = 3/(self.warmup_epoch + 1)
        else:
            r = 1

        return r


def get_optim(__C, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = __C.LR_BASE

    std_optim = getattr(Optim, __C.OPT)
    if len(__C.SPRCIAL_W)  != 0:
        params = []
        l_id = []
        for k in __C.SPRCIAL_W.keys():
            l_id += __C.SPRCIAL_W[k]
            params.append(
                {"params": __C.SPRCIAL_W[k], "lr": lr_base * __C.SPRCIAL_LR[k]}
            )
        l_id = list(map(id, l_id))
        normal_params = filter(lambda p: id(p) not in l_id, model.parameters())
        params.append(
            {"params": normal_params, "lr": lr_base},
        )
        eval_str = 'params'
    else:
        params = filter(lambda p: p.requires_grad, model.parameters())
        eval_str = 'params, lr=lr_base'
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])

    optim = WarmupOptimizer(
        lr_base,
        eval('std_optim' + '(' + eval_str + ')'),
        data_size,
        __C.BATCH_SIZE,
        __C.WARMUP_EPOCH
    )

    return optim


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r
