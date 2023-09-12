import torch, torch.nn as nn, torch.nn.functional as F

class LossDict(dict):
    def __init__(self,
                 alpha1 = .01,
                 alpha2 = .001,
                 excludePlot=[]
                 ):
        super().__init__()
        self.times = {} # For plotting.
        self.acc0 = {}
        self.acc1 = {}
        self.acc2 = {}
        self.alpha1,self.alpha2 = alpha1,alpha2
        self.excludePlot = set(excludePlot)

    def push(self, ii, k, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().item()

        if k not in self.acc0:
            self.times[k] = [ii]
            self.acc0[k] = [v]
            self.acc1[k] = [v]
            self.acc2[k] = [v]
        else:
            self.times[k].append(ii)
            self.acc0[k].append(v)
            self.acc1[k].append(self.acc1[k][-1] * (1-self.alpha1) + self.alpha1*v)
            self.acc2[k].append(self.acc1[k][-1] * (1-self.alpha2) + self.alpha2*v)

    def print(self, ii, lr=None):
        s = ''
        # FIXME: Print each acc
        for k,vs in self.acc2.items():
            s += '({:}: {:.4f})'.format(k,vs[-1])
        if lr is None:
            print(' - iter {:>6d} ::'.format(ii), s)
        else:
            print(' - iter {:>6d} (lr {:.5f}) ::'.format(ii,lr), s)

    def savePlot(self, ii, path):
        from matplotlib import pyplot as plt
        plt.clf()
        for k,vs in self.acc2.items():
            if k not in self.excludePlot:
                plt.plot(self.times[k], vs, label=k)
        plt.legend()
        plt.savefig(path)
        plt.clf()
