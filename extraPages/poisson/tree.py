import torch
import time

class Timed:
    def __init__(self, msg):
        self.msg = msg
    def __enter__(self):
        torch.cuda.synchronize()
        self.st = time.time()
    def __exit__(self, *a, **k):
        torch.cuda.synchronize()
        et = time.time()
        dt = (et - self.st) * 1e6 # micros
        ss = '{:>5.3f}us'.format(dt)
        if dt > 1e3: ss = '{:>5.3f}ms'.format(dt*1e-3)
        if dt > 1e6: ss = '{:>5.3f}s '.format(dt*1e-6)
        print(' - {} took {}.'.format(self.msg, ss))


class Tree:
    def __init__(self):
        pass

    def set(self, k, v, maxLvl=12,
            do_average=True):

        self.lvls = [None] * maxLvl
        dev = torch.device('cuda')

        N,D = k.size()
        C0 = v.size(-1)
        if v.ndim == 1: v = v.view(N,1)
        assert v.ndim == 2 and v.size(0) == N

        # Append 1, so that we can compute average later.
        if do_average:
            v = torch.cat((v, torch.ones_like(v[:,:1])),1)
        C = v.size(1)

        # Transpose to [D,N]
        k = k.cuda().permute(1,0).float()

        # Normalize k
        tl = k.min(1).values
        br = k.max(1).values
        tl = tl
        sz = (br - tl).max().ceil().item()
        self.tl, self.sz = tl, sz

        print(tl,br,sz)

        # Create deepest level.
        lvl = maxLvl-1
        iszL = (1<<lvl) / sz
        kL = k.sub(tl.unsqueeze(1)).mul_(iszL)
        st = torch.sparse_coo_tensor(kL,v,device=dev,size=(*((1<<lvl,)*D),C)).coalesce()
        if do_average:
            # NOTE: Can do this without calling coalesce() again by using .values_().div_(.)
            kk1,vv1 = st.indices(), st.values()
            vv1 = vv1[...,:-1]/vv1[...,1:]
            st = torch.sparse_coo_tensor(kk1,vv1,device=dev).coalesce()
        self.lvls[maxLvl-1] = st
        print(' - lvl {:>2d}: {:> 8d} ents'.format(lvl, st._nnz()))

        for lvl in range(maxLvl-2, -1, -1):
            # Halfscale
            prev = self.lvls[lvl+1]

            if 1:
                # Version that uses previous level. Way faster
                kk,vv = prev.indices(), prev.values()
                if do_average: vv = torch.cat((vv,torch.ones_like(vv[:,:1])), 1)
                kk = (kk >> 1)
            else:
                # Version that uses original data
                iszl = (1<<lvl) / sz
                kk = k.sub(tl).mul_(iszl)
                vv = v

            st = torch.sparse_coo_tensor(kk,vv,device=dev,size=(*((1<<lvl,)*D),C)).coalesce()
            if do_average:
                kk1,vv1 = st.indices(), st.values()
                vv1 = vv1[...,:-1]/vv1[...,1:]
                st = torch.sparse_coo_tensor(kk1,vv1,device=dev).coalesce()
            self.lvls[lvl] = st
            print(' - lvl {:>2d}: {:> 8d} ents'.format(lvl, st._nnz()))

    def getLvl(self, i):
        return self.lvls[i]
    def numLvls(self): return len(self.lvls)


def simple_test():
    torch.manual_seed(0)
    maxLvl = 12
    k = torch.randn(55_000_000, 3)
    v = torch.randn(k.size(0))
    ot = Tree()
    for i in range(1):
        with Timed('set'):
            ot.set(k,v,
                maxLvl=maxLvl,
                do_average=False
                )

if __name__ == '__main__':
    simple_test()
