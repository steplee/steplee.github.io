'''
Returns random patches in some aoi (and there quad points), supporting a pair of overlapping datasets.
'''

import numpy as np, random, signal
import cv2
import frastpy2

import torch
from torch.utils.data import Dataset, DataLoader

'''
Return true if the img is too black, too white, etc.
This is a possible perf bottleneck. TODO check.
'''
def img_is_bad(qimg):
    if qimg is None: return True
    #bad = np.diag(qimg[...,0]).std() < 3
    #return bad
    a,b,c = qimg[20,20].astype(np.int32), qimg[40,40].astype(np.int32), qimg[30,110].astype(np.int32)
    return (abs(a-b)<3).all() and (abs(b-c)<3).all()


class DsetWrapper():
    def __init__(self,fpath):
        opts = frastpy2.EnvOptions()
        opts.readonly = True
        opts.isTerrain = False
        opts.cache = False # Might as well, since we are unlikely to get the same tiles soon.
        self.reader = frastpy2.FlatReaderCached(fpath, opts)

        regions = np.array(self.reader.getRegions()).reshape(-1,4)
        print(regions)
        assert regions.shape[0], "only one region supported right now"
        self.tlbrDwm = regions[0]

    def rasterIo(self, tlbr, w,h):
        img = self.reader.rasterIo(tlbr, w, h, 1)
        return img

class RandomPatchFrastDataset(Dataset):
    def __init__(self,
            sets,
            warpVersion=1,
            wh=256,
            N=10000):

        self.sets_ = sets
        self.numSets = len(self.sets_)
        self.warpVersion = warpVersion

        self.wh = wh
        self.N = N
        # self.sampleWH = wh + wh // 2
        self.sampleWH = wh + wh

        self.epochs = 0

        #self.generate(N)
        if 1:
            d = DsetWrapper(self.sets_[0][0][0])
            tlbr = d.tlbrDwm
            self.dsetTl = tlbr[:2]
            self.dsetSize = max(tlbr[2:] - tlbr[:2])
            self.danglingDataset = d # Awful.

    @staticmethod
    def init(worker_id):

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        seed = 1

        info = torch.utils.data.get_worker_info()
        if info is not None:
            self = info.dataset
            id = info.id
            #seed = info.seed
            seed = id + self.epochs * 1000
            #print(' - init with epochs =', self.epochs, 'seed:', seed)
        else:
            seed = seed + self.epochs * 1000
        np.random.seed(seed)
        random.seed(seed)

        self.sets = []
        #for ns,t in self.sets_: self.sets.append(([SimpleGeoDataset(n) for n in ns],t))
        # self.sets = [([SimpleGeoDataset(n) for n in self.sets_[id][0]], self.sets_[id][1])]
        # self.sets = [([DsetWrapper(n) for n in self.sets_[id][0]], self.sets_[id][1])]
        for ns,t in self.sets_: self.sets.append(([DsetWrapper(n) for n in ns],t))

        '''
        tlbr = self.sets[0][0][0].tlbrDwm
        self.dsetTl = tlbr[:2]
        self.dsetSize = max(tlbr[2:] - tlbr[:2])
        '''


    def __len__(self):
        return self.N
        #return len(self.spec)

    def __getitem__(self, i):
        return self.get_one()


    def get_one(self):
        trials = 0
        while True:
            trials += 1
            if trials > 30: print(' - lots of trials:',trials)
            #dsets = np.random.choice(self.sets)
            dsets, (minMeters,maxMeters) = random.choice(self.sets)
            dseta, = np.random.choice(dsets, 1, replace=False)

            size = np.random.rand() * (maxMeters - minMeters) + minMeters

            xlo,ylo = np.random.random(2)
            xlo = dseta.tlbrDwm[0] + (dseta.tlbrDwm[2] - dseta.tlbrDwm[0] - size) * xlo
            ylo = dseta.tlbrDwm[1] + (dseta.tlbrDwm[3] - dseta.tlbrDwm[1] - size) * ylo

            xsz,ysz = size,size
            tlbr = np.array((xlo,ylo,xlo+xsz,ylo+ysz),dtype=np.float64)
            aimg0 = dseta.rasterIo(tlbr, self.sampleWH,self.sampleWH )


            if not img_is_bad(aimg0):
                if aimg0.shape[-1] == 3: aimg0 = cv2.cvtColor(aimg0, cv2.COLOR_RGB2GRAY)[...,np.newaxis]

                tl = ((tlbr[:2] - self.dsetTl) / self.dsetSize) - .5
                br = ((tlbr[2:] - self.dsetTl) / self.dsetSize) - .5
                wh = ((tlbr[2:] - tlbr[:2]) / self.dsetSize) - .5
                tlwh = np.array((*tl,*wh),dtype=np.float32)
                pts = np.array((
                    tl[0],tl[1],
                    br[0],tl[1],
                    br[0],br[1],
                    tl[0],br[1],
                )).reshape(-1).astype(np.float32)

                # return torch.from_numpy(aimg0), torch.from_numpy(tlwh)
                return torch.from_numpy(aimg0), torch.from_numpy(pts)


    def generate(self, N):
        print(' - Dataset generating', N)
        self.specs = []
        for ii in range(N):
            pass
        print(' - Dataset finished generating')



if __name__ == '__main__':
    # dset = RandomWarpedTiffDataset([[[
        # '/data/midwest_planet/gray/august.tif',
        # '/data/midwest_planet/gray/may2019.tif'], (2400,4000)]])
    dset = RandomPatchFrastDataset([[[
        '/data/naip/mdpa/mdpa.fft',
        '/data/naip/mdpa/mdpa.fft'],
            (600,4000)]])

    loader = DataLoader(dset, batch_size=4, worker_init_fn=RandomPatchFrastDataset.init, num_workers=1)
    loader_iter = iter(loader)

    for i in range(10):
        #a,b,ap,bp = dset[i]
        a,tlwh = next(loader_iter)
        tlwh += .5
        print(a.shape)
        a = np.array(a[0])
        cv2.imshow('a',a); cv2.waitKey(0)
        print(a.shape, tlwh)





