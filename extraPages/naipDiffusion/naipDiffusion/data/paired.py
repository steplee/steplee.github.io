""" A Dataset that yields pairs of images from the exact same boxes, but different datasets.

@FrastPairedDataset yields the pairs after being constructed with two different frast imagesets.

@get_dataset builds a ConcatDataset from multiple @FrastPairedDataset given a root dir. It assumes the naming convention
from the usgs downloader code and automatically finds matching frast imagesets.

No preprocessing is needed. No caching is done.

"""

import torch, torch.nn as nn, torch.nn.functional as F, time
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import frastpy2, numpy as np, cv2, os

# FIXME: This works without worker_init_fn, but is it really thread safe?
# Or like how does it even work? Did they fix Dataset s.t. you needn't have a seperate init func?

class FrastPairedDataset(Dataset):
    def __init__(self, path1, path2, levels, squareSize=1):
        super().__init__()
        self.path1 = path1
        self.squareSize=squareSize

        opts = frastpy2.EnvOptions()
        opts.isTerrain = False
        opts.readonly = True
        opts.cache = False # 10% faster.

        dset1 = frastpy2.FlatReaderCached(path1, opts)
        dset2 = frastpy2.FlatReaderCached(path2, opts)

        coords1,coords2 = set(), set()

        self.coords = []

        if squareSize == 1:
            for level in levels:
                for c in dset1.iterCoords(level): coords1.add(c.c())
                for c in dset2.iterCoords(level): coords2.add(c.c())
                lvlCoords = coords1.intersection(coords2)
                lvlCoords = [(level, c) for c in lvlCoords]
                self.coords.extend(lvlCoords)
        else:
            for level in levels:
                minx,maxx = 9e9, -9e9
                miny,maxy = 9e9, -9e9
                for c in dset1.iterCoords(level):
                    miny = min(miny, c.y())
                    minx = min(minx, c.x())
                    maxy = max(maxy, c.y())
                    maxx = max(maxx, c.x())
                    coords1.add(c.c())
                for c in dset2.iterCoords(level):
                    coords2.add(c.c())

                lvlCoords = []
                # print('rng', minx, miny, '->', maxx,maxy)
                for y in range(miny,maxy,squareSize):
                    for x in range(minx,maxx,squareSize):
                        haveAll = True
                        for dy in range(squareSize):
                            for dx in range(squareSize):
                                c = frastpy2.BlockCoordinate(level,y+dy,x+dx).c()
                                haveAll = haveAll and (c in coords1 and c in coords2)
                        if haveAll:
                            # lvlCoords.append((level,c))
                            lvlCoords.append((level,np.array((x,y,x+squareSize,y+squareSize),dtype=np.uint32)))
                self.coords.extend(lvlCoords)

        self.dset1,self.dset2 = dset1,dset2

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        # wi = torch.utils.data.get_worker_info()

        # print(self.path1, c)
        if self.squareSize == 1:
            lvl, c = self.coords[i]
            img1 = self.dset1.getTile(c, 3)
            img2 = self.dset2.getTile(c, 3)
        else:
            lvl, c = self.coords[i]
            img1 = self.dset1.getTlbr(lvl, c, 3)
            img2 = self.dset2.getTlbr(lvl, c, 3)

        # NOTE: Downscale the second one.
        # img2 = cv2.pyrDown(img2)

        return img1, img2



# The way the files are downloaded, we can find pairs by seeing which files have a difference of 'a' and 'b'
def isDownloadedPair(a,b):
    acc = 0
    for x,y in zip(a,b):
        if ((x=='a' and y=='b') or (x=='b' and y=='a')) and x!=y: acc += 1
        elif x!=y: acc += 2
    # print(' - is pair({},{}) ='.format(a,b), acc==1)
    return acc == 1

# @requireSubstrings allows to filter by requiring a file name to include a substring
def get_dataset_pairs(cfg, rootDir='/data/multiDataset1/', requireSubstrings=['.']):

    allFftFiles, pairs = [], []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            if file.endswith('.fft'):
                allFftFiles.append(os.path.join(root,file))

    allFftFiles = sorted(allFftFiles)

    for f1 in allFftFiles:
        for f2 in allFftFiles:
            if len(f1) == len(f2) and isDownloadedPair(f1,f2):
                good = False
                for ss in requireSubstrings:
                    if ss in f1 or ss in f2:
                        good = True
                if good:
                    if f1 < f2: f1,f2 = f2,f1
                    pairs.append((f1,f2))

    pairs = sorted(list(set(pairs)))
    return pairs

# def get_dataset(cfg, rootDir='/data/multiDataset1/', requireSubstrings=[], squareSize=1, clazz=FrastPairedDataset):
    # pairs = get_dataset_pairs(cfg, rootDir, requireSubstrings)
def get_dataset(cfg, pairs, requireSubstrings=[], squareSize=1, clazz=FrastPairedDataset):

    print(f' - Concatting datasets from these pairs (SquareSize={squareSize}):')
    for f1,f2 in pairs:
        print('    -',f1,f2)
    datasets = [clazz(f1,f2, [14,15,16,17], squareSize=squareSize) for (f1,f2) in pairs]

    # dset = FrastPairedDataset(
                        # '/data/multiDataset1/naip_simi1/simi1a/4_UTM_11N.fft',
                        # '/data/multiDataset1/naip_simi1/simi1b/4_UTM_11N.fft',
                       # [14,15,16,17])
                       # [13])
    # datasets.append(dset)
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(' - Len dataset:', len(dataset))
    # print(' - Len loader :', len(dloader))
    return dataset


if __name__ == '__main__':
    dset = FrastPairedDataset(
                        '/data/multiDataset1/naip_simi1/simi1a/4_UTM_11N.fft',
                        '/data/multiDataset1/naip_simi1/simi1b/4_UTM_11N.fft',
                       [14,15,16,17])

    torch.manual_seed(0)
    dloader = DataLoader(dset, batch_size=96, num_workers=2, shuffle=True)
    # dloader = DataLoader(dset, batch_size=96, num_workers=2, shuffle=False)

    i = 0
    st = time.time()
    for img1, img2 in dloader:
        i += img1.shape[0]

        img1 = img1[0:5].cpu().numpy()
        img2 = img2[0:5].cpu().numpy()
        img1 = np.vstack(img1)
        img2 = np.vstack(img2)
        img = np.hstack((img1,img2))
        if i > 1000: break

        if True:
            img = img[...,[2,1,0]]
            cv2.imshow('img',img)
            cv2.waitKey(0)
    print('took', time.time()-st, 's')

