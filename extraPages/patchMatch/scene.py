import torch, numpy as np

class Intrinsics:
    def __init__(self, wh, f, c, dc=None):
        self.wh, self.f, self.c, self.dc = wh, f, c, dc
        if not isinstance(self.f, np.ndarray): self.f = np.array(self.f)
        if not isinstance(self.c, np.ndarray): self.c = np.array(self.c)
        if not isinstance(self.dc, np.ndarray): self.dc = np.array(self.dc)
        if not isinstance(self.wh, np.ndarray): self.wh = np.array(self.wh)
        self.map = None

    def K(self):
        return np.array((
            self.f[0], 0, self.c[0],
            0, self.f[1], self.c[1],
            0, 0, 1)).reshape(3,3)

    def projectStage1(self, x):
        if x.ndim > 1 and x.shape[-1] == 3:
            return (x[..., :2] / x[..., 2:]) * self.f + self.c

    def projectStage2(self, x):
        y = self.projectStage1(x)
        return self.distort(y)

    def distort(self, x):
        y = (x-self.c)
        yy = y / self.f
        r2 = ((yy*yy)).sum(-1)[...,np.newaxis]
        return c + y / (1 + self.d[0]*r2 + self.d[1]*r2*r2)

    def resize(self, wh):
        factor = wh / self.wh
        f = self.f * factor
        c = self.c * factor
        return Intrinsics(wh, f, c, self.dc)

    def get_unwarp_map_gradient_descent():
        h,w = self.wh
        map = np.zeros((h,w,2),dtype=np.float32)
        grid = np.stack(np.meshgrid(np.arange(0,w), np.arange(0,h)), -1).astype(np.float32)

        print(' - Computing unwarp map...')
        for iter in range(5):
            p1 = distort(grid+map, d, f, c)
            res = (p1 - grid).reshape(-1,2)
            print('mse',(res*res).mean())
            map -= res.reshape(h,w,2) * .96

        return grid+map

    # Bilinear interpolation with the computed undistort map.
    # The first time this is called, it will compute the map.
    def undistort(self, x):
        assert x.shape[1] == self.wh[0]
        assert x.shape[0] == self.wh[1]

        if self.map is None:
            self.map = self.get_unwarp_map_gradient_descent()

        imap = self.map.astype(np.int32)

        def clamp_(a):
            np.clip(a[...,0], 0, w-1, out=a[...,0])
            np.clip(a[...,1], 0, h-1, out=a[...,1])
            return a

        ai = (clamp_(imap + (0, 0)) * (1, w)).sum(-1)
        bi = (clamp_(imap + (1, 0)) * (1, w)).sum(-1)
        ci = (clamp_(imap + (1, 1)) * (1, w)).sum(-1)
        di = (clamp_(imap + (0, 1)) * (1, w)).sum(-1)

        img_ = x.reshape(h*w,-1).astype(np.float32)
        fx,fy = (map - imap).transpose(2,0,1)
        fx = fx.reshape(h,w,-1)
        fy = fy.reshape(h,w,-1)
        a = img_[ai] * (1-fx) * (1-fy)
        b = img_[bi] * (  fx) * (1-fy)
        c = img_[bi] * (  fx) * (  fy)
        d = img_[bi] * (1-fx) * (  fy)

        return (a+b+c+d).reshape(h,w,-1).astype(img.dtype)

class Scene:
    def __init__(self, meta):
        self.meta = meta
        for k,v in self.meta.items(): setattr(self,k,v)

    def save(self, path):
        torch.save(self.meta, path)

    def __getitem__(self, k): return getattr(self,k)

    @staticmethod
    def load(path):
        meta = torch.load(path)
        for k,v in meta.items():
            if isinstance(v, np.ndarray):
                meta[k] = torch.from_numpy(v).cuda()
        return Scene(meta)

    @staticmethod
    def fromImagePair(
            path1, path2,
            intrina, intrinb,
            downscales=1,
            siftPoints=2500,
            loweRatio=.75,
            fmReprojThresh=3,
            debugMatching=False, debugRectification=True,
            **matchingKw
            ):

        import cv2
        from matchingTransformation import find_matching_transformation, fundamental_from_pts, apply_homog

        imga,imgb = cv2.imread(path1), cv2.imread(path2)
        for i in range(downscales):
            imga = cv2.resize(imga, (0,0), fx=.5, fy=.5)
            imgb = cv2.resize(imgb, (0,0), fx=.5, fy=.5)
        sift = cv2.SIFT_create(siftPoints)
        kptsa, desa = sift.detectAndCompute(imga,None)
        kptsb, desb = sift.detectAndCompute(imgb,None)
        ptsa = np.array([kpt.pt for kpt in kptsa])
        ptsb = np.array([kpt.pt for kpt in kptsb])
        matcher = cv2.BFMatcher()
        mms = matcher.knnMatch(desa,desb,k=2)
        ms = [m1 for m1,m2 in mms if m1.distance < m2.distance * loweRatio]
        pts1a = ptsa[[m.queryIdx for m in ms]]
        pts1b = ptsb[[m.trainIdx for m in ms]]
        Fcv,mask = cv2.findFundamentalMat(pts1a, pts1b, cv2.FM_RANSAC, fmReprojThresh)
        mask = mask.squeeze()
        print(' - Matches ({} kpt) ({} lowe) ({} epi)'.format(len(mms), len(ms), mask.sum()))
        pts2a = pts1a[mask==1]
        pts2b = pts1b[mask==1]

        # Debug matching
        if debugMatching:
            baseImg, scale = np.concatenate((imga,imgb), 1), 1
            baseImg, scale = cv2.resize(baseImg, (0,0), fx=.5, fy=.5), scale * .5
            w = baseImg.shape[1]//2
            for pt in ptsa: cv2.circle(baseImg, (  int(scale*pt[0]), int(scale*pt[1])), 1, (0,255,0), -1)
            for pt in ptsb: cv2.circle(baseImg, (w+int(scale*pt[0]), int(scale*pt[1])), 1, (0,255,0), -1)
            for pta, ptb in zip(pts1a, pts1b):
                cv2.line(baseImg,
                        (int(scale*pta[0]), int(scale*pta[1])),
                        (int(w+scale*ptb[0]), int(scale*ptb[1])), (120,2,200), 1)
            for pta, ptb in zip(pts2a, pts2b):
                cv2.line(baseImg,
                        (int(scale*pta[0]), int(scale*pta[1])),
                        (int(w+scale*ptb[0]), int(scale*ptb[1])), (202,255,0), 1)
            dimg = baseImg
            cv2.imshow('dimg', dimg)
            cv2.waitKey(0)

        # pts2a = pts2a - pts2a.mean(0)
        # pts2b = pts2b - pts2b.mean(0)
        F = fundamental_from_pts(pts2a, pts2b)
        Ha, Hb, rectifyMask = find_matching_transformation(F, pts2a, pts2b, **matchingKw)
        h,w = imga.shape[:2]

        ra = cv2.warpPerspective(imga,Ha, (w,h))
        rb = cv2.warpPerspective(imgb,Hb, (w,h))

        h_ptsa = apply_homog(ptsa,Ha)
        h_ptsb = apply_homog(ptsb,Hb)
        h_pts2a = apply_homog(pts2a,Ha)
        h_pts2b = apply_homog(pts2b,Hb)

        d = (h_pts2a - h_pts2b)
        print(' - RMSE in each axis, of original points', np.sqrt((d*d).sum(0)) / d.shape[0])
        print(' - RMSE in each axis, of masked   points', np.sqrt((d*d)[rectifyMask==1].sum(0)) / rectifyMask.sum())
        print(' - In above, y should be very small!')
        print(' - std  in each axis, of masked   points', np.std(d[rectifyMask==1], axis=0))
        print(' - max pt diffs:', abs(d[rectifyMask==1]).max(0))

        if debugRectification:

            baseImg, scale = np.concatenate((ra,rb), 1), 1
            baseImg, scale = cv2.resize(baseImg, (0,0), fx=.5, fy=.5), scale * .5
            w = baseImg.shape[1]//2
            for pt in h_ptsa: cv2.circle(baseImg, (  int(scale*pt[0]), int(scale*pt[1])), 1, (0,255,0), -1)
            for pt in h_ptsb: cv2.circle(baseImg, (w+int(scale*pt[0]), int(scale*pt[1])), 1, (0,255,0), -1)
            for pta, ptb in zip(h_pts2a, h_pts2b):
                cv2.line(baseImg,
                        (int(scale*pta[0]), int(scale*pta[1])),
                        (int(w+scale*ptb[0]), int(scale*ptb[1])), (202,255,0), 1)
            for pta, ptb in zip(h_pts2a[rectifyMask==1], h_pts2b[rectifyMask==1]):
                cv2.line(baseImg,
                        (int(scale*pta[0]), int(scale*pta[1])),
                        (int(w+scale*ptb[0]), int(scale*ptb[1])), (0,255,0), 1)
            for y in range(0,baseImg.shape[0]-1,64):
                baseImg[y] //= 2

            dimg = baseImg
            cv2.imshow('dimg', dimg)
            cv2.waitKey(0)

        meta = {}
        meta['intrina'] = intrina
        meta['intrinb'] = intrinb
        meta['rawImga'] = imga
        meta['rawImgb'] = imgb
        meta['imga'] = ra
        meta['imgb'] = rb
        meta['F'] = F
        meta['Ha'] = Ha
        meta['Hb'] = Hb
        meta['inlyingPtsa'] = pts2a
        meta['inlyingPtsb'] = pts2b
        return Scene(meta)



if __name__ == '__main__':
    # f=np.array((3000,3000, 3000*.5, 4000*.5, 3000,4000)) # fxfy cxcy wh
    # f=np.array((3000,3000, 3000*.5, 4000*.5, 3000,4000))*.5 # fxfy cxcy wh
    # Scene.fromImagePair('../../data/mike1.jpg', '../../data/mike2.jpg', intrina=f, intrinb=f)

    # See calibrateGraphPaper.py for how to get these numbers
    intrin = Intrinsics(f=(2986.120336715084,2991.4386183123625), c=(1474.1917881681275,2009.588907222042), dc=(-0.10090114785478849,0.12932784087791988), wh=(3000,4000)).resize((1500,2000))

    if 0:
        scene = Scene.fromImagePair('data/mike1.jpg', 'data/mike2.jpg', intrina=intrin, intrinb=intrin, secondStageThresh=2)
        scene.save('data/sceneMike.pt')
    if 0:
        scene = Scene.fromImagePair('data/strathmore1.jpg', 'data/strathmore2.jpg', intrina=intrin, intrinb=intrin, secondStageThresh=1, siftPoints=5000, loweRatio=.8)
        scene.save('data/sceneStrathmore.pt')
    if 1:
        # f=np.array((3200,3200, 3000*.5, 4000*.5, 3000,4000))*.5 # fxfy cxcy wh
        scene = Scene.fromImagePair('data/night1.jpg', 'data/night2.jpg', intrina=intrin, intrinb=intrin, secondStageThresh=2, siftPoints=5000, loweRatio=.8)
        scene.save('data/sceneNight.pt')
    if 0:
        scene = Scene.fromImagePair('data/graphPaper/20221020_122347.jpg', 'data/graphPaper/20221020_122359.jpg', intrina=intrin, intrinb=intrin, secondStageThresh=2, siftPoints=500, loweRatio=.9)
        scene.save('data/scenePaper.pt')

    '''
    Okay we have our rectification ... now what how does disparity relate to depth?
    Well, we can transfer back to original image by applying inverse homograpghies.
    Then we have intrinsics for both cameras.
    So isn't it just normal triangulation, and can be done via DLT or inhomo?
    '''

