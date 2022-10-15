import torch, numpy as np

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
        meta['inlyingPtsa'] = pts2a
        meta['inlyingPtsb'] = pts2b
        return Scene(meta)



if __name__ == '__main__':
    f=np.array((1024,1024, 3000*.5, 4000*.5))
    # Scene.fromImagePair('../../data/mike1.jpg', '../../data/mike2.jpg', intrina=f, intrinb=f)
    scene = Scene.fromImagePair('data/mike1.jpg', 'data/mike2.jpg', intrina=f, intrinb=f, secondStageThresh=2)
    scene.save('data/sceneMike.pt')

    '''
    Okay we have our rectification ... now what how does disparity relate to depth?
    Well, we can transfer back to original image by applying inverse homograpghies.
    Then we have intrinsics for both cameras.
    So isn't it just normal triangulation, and can be done via DLT or inhomo?
    '''

