import cv2, numpy as np
from ..models import *
from ..viz import LD_Renderer_2

from ...data.posePriorDataset import PosePriorDataset, DataLoader, Map_PP_to_Coco_v1

from ...est2d.run import FasterRcnnModel, get_resnet50, run_model_and_viz, get_coco_skeleton
from ...solver.pnp import recover_camera, recover_camera_batch, q_to_matrix1

def crop_image(img):
    h,w = img.shape[:2]
    e = (min(h,w) // 2) * 2
    img = img[h//2-e//2:h//2+e//2, w//2-e//2:w//2+e//2]
    img = cv2.resize(img, (512,512))
    return img

def extract_kpts(kptModel, img):
    pass

# This is *not* the real median, but close enough.
def pick_median_skeleton_per_time(xs):
    d = xs - xs.mean(1, keepdim=True)
    d = d.norm(dim=2)
    mind = d.min(1)
    # return xs.gather(1,mind.indices)
    ys = []
    for i,idx in enumerate(mind.indices.view(-1)): ys.append(xs[i,idx])
    return torch.stack(ys,0)


def solve_cams(xsPp, xsCoco, obsPts, intrins):

    # q0=torch.FloatTensor((1,0,0,0))
    q0=torch.FloatTensor((0,1,0,0))
    t0=torch.FloatTensor((0,1,2.7))

    N = xsCoco.size(0)
    print(' - Recovering cameras...')

    # camPoses = recover_camera(camWh, camFxy, obsPts,
                                # xsCoco,
                                # initialQ=q0,initialEye=t0).view(N,-1)
    # camCxy = intrins[:, :2]
    # camFxy = intrins[:, 2:]
    camPoses = recover_camera_batch(
            # camCxy, camFxy, obsPts,
            intrins, obsPts,
                                xsCoco,
                                initialQ=q0,initialEye=t0,
                                iniHessWeightR=500,
                                iniHessWeightT=400,
                                ).view(1,-1).repeat(N,1)

    print(' -                   ... Done')
    return camPoses

class LD_Sampler:
    def __init__(self,
                 model,
                 kptModel,
                 dataset,
                 ppToCoco,
                 opts,
                 ):
        self.model = model
        self.kptModel = kptModel
        self.dataset = dataset
        self.ppToCoco = ppToCoco
        self.opts = opts

        self.initialPose = self.dataset[0]
        self.steps = 0


    def step(self, img):

        B, T = 32, 30

        # Extract 2d points, format as conditional input
        out, vimg = run_model_and_viz(self.kptModel, img, show=False)
        cv2.imshow('vimg', cv2.cvtColor(vimg,cv2.COLOR_RGB2BGR)); cv2.waitKey(1)
        kpts = out['keypoints'][0]
        kpts = kpts[:, :2].reshape(-1,2).cuda()
        kpts = kpts.view(1,kpts.size(0),kpts.size(1)).repeat(B,1,1)
        assert img.shape[0] == img.shape[1]
        camWh = img.shape[0]


        self.camWh = camWh
        self.camCxy0 = camWh*.5
        self.camFxy0 = (camWh*.5) / np.tan(np.deg2rad(43) * .5)

        camCxy = self.camCxy0,self.camCxy0
        camFxy = self.camFxy0,self.camFxy0

        SELECT_BBOX=True
        if SELECT_BBOX:
            bbox = out['boxes'][0].cpu().view(2,2)
            lo,hi = bbox.min(0).values, bbox.max(0).values
            c = (lo+hi)*.5
            s = (hi-lo).max()
            # lo,hi = lo - s*.5, hi + s*.5
            # camCxy = c
            # camFxy = ((s/camWh) * self.camFxy0).repeat(2) # WARNING: I don't think this is correct
            # print(camCxy,camFxy)

            # z = .5 + 1.0*(kpts.cpu()-c) / s
            z = (kpts) / (camWh)
            z = z.view(B,-1).cuda()

            # tl,br = lo/camWh, hi/camWh
            tl,br = (lo/camWh)*2-1, (hi/camWh)*2-1

            lo,hi = lo.long(), hi.long()
            croppedImg = cv2.resize(vimg[lo[1]:hi[1], lo[0]:hi[0]], (camWh,camWh))

        else:
            z = (kpts) / (camWh)
            z = z.view(B,-1)
            croppedImg = vimg
            # tl,br = torch.zeros(2), torch.ones(2)
            tl,br = -torch.ones(2), torch.ones(2)

        if 1:
            sz=256+128
            pad=48
            img = np.zeros((2*pad+sz,2*pad+sz,3),dtype=np.uint8)
            c = (90,90,90)
            img[pad:pad+sz, pad] = c
            img[pad:pad+sz, sz+pad] = c
            img[pad,pad:pad+sz] = c
            img[sz+pad,pad:pad+sz] = c
            pts = z[0].view(-1,2)
            for j,p in enumerate(pts):
                xx,yy = (p*sz+pad).cpu().numpy().astype(int)
                img[yy-1:yy+1,xx-1:xx+1] = (((j+999)*7779)%255, ((j+32)*99913)%255, ((j+132)*7993)%255,)
            cv2.imshow('pts',img); cv2.waitKey(10)

        # Start
        x = self.initialPose.unsqueeze(0).repeat(B,1).cuda()
        # x += torch.randn_like(x) * 1.5
        x += torch.randn_like(x) * .5

        corruptFactor = .7
        stepFactor    = .9
        xs = [x.clone()]

        for i,t in enumerate(np.linspace(1,0,T)):
            nl = torch.FloatTensor((t,)).repeat(B,1).to(x.device)

            sig = self.model.mapNoiseLevelToSigma(nl)
            if self.steps == 0: print(t, '->', sig[0,0].item())
            n = torch.randn_like(x) * sig
            x += corruptFactor * n

            pdx = self.model(x, nl, z)
            x += stepFactor * pdx
            xs.append(x.clone())

        xs = torch.stack(xs,0)
        # xs = xs.permute(1,0,2)
        print(f' - Final shape: {xs.shape}')
        self.steps += 1
        # return dict(xs=xs,z=z,camWh=(camWh,camWh),camFxy=(camFxy,camFxy))
        return dict(xs=xs,z=kpts,
                    croppedImg=croppedImg,
                    camWh=camWh,
                    camFxy=camFxy,
                    camCxy=camCxy,
                    tl=tl,br=br
                    )




def main():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--image', default='')
    p.add_argument('--video', default='')
    p.add_argument('--skip', default=0, type=int)
    p.add_argument('--stride', default=20, type=int)
    p.add_argument('--nframes', default=20, type=int)
    p.add_argument('--sliceY', default=(0,-1), nargs=2, type=int)
    args = p.parse_args()

    model = get_model(dict(load=args.model))['model'].eval().cuda()
    datasetPath = model.meta['dset']
    dataset = PosePriorDataset(datasetPath)

    kptModel = FasterRcnnModel(get_resnet50().cuda().eval())

    # Create sampler.
    from ...est2d.run import get_coco_skeleton
    coco_inds, cocoJoints = get_coco_skeleton()
    coco_inds = np.array(coco_inds,dtype=np.uint16)
    ppToCoco = Map_PP_to_Coco_v1(cocoJoints, dataset.joints)
    opts = {}
    ld = LD_Sampler(model, kptModel, dataset, ppToCoco, opts)

    xsPerBatchPerTime = []
    obsPts = []
    intrins = []

    # Load and run on data. Collect results.
    #
    if len(args.image) > 0:
        img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        img = img[args.sliceY[0]:args.sliceY[1]]
        img = crop_image(img)
        sampled = ld.step(img)
        xsPerBatchPerTime = sampled['xs'].cpu()
        intrins.append(torch.cat((sampled['camCxy'], sampled['camFxy']), 0))
        obsPts = [sampled['z'][-1],]*len(xsPerBatchPerTime)
    else:
        assert len(args.video) > 0, "Must provide --image or --video"
        frames=[]
        vcap = cv2.VideoCapture(args.video)
        for i in range(999999):
            good,frame = vcap.read()
            if not good: break
            if i > args.skip and i % args.stride == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = img[args.sliceY[0]:args.sliceY[1]]
                img = crop_image(img)
                sampled = ld.step(img)

                x = sampled['xs'][-1]
                xsPerBatchPerTime.append(x)
                # frames.append(img)
                frames.append(sampled['croppedImg'])
                obsPts.append(sampled['z'][-1])

                K = torch.eye(3)
                K[0,0] = sampled['camFxy'][0]
                K[1,1] = sampled['camFxy'][1]
                K[0,2] = sampled['camCxy'][0]
                K[1,2] = sampled['camCxy'][1]
                B = torch.eye(3)
                tl,br = sampled['tl'], sampled['br']
                B[0,0] = (br[0]-tl[0]) * .5
                B[1,1] = (br[1]-tl[1]) * .5
                B[0,2] = tl[0]*.5+.5
                B[1,2] = tl[1]*.5+.5
                K = K @ B
                # intrins.append(torch.cat((sampled['camCxy'], sampled['camFxy']), 0))
                intrins.append(K)

                if len(frames) > args.nframes: break
        xsPerBatchPerTime = torch.stack(xsPerBatchPerTime, 0).cpu()

    obsPts = torch.stack(obsPts,0).cpu()
    intrins = torch.stack(intrins,0).cpu()

    xsPp = pick_median_skeleton_per_time(xsPerBatchPerTime)
    N = xsPp.size(0)
    print(xsPerBatchPerTime.shape)
    print(xsPp.shape)
    xsCoco = ppToCoco(xsPp).view(N,-1,3)
    xsPp = xsPp.view(N,-1,3)
    obsPts = obsPts.view(N,-1,2)
    print(xsPp.shape,obsPts.shape)
    cams_ = solve_cams(xsPp, xsCoco, obsPts, intrins)
    cams = []
    for cam in cams_:
        t,q = cam[-3:], cam[:4]
        R = q_to_matrix1(q)
        V = torch.eye(4)
        V[:3,:3] = R
        V[:3,3] = t
        cams.append(V)
    cams = torch.stack(cams, 0)
    cams = cams[0:1].repeat(cams.size(0),1,1)

    # print('Cams:\n',cams)
    print('Cams[0]:\n',cams[0])

    viz = LD_Renderer_2(1024,1536, dataset)
    viz.init(True)

    viz.setJoints('pred', xsPerBatchPerTime.cpu().numpy())

    camFxy0 = (ld.camFxy0, ld.camFxy0)
    camWh = torch.FloatTensor((ld.camWh, ld.camWh))
    # viz.setCameras(cams,camFxy0,camWh,imgs=frames,pts3d=xsCoco)
    viz.setCameras(cams,intrins,imgs=frames,pts3d=xsCoco, wh=camWh)
    # viz.setCameras(cams,camFxy,camWh,imgs=frames,pts3d=xsPp)


    # viz.setCameras()

    viz.run()

with torch.no_grad():
    if __name__ == '__main__': main()


