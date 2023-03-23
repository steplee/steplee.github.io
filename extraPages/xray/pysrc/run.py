import torch, torch.nn, torchvision, numpy as np
import cv2

from .skeleton_linear_1 import *

def get_model():

    from torchvision.models.detection import \
            fasterrcnn_resnet50_fpn_v2, \
            FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection import \
            keypointrcnn_resnet50_fpn, \
            KeypointRCNN_ResNet50_FPN_Weights

    if 0:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                box_score_thresh=.9).eval().cuda()
        model.weights = weights
    else:
        weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        model = keypointrcnn_resnet50_fpn(
                weights=weights,
                box_score_thresh=.9).eval().cuda()
        model.weights = weights

    return model

skeletonVertices_ = {
    0: 'eye',
    1: 'ear',
    2: 'shoulder',
    3: 'elbow',
    4: 'hand',
    5: 'hip',
    6: 'knee',
    7: 'foot' }

skeletonVertices = {
    0: 'nose',
}
skeletonVertices.update({k*2+1: 'l_'+v for k,v in skeletonVertices_.items()})
skeletonVertices.update({k*2+2: 'r_'+v for k,v in skeletonVertices_.items()})
skeletonVerticesInv = {v:k for k,v in skeletonVertices.items()}

skeletonIndices_ = [
        'nose', 'l_eye',
        'l_eye', 'l_ear',
        'l_ear', 'l_shoulder',
        'l_shoulder', 'l_elbow',
        'l_elbow', 'l_hand',
        'l_shoulder', 'l_hip',
        'l_hip', 'l_knee',
        'l_knee', 'l_foot',

        'nose', 'r_eye',
        'r_eye', 'r_ear',
        'r_ear', 'r_shoulder',
        'r_shoulder', 'r_elbow',
        'r_elbow', 'r_hand',
        'r_shoulder', 'r_hip',
        'r_hip', 'r_knee',
        'r_knee', 'r_foot',

        'l_shoulder', 'r_shoulder',
        'l_hip', 'r_hip' ]
skeletonIndices = [
        (skeletonVerticesInv[a],skeletonVerticesInv[b]) for a,b in \
                zip(skeletonIndices_[0::2], skeletonIndices_[1::2],)
]

# pytorch IK solver. TODO.
# NOTE: Ball and socket joints required a full (3) set of rotation angles
# NOTE: The shoulder does supinate, but the elbow does too.
class EstSkeleton_angled:
    # head: 3
    # shoulder: 3 x 2
    # elbow: 3 x 2     [restricted]
    # hand: 3 x 2
    # hip: 3 x 2
    # knee: 3 x 2      [restricted]
    # foot: 3 x 2      [restricted]

    def addBallSocket(self, x, d, k):
        x = torch.cat((self.state, torch.zeros(3)), 0)
        d[k] = (x.size(0)-3, x.size(0))


    def __init__(self):
        # FIXME: Actually don't use same skeleton as the observed targets (that is ok)
        x,d = torch.zeros(0), {}
        x,d = self.addBallSocket(x,d,'neck_base')
        x,d = self.addBallSocket(x,d,'l_shoulder')
        x,d = self.addBallSocket(x,d,'l_elbow')
        x,d = self.addBallSocket(x,d,'l_hand')
        x,d = self.addBallSocket(x,d,'mid')
        x,d = self.addBallSocket(x,d,'waist')
        x,d = self.addBallSocket(x,d,'l_hip')
        x,d = self.addBallSocket(x,d,'l_knee')
        x,d = self.addBallSocket(x,d,'l_foot')

        self.connect

        self.label = d
        self.state = x.clone()

    def angle(self, name):
        return self.state[idx]



def run_model(model, img):
    with torch.no_grad():
        x = torch.from_numpy(img).cuda().float().div_(255).unsqueeze_(0).permute(0,3,1,2)
        x = model.weights.transforms()(x)
        return model(x)

def show_viz(img, boxes, keypointss, kscores, show=True):
    for box in boxes.cpu().numpy():
        pt1 = box[:2].astype(int)
        pt2 = box[2:].astype(int)
        cv2.rectangle(img, pt1,pt2, (0,255,0), 1)

    for keypoints in keypointss.cpu().numpy():
        for i, kpt in enumerate(keypoints):
                if kpt[2] > .5:
                    pt = kpt[:2].astype(int)
                    pt1 = kpt[:2].astype(int) + (1,0)
                    score = kscores.view(-1)[i].sigmoid().item()
                    c = (255-int(score*255),int(score*255),0)
                    cv2.circle(img, pt, 4, c, 1)
                    print(pt,score)
                    cv2.putText(img, str(i), pt1, 0, .6, (0,0,0))
                    cv2.putText(img, str(i), pt, 0, .6, c)

    for keypoints in keypointss.cpu().numpy():
        for (a,b) in skeletonIndices:
            if keypoints[a,2].item() > .5 and keypoints[b,2].item() > .5:
                pta = keypoints[a,:2].astype(int)
                ptb = keypoints[b,:2].astype(int)
                scorea = kscores.view(-1)[a].sigmoid().item()
                scoreb = kscores.view(-1)[b].sigmoid().item()
                score = scorea * scoreb
                c = (255-int(score*255),int(score*255),0)
                cv2.line(img, pta, ptb, c)

    if show:
        cv2.imshow('img', img[...,::-1])
        cv2.waitKey(0)
    return img


def run_model_and_viz():
    img = np.copy(cv2.imread('data/me.jpg')[...,[2,1,0]], 'C')
    # img = np.copy(cv2.imread('data/climb.jpg')[...,[2,1,0]], 'C')
    # img = np.copy(cv2.imread('data/sit.jpg')[...,[2,1,0]], 'C')
    img = cv2.resize(img,(0,0),fx=.5,fy=.5)

    m = get_model()
    out = run_model(m, img)[0]
    print(out)
    show_viz(img, out['boxes'], out['keypoints'], out['keypoints_scores'])


def run_ik():

    img = np.copy(cv2.imread('data/me.jpg')[...,[2,1,0]], 'C')
    img = np.copy(cv2.imread('data/climb.jpg')[...,[2,1,0]], 'C')
    img = cv2.resize(img,(0,0),fx=.5,fy=.5)
    m = get_model()
    out = run_model(m, img)[0]
    dimg = show_viz(img, out['boxes'], out['keypoints'], out['keypoints_scores'],show=False)

    print(' - boxes', out['boxes'].shape)
    print(' - keypoints', out['keypoints'].shape)
    print(' - keypoints_scores', out['keypoints_scores'].shape)

    # Form observations. Our skeleton model does not necessarily match the KeypointRCNN/COCO one.
    kpts = out['keypoints'][0].cpu().detach()
    kptScores = out['keypoints_scores'][0].cpu().detach().sigmoid()
    obs,obsScores = {}, {}
    directCopy = 'l_shoulder r_shoulder l_elbow r_elbow l_hand r_hand l_hip r_hip l_knee r_knee l_foot r_foot'.split(' ')
    for k in directCopy:
        obs[k] = kpts[skeletonVerticesInv[k]][:2]
        obsScores[k] = kptScores[skeletonVerticesInv[k]]

    obs['neck_base'] = (kpts[skeletonVerticesInv['l_shoulder']][:2] + \
                       kpts[skeletonVerticesInv['r_shoulder']][:2]) * .5
    obsScores['neck_base'] = (kptScores[skeletonVerticesInv['l_shoulder']] * \
                       kptScores[skeletonVerticesInv['r_shoulder']]).sqrt()


    # Optimize, showing results along the way
    es = EstSkeleton_linear()
    # pixTlbr = torch.FloatTensor([0,0,1,1]).view(2,2)
    pixTlbr = torch.stack((
            kpts.min(0).values,
            kpts.max(0).values))[:,:2].cpu()
    print(' - Pix Tlbr:\n', pixTlbr)
    es.optimize(pixTlbr, obs, obsScores, dimg=dimg)



# run_model_and_viz()
run_ik()

