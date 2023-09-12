from .skeleton import skeleton
from ...est2d.run import FasterRcnnModel, get_resnet50, run_model_and_viz, get_coco_skeleton

import torch, torch.nn, torchvision, numpy as np
import cv2

def get_coco_skeleton():
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

    return Skeleton(skeletonIndices, skeletonVerticesInv)

def get_resnet50():
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

def run_model(model, img):
    with torch.no_grad():
        x = torch.from_numpy(img).cuda().float().div_(255).unsqueeze_(0).permute(0,3,1,2)
        x = model.weights.transforms()(x)
        return model(x)


class FasterRcnnModel():
    def __init__(self, model):
        self.model = model
        # self.skeletonIndices, self.skeletonVerticesInv = get_coco_skeleton()
        self.cocoSkeleton = get_coco_skeleton()
        print('Faster RCNN Skeleton:')
        from pprint import pprint
        # pprint(self.skeletonVerticesInv)

    def forward(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).cuda().float().div_(255).unsqueeze_(0).permute(0,3,1,2)
            else:
                assert isinstance(x, torch.Tensor)
                assert x.dtype == torch.float32 or x.dtype == torch.uint8
                if x.dtype == torch.float32:
                    x = x.cuda().unsqueeze_(0).permute(0,3,1,2)
                else:
                    x = x.cuda().float().div_(255).unsqueeze_(0).permute(0,3,1,2)

            x = self.model.weights.transforms()(x)
            return self.model(x)

    def __call__(self, x): return self.forward(x)

    def show_viz(self, img, boxes, keypointss, kscores, show=True):
        for box in boxes.cpu().numpy():
            pt1 = box[:2].astype(int)
            pt2 = box[2:].astype(int)
            cv2.rectangle(img, pt1,pt2, (0,255,0), 1)

        textImg = img*0
        circImg = img*0

        for keypoints in keypointss.cpu().numpy():
            for i, kpt in enumerate(keypoints):
                    if kpt[2] > .5:
                        pt = kpt[:2].astype(int)
                        pt1 = kpt[:2].astype(int) + (1,0)
                        score = kscores.view(-1)[i].sigmoid().item()
                        c = (255-int(score*255),int(score*255),0)
                        cv2.circle(circImg, pt, 4, c, 1)
                        # print(pt,score)
                        cv2.putText(textImg, str(i), pt1, 0, .6, (0,0,0))
                        cv2.putText(textImg, str(i), pt, 0, .6, c)

        for keypoints in keypointss.cpu().numpy():
            for (a,b) in self.cocoSkeleton.indices:
                if keypoints[a,2].item() > .5 and keypoints[b,2].item() > .5:
                    pta = keypoints[a,:2].astype(int)
                    ptb = keypoints[b,:2].astype(int)
                    scorea = kscores.view(-1)[a].sigmoid().item()
                    scoreb = kscores.view(-1)[b].sigmoid().item()
                    score = scorea * scoreb
                    c = (255-int(score*255),int(score*255),0)
                    cv2.line(img, pta, ptb, c)

        img = cv2.addWeighted(img, 1, textImg, .3, 0)
        img = cv2.addWeighted(img, 1, circImg, .6, 0)

        if show:
            cv2.imshow('img', img[...,::-1])
            cv2.waitKey(0)
        return img


if __name__ == '__main__':
    # img = np.copy(cv2.imread('data/me.jpg')[...,[2,1,0]], 'C')
    img = np.copy(cv2.imread('data/me2.jpg')[...,[2,1,0]], 'C')
    img = cv2.resize(img,(0,0),fx=.5,fy=.5)

    m = get_resnet50()
    m = FasterRcnnModel(m)

    run_model_and_viz(img)
