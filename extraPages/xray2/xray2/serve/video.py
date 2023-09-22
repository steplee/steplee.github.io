from .serve import BaseHandler, run

import torch, numpy as np
from xray2.skeleton.coco import get_coco_skeleton
from xray2.data.posePriorDataset import PosePriorDataset, Map_PP_to_Coco_v1

def get_example(T):
    cocoSkel = get_coco_skeleton()
    dset = PosePriorDataset('/data/human/posePrior/procAugment1Stride2/data.npz')
    mapToCoco = Map_PP_to_Coco_v1(cocoSkel.jointMap, dset.joints)

    pp_pos, cc_pos = [], []

    for i in range(T):
        s = dset[i].unsqueeze(0)
        pp_pos.append(s[0].numpy().reshape(-1))
        cc_pos.append(mapToCoco(s)[0].numpy().reshape(-1))

    d = {
        'skeletons': {

            'posePrior': {
                'indices': dset.inds.numpy().tolist(),
                'jointNames': list(dset.joints.keys()),
                'N': 1,
                'T': T,
                'positions': np.stack(pp_pos).reshape(-1).tolist()
            },

            'coco': {
                'indices': cocoSkel.indices.tolist(),
                'jointNames': list(cocoSkel.jointMap.keys()),
                'N': 1,
                'T': T,
                'positions': np.stack(cc_pos).reshape(-1).tolist()
            },
        }
    }
    # print(d)
    for n,skel in d['skeletons'].items():
        print(n,T*3*len(skel['jointNames']), len(skel['positions']))
        assert T*3*len(skel['jointNames']) == len(skel['positions'])


    return d





class VideoHandler(BaseHandler):
    def __init__(self, *args, **kw):
        super().__init__(*args,**kw)

    def get_skeletons(self, body):
        N = int(body.get('N', 1))

        out = { 'skeletons': {
            'first': {
                'indices': [0,1, 1,2],
                'jointNames': ['foot', 'hip', 'chest'],
                'T': 2,
                'N': N,
                'positions': [
                    -.1,0,0, -.1,1,0, .5,2,0,
                    .1,0,0, .1,1,0, .5,2,0,

                    ] * N,
                }
            }}

        modelName = body.get('model', 'dataset')
        if modelName == '' or modelName == 'dataset':
            out = get_example(N)

        return out

if __name__ == '__main__':
    run(VideoHandler)
