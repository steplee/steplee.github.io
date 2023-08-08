import torch, torch.nn, torchvision, numpy as np
import cv2

root = 'neck_base'
nodes = [
        # ('l_elbow', 'l_hand', 'hingeLateral', {'hingeMinMax': (0,150), 'latMinMax': (
]

# NOTE: Actually don't use the different joints: they are all general to varying degrees
# Instaed just specify joints and
#     - the length of the bone
#     - minMax rotation ranges around each axis
#     - stiffness around each axis, perhaps by seperate ranges as well.
nodes = [
    # Shoulder is rigidly attached to neck base
    ('neck_base', 'l_shoulder', .15, {'minMax': [(0,0), (-89.9,90.1), (0,0)]}),

    ('l_shoulder', 'l_elbow', .3, {'minMax': [(-90,90), (-90,60), (-50,90)]}),
    ('l_elbow', 'l_hand', .3, {'minMax': [(-2,2), (-90,90), (0,130)]}),
]

class EstSkeleton_angle_2:

    # Rigid body, fixed
    def addJoint0(self, pred, chld, d):
        pass

    # Basic hinge joint (e.g. the last two parts of each finger)
    # Rotates in exactly one dim
    def addHingeJoint(self, pred, chld, d):
        pass

    # Hinge join with longitudinal rotation (e.g. elbow)
    def addHingeJointLng(self, pred, chld, d, hingeMinMax, lngMinMax):
        pass

    # Hinge join with lateral rotation (e.g. base part of each finger)
    def addHingeJointLat(self, pred, chld, d, hingeMinMax, lateralMinMax):
        pass

    # Fully general join (e.g. ball-and-socket: shoulder)
    def addGeneralJoint(self, pred, chld, d):
        pass

    def addNode(self, k, pred, d):
        d = torch.FloatTensor(d)
        self.state = torch.cat((self.state, d), 0)
        self.label[k] = (self.state.size(0)-3, self.state.size(0))
        if pred is not None:
            self.constraints.append((pred,k,d))

    def connect(self, lst, a,b, d):
        lst.append((a,b,torch.FloatTensor(d)))

    def __init__(self):
        self.state = torch.zeros(0)
        self.label = {}
        self.constraints = []

        self.addNode('neck_base', None, (0,0,0))
        self.addNode('mid', 'neck_base', (0,-.43,0))
        self.addNode('groin', 'mid', (0,-.33,0))
        # for a,s in zip('lr',[-1,1]):
        for a,s in zip('lr',[1,-1]): # FLIP SIGN, because we view head on
            self.addNode(a+'_shoulder', 'neck_base', (s*.1,0,0))
            self.addNode(a+'_elbow', a+'_shoulder', (s*.15,0,0))
            self.addNode(a+'_hand', a+'_elbow', (s*.2,0,0))
            self.addNode(a+'_hip', 'groin', (s*.2,-.05,0))
            self.addNode(a+'_knee', a+'_hip', (0,-.5,0))
            self.addNode(a+'_foot', a+'_knee', (0,-.5,0))

        self.state = torch.autograd.Variable(self.state,True)
        graph = self.forward()
        print(self.constraints, graph)

        self.bodyMetricTlbr = torch.stack((
                torch.stack([v for k,v in graph.items()]).min(0).values,
                torch.stack([v for k,v in graph.items()]).max(0).values))[:,:2]
        # print(torch.stack([v for k,v in graph.items()]))
        # print(self.bodyMetricTlbr)

    def getNode(self, k): return self.state[self.label[k][0]:self.label[k][1]]
