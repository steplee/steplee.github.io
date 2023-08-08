import torch, torch.nn, torchvision, numpy as np
import cv2

class EstSkeleton_linear:
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

    def forward(self):
        out = {}
        x = self.getNode('neck_base')
        st = [('neck_base', x)]
        while st:
            label,x = st.pop()
            x = x + self.getNode(label)
            out[label] = x
            for a,b,_ in self.constraints:
                if a == label:
                    st.append((b, x))
        return out

    def optimize(self, bodyPixTlbr, obsDict, scoresDict, dimg=None):
        opt = torch.optim.Adam([self.state], lr=1e-1)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=.98)

        # The projection matrix. Transform 3d coords to pixels
        # Should probably not change aspect ratio...
        sx,sy = (bodyPixTlbr[1]-bodyPixTlbr[0]) / (self.bodyMetricTlbr[1]-self.bodyMetricTlbr[0])
        P = torch.FloatTensor([
            sx,0, 0, bodyPixTlbr[0,0] - self.bodyMetricTlbr[0,0]*sx,
            # 0,sy, 0, bodyPixTlbr[0,1] - self.bodyMetricTlbr[0,0]*sy]).reshape(2,4)
            0,-sy, 0, bodyPixTlbr[1,1] + self.bodyMetricTlbr[0,0]*sy]).reshape(2,4) # Flip Y

        for i in range(1000):
            # Build graph
            g = self.forward()

            cstLoss = None
            for a,b,d in self.constraints:
                # aa,bb = self.getNode(a), self.getNode(b)
                aa,bb = g[a], g[b]

                # print(aa,bb,d, ((bb-aa).norm() - d.norm()).pow(2))
                if cstLoss is None:
                    cstLoss = ((bb-aa).norm() - d.norm()).pow(2)
                else:
                    cstLoss = cstLoss + ((bb-aa).norm() - d.norm()).pow(2)

            # print(' - P', P)

            obsLoss = None
            for k,obs in obsDict.items():
                est = P[:2,:3] @ g[k] + P[:2,2]
                l = (est - obs).norm() * scoresDict[k]
                # l = abs(est - obs).sum() * scoresDict[k]
                # print(' - Residual', l, 'from est', est, 'obs', obs)
                if obsLoss is None: obsLoss = l
                else: obsLoss = obsLoss + l

            loss = (cstLoss + obsLoss)
            print('cstLoss',cstLoss.item(), 'obsLoss', obsLoss.item(), 'full',loss.item(), 'lr', sched.get_lr())
            loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()

            if dimg is not None:
                simg = np.zeros_like(dimg)
                st = ['neck_base']
                while st:
                    label = st.pop()
                    x = g[label].detach()
                    for a,b,_ in self.constraints:
                        if a == label:
                            y = g[b].detach()
                            st.append(b)

                            pt1 = (P[:2,:3] @ x + P[:2,2]).numpy().astype(int)
                            pt2 = (P[:2,:3] @ y + P[:2,2]).numpy().astype(int)
                            # print('line', pt1,pt2)
                            cv2.line(simg, pt1, pt2, (100,200,20), 2)

                simg = cv2.addWeighted(dimg,.9, simg,.5, 0)
                cv2.imshow('alignment', simg[...,::-1])
                cv2.waitKey(0)

