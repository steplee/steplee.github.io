import torch, numpy as np

from .render import *

class Skeleton:
    def __init__(self):
        constraints = [
                [None, 'neck_base', (0,0,2)],
                ['neck_base', 'head', (0,0,.3)],
                ['neck_base', 'mid', (0,0,-.5)],
                ['mid', 'groin', (0,0,-.5)],

                ['groin', 'l_hip', (-.2,0,-.05)],
                ['l_hip', 'l_knee', (0,0,-.5)],
                ['l_knee', 'l_foot', (0,0,-.5)],

                ['neck_base', 'l_shoulder', (-.2,0,-.025)],
                ['l_shoulder', 'l_elbow', (-.5,0,0)],
                ['l_elbow', 'l_hand', (-.5,0,0)]
                ]
        for (a,b,d) in constraints:
            if (a is not None and a.startswith('l_')) or b.startswith('l_'):
                constraints.append([a.replace('l_','r_'), b.replace('l_','r_'), (-d[0],d[1],d[2])])

        self.constraints = []
        for (a,b,d) in constraints:
            d = torch.FloatTensor(d)
            if a is not None: self.constraints.append((a,b,d))
            else: self.root, self.rootOffset = b, d

        self.reverseTree = {}
        for (a,b,d) in constraints:
            assert b not in self.reverseTree, 'no cycles allowed'
            if a is not None:
                self.reverseTree[b] = a

        self.forwardLists = {}
        for (a,b,d) in constraints:
            if a is not None:
                if a not in self.forwardLists: self.forwardLists[a] = [b]
                else: self.forwardLists[a].append(b)

        # print(self.reverseTree)
        self.defaultState = {}
        for a,b,d in self.constraints:
            self.defaultState[b] = d

        self.label2ind = {}
        self.ind2label = {}
        for k,_ in self.dfs(self.defaultState):
            self.label2ind[k] = len(self.label2ind)
        for k,v in self.label2ind.items():
            self.ind2label[v] = k

    def dfs(self, state):
        st = [(self.root, self.rootOffset)]
        while st:
            a,x = st.pop()
            yield a,x
            if a in self.forwardLists:
                for b in self.forwardLists[a]:
                    d = state[b]
                    st.append((b, x+d))


    def getVertsAndInds(self, stateTensor):
        verts = stateTensor
        inds = []
        for a,b,_ in self.constraints:
            if a is not None:
                inds.append((self.label2ind[a], self.label2ind[b]))
        inds = torch.from_numpy(np.array(inds,dtype=np.int16))
        return verts, inds



sk = Skeleton()
# print(sk.defaultState)
state = {k:v for k,v in sk.dfs(sk.defaultState)}
state = torch.stack([state[v] for k,v in sk.ind2label.items()])
print(state)


verts,inds = sk.getVertsAndInds(state)
verts = verts.cpu().numpy()
inds = inds.cpu().numpy()
print(sk.getVertsAndInds(0))

r = SurfaceRenderer(1024,1024)
r.init(True)
while True:

    r.startFrame()
    r.render()

    glBegin(GL_LINES)
    glColor4f(1,0,0,1); glVertex3f(0,0,0); glVertex3f(1,0,0);
    glColor4f(0,1,0,1); glVertex3f(0,0,0); glVertex3f(0,1,0);
    glColor4f(0,0,1,1); glVertex3f(0,0,0); glVertex3f(0,0,1);
    glEnd()

    glBegin(GL_LINES)
    glColor4f(1,1,1,1)
    for a,b in inds:
        glVertex3f(verts[a,0], verts[a,1], verts[a,2])
        glVertex3f(verts[b,0], verts[b,1], verts[b,2])
    glEnd()

    r.endFrame()



