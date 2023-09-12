from ..render import *

frustum_inds = np.array((
    0,1, 1,2, 2,3, 3,0,
    4+0,4+1, 4+1,4+2, 4+2,4+3, 4+3,4+0,
    0, 4+0, 1, 4+1, 2, 4+2, 3, 4+3), dtype=np.uint16)
frustum_pts = np.array((
    -1,-1,0,
     1,-1,0,
     1, 1,0,
    -1, 1,0,
    -1,-1,1,
     1,-1,1,
     1, 1,1,
    -1, 1,1),dtype=np.float32).reshape(8,3)
    # -1, 1,1),dtype=np.float32).reshape(8,3) * (.5,.5,1)
frustum_uvs = np.array((
    0,0,
    1,0,
    1,1,
    0,1
    ),dtype=np.float32)
frustum_face_inds = np.array((0,1,2, 2,3,0), dtype=np.uint16)

def draw_frustum(eye, R, wh, uv, pts2d, pts3d):
    glEnableClientState(GL_VERTEX_ARRAY)

    if 1:
        q = pts2d - wh * .5
        q = q * (uv/wh)
        q = torch.cat((q,torch.ones_like(q[...,:1]), -1))
        q = q * 1 # choose depth

        glColor4f(1,1,1,1)

        glVertexPointer(3,GL_FLOAT,0,q)
        glDrawArrays(GL_LINES,0,len(q))


    if 1:
        w = np.array((0,0, 1,0, 1,1, 0,1),dtype=np.float32)*wh
        w = w - wh * .5
        q = w * (uv/wh)
        w = torch.cat((w,torch.ones_like(w[...,:1]), -1))
        w = w * 1 # choose depth
        glColor4f(1,1,1,.6)
        glVertexPointer(3,GL_FLOAT,0,w)
        glDrawElements(GL_LINES,frustum_inds.size,GL_UNSIGNED_SHORT,frustum_inds)

    glDisableClientState(GL_VERTEX_ARRAY)

class ExampleRendererBase(SingletonApp):
    def __init__(self, h, w):
        super().__init__((w,h), 'SurfaceRenderer')
        self.q_pressed = False
        self.n_pressed = False
        self.mprev = np.array((-1,0))
        self.mprev2 = np.array((-1,0))
        self.md = np.array((0,.0))
        self.angles = np.zeros(3, dtype=np.float32)
        self.velTrans = np.zeros(3, dtype=np.float32)
        self.accTrans = np.zeros(3, dtype=np.float32)
        self.eye = np.array((0,0,1.),dtype=np.float32)

        self.R = np.eye(3,dtype=np.float32)
        self.t = np.copy(self.eye)
        self.view = np.eye(4,dtype=np.float32)
        self.lastKey = None

    def do_init(self):
        pass

    def startFrame(self):
        self.q_pressed = False
        glutMainLoopEvent()
        self.startTime = time.time()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        add_md = 0
        if self.mprev[0] > 0 and self.mprev2[0] > 0: add_md = (self.mprev - self.mprev2)
        self.md = self.md * .9 + .001 * add_md

        self.velTrans = self.velTrans * .9 + .01 * self.accTrans
        self.angles = (self.md[1], self.md[0], 0)
        self.R = self.R @ cv2.Rodrigues(self.angles)[0].astype(np.float32)
        self.eye += self.R @ self.velTrans
        self.view[:3,:3] = self.R.T
        self.view[:3,3] = -self.R.T @ self.eye

        self.mprev[:] = -1
        self.accTrans *= 0

    def endFrame(self):
        glutSwapBuffers()
        dt = time.time() - self.startTime
        st = .011 - dt
        #if st > 0: time.sleep(st)
        st = max(0,st)
        self.lastKey = None

        return self.q_pressed

    def keyboard(self, key, x, y):
        key = (key).decode()
        if key == 'q': self.q_pressed = True
        if key == 'n': self.n_pressed = True
        if key == 'w': self.accTrans[2] = -1
        if key == 's': self.accTrans[2] = 1
        if key == 'a': self.accTrans[0] = -1
        if key == 'd': self.accTrans[0] = 1
        if key == 'e': self.accTrans[1] = -1
        if key == 'r': self.accTrans[1] = 1
        if key == 'f': self.addSign *= -1
        self.lastKey = key

    def motion(self, x, y):
        self.mprev2[:] = self.mprev
        self.mprev[:] = x,y


    def renderAxes(self):
        glBegin(GL_LINES)
        glColor4f(1,0,0,1); glVertex3f(0,0,0); glVertex3f(1,0,0);
        glColor4f(0,1,0,1); glVertex3f(0,0,0); glVertex3f(0,1,0);
        glColor4f(0,0,1,1); glVertex3f(0,0,0); glVertex3f(0,0,1);
        glEnd()

class ExampleRenderer(ExampleRendererBase):
    def __init__(self, h, w):
        super().__init__(h,w)

        self.inds,self.x = None, None
        self.addSign = 1
        self.animationTime = -1
        self.attentionMapIdx = 0
        self.cam,self.z = None,None
        self.attnMaps = None


    def render(self):
        glViewport(0, 0, *self.wh)
        glMatrixMode(GL_PROJECTION)
        n = .001
        v = .5
        u = (v*self.wh[0]) / self.wh[1]
        glLoadIdentity()
        glFrustum(-u*n,u*n,-v*n,v*n,n,100)

        #print(' - view:\n', self.view)
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(self.view.T.reshape(-1))

        glClearColor(0,0,0,1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.x is not None:
            self.renderSkeleton()
        if self.animationTime >= 0:
            self.renderSkeletonAnimated()



    def setData(self, inds, x, nx, estScore, ts=None):
        self.inds = inds.cpu().numpy().astype(np.uint16)
        self.x = x.reshape(-1)
        self.xn = nx.reshape(-1)
        self.estScore = estScore.reshape(-1)

    # ----------------------------------------------------------------------
    # Basic Skeleton
    #

    # Given a generator that loads next data and computes the corrupted and scores,
    # show the viewer until the user presses 'q'.
    # Press 'n' to go to the next data.
    def runBasicUntilQuit(self, genDataDict):
        while not self.q_pressed:
            for data in genDataDict():
                self.setData(**data)
                while not (self.n_pressed or self.q_pressed):
                    self.startFrame()
                    self.render()
                    self.endFrame()
                self.n_pressed = False
                if self.q_pressed: break
        self.q_pressed = False

    def renderSkeleton(self):
        if self.inds is None:
            return None

        self.renderAxes()

        if self.x is not None:
            glColor4f(1,1,1,1)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3,GL_FLOAT,0,self.x)
            glDrawElements(GL_LINES,self.inds.size,GL_UNSIGNED_SHORT,self.inds)
            glDisableClientState(GL_VERTEX_ARRAY)

        if self.xn is not None:
            glColor4f(1,1,0,.5)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3,GL_FLOAT,0,self.xn)
            glDrawElements(GL_LINES,self.inds.size,GL_UNSIGNED_SHORT,self.inds)
            glDisableClientState(GL_VERTEX_ARRAY)

        if self.estScore is not None:
            est_x = self.xn[:self.estScore.size] + self.estScore*self.addSign
            glColor4f(.2,.9,0,.5)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3,GL_FLOAT,0,est_x)
            glDrawElements(GL_LINES,self.inds.size,GL_UNSIGNED_SHORT,self.inds)
            glDisableClientState(GL_VERTEX_ARRAY)

    # ----------------------------------------------------------------------
    # Animated Skeleton
    #

    # Similar to the basic one above.
    # Use ']' and '[' to go forward/backward in time (time referring to noise level)
    def runAnimatedUntilQuit(self, genDataDict):
        while not self.q_pressed:
            for data in genDataDict():
                self.setDataAnimated(**data)
                if 'cams' in data and data['cams'] is not None: self.setCameraData(**data)
                if 'attnMaps' in data and data['attnMaps'] is not None: self.setAttnMaps(**data)
                else: self.attnMaps = None

                while not (self.n_pressed or self.q_pressed):
                    self.startFrame()
                    if self.lastKey == "[": self.animationTime = max(self.animationTime - 1, 0)
                    if self.lastKey == "]": self.animationTime = min(self.animationTime + 1, self.animTrue.shape[0]-1)
                    if self.lastKey == "m": self.attentionMapIdx = self.attentionMapIdx + 1
                    self.render()
                    self.endFrame()
                self.n_pressed = False
                if self.q_pressed: break
        self.q_pressed = False

    def setDataAnimated(self, inds, true, noisy, estScore, ts, **kw):
        # print(f' - set data animated :: {true.shape}')
        self.inds = inds.cpu().numpy().astype(np.uint16)
        self.animationTime = 0
        self.animTrue = true
        self.animNoisy = noisy
        self.animEst = estScore
        self.animTs = ts

    def setCameraData(self, z, cams, **kw):
        N,C = cams.size()
        assert N == 1
        self.cam = cams[0].cpu().numpy()
        self.z = z[0].cpu().numpy()

    def setAttnMaps(self, attnMaps, **kw):
        self.attnMaps = [m.cpu().numpy().astype(np.float32) for m in attnMaps]


    def renderSkeletonAnimated(self):
        I = self.animationTime
        assert I >= 0

        x,xn,estScore = self.animTrue[I], self.animNoisy[I], self.animEst[I]
        xe = xn[:estScore.size] + estScore*self.addSign

        colors = ((1,1,1,1),
                (1,1,0,.5),
                (.2,.9,0,.5))

        for color, xx in zip(colors, (x,xn,xe)):
            glColor4f(*color)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3,GL_FLOAT,0,xx)
            glDrawElements(GL_LINES,self.inds.size,GL_UNSIGNED_SHORT,self.inds)
            glDisableClientState(GL_VERTEX_ARRAY)

        camPts = self.renderCam()

        self.renderAttn(x,camPts)

    def renderAttn(self, x, camPts):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        if self.attnMaps is not None and len(self.attnMaps)>0:

            # 0 means off.
            idx = (self.attentionMapIdx) % (1+len(self.attnMaps))
            if idx == 0:
                return

            map = self.attnMaps[idx-1]
            if map is not None:

                from matplotlib.cm import inferno
                weight = map[self.animationTime]

                # [L,3]
                x = x.reshape(-1,3)
                print(x.shape,camPts.shape)
                pos0 = np.concatenate((x,camPts),0) if camPts is not None else x
                # [L,L,4]
                weight = weight / weight.max(0,keepdims=True)
                col0 = inferno(weight)[...,:].astype(np.float32)
                # print('size0',pos0.shape, col0.shape, 'weight',weight.shape)
                # [L,7]
                # verts0 = np.concatenate((pos0, col0), -1)

                L = weight.shape[0]
                assert L == len(pos0)
                l = np.arange(L,dtype=np.uint8)
                pairs = np.stack(np.meshgrid(l,l), -1) # [L,L,2]

                # FIXME: Is the layout matching?
                pos1 = pos0[pairs] # [L,L,2,3]
                col1 = col0[...,np.newaxis,:].repeat(2,2) # [L,L,2,3]
                # print('size1',pos1.shape, col1.shape)

                # [L*L,3]
                verts = np.concatenate((pos1,col1),-1)
                # print('verts',verts.shape, verts.dtype)


                glColor4f(1,1,1,1)
                glEnableClientState(GL_VERTEX_ARRAY)
                glEnableClientState(GL_COLOR_ARRAY)
                glVertexPointer(3,GL_FLOAT, 28, verts)
                glColorPointer(4,GL_FLOAT, 28, ctypes.c_void_p(verts.ctypes.data+12))
                glDrawArrays(GL_LINES,0,L*L)
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_COLOR_ARRAY)




    def renderCam(self):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        if self.cam is not None:
            u,v = self.cam[0:2]
            wh = self.cam[2:4]
            eye = self.cam[4:7]
            R = self.cam[7:7+9].reshape(3,3)

            CIV = np.eye(4,dtype=np.float32)
            CP = np.eye(4,dtype=np.float32)
            CIV[:3,:3] = R.T
            CIV[:3,3 ] = eye
            n = .1
            f = 1.
            uu,vv = u, v
            uu,vv = .5*u, .5*v
            # uu,vv = 1,1
            # u,v = 2*u,2*v
            CP[:] = np.array((
                1/uu, 0,0,0,
                # 0, -1/v, 0,0,
                0, 1/vv, 0,0,
                0,0, (f+n)/(f-n), -2*f*n/(f-n),
                0,0, 1,0),dtype=np.float32).reshape(4,4)
            model = CIV @ np.linalg.inv(CP)

            # Actually: don't use gl matrices here because we may need the points for later effects: just compute them on cpu.

            # Frustum.
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            mv = self.view @ model
            glLoadMatrixf(mv.T.reshape(-1))
            glColor4f(1,1,1,1)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3,GL_FLOAT,0,frustum_pts)
            glDrawElements(GL_LINES,frustum_inds.size,GL_UNSIGNED_SHORT,frustum_inds)
            glDisableClientState(GL_VERTEX_ARRAY)
            glPopMatrix()

            # Points.
            # pts0 = self.z.reshape(-1,2) / wh - .5
            # pts0 = self.z.reshape(-1,2) * .5 * (u,v)
            pts0 = self.z.reshape(-1,2) #* 1 * (u,v)

            ptsFar = np.ones((pts0.shape[0], 4), dtype=np.float32)
            ptsFar[:,:2] = pts0
            ptsFar[:,2] *= 1+3e-0
            ptsFar = (ptsFar @ (model.T))
            # ptsFar = ptsFar[...,:3] / ptsFar[...,3:]

            pts = np.ones((pts0.shape[0], 4), dtype=np.float32)
            pts[:,:2] = pts0
            pts[:,2] *= 1
            pts = (pts @ (model.T))
            # pts = pts[...,:3] / pts[...,3:]

            # print(pts.shape)
            glColor4f(1,0,1,1)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(4,GL_FLOAT,0,pts)
            glDrawArrays(GL_POINTS,0,len(pts))
            glDisableClientState(GL_VERTEX_ARRAY)

            # Lines connecting points.
            glColor4f(1,0,1,.2)
            # pts1 = np.hstack((pts*(1,1,1), pts*(1,1,3))).reshape(-1,3)
            pts1 = np.hstack((pts, ptsFar)).reshape(-1,4)
            # print(pts1)
            # pts1 = pts1[...,:3]/pts1[...,3:]
            # print(pts1)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(4,GL_FLOAT,0,pts1)
            glDrawArrays(GL_LINES,0,len(pts1))
            glDisableClientState(GL_VERTEX_ARRAY)

            pts = pts[...,:3] / pts[...,3:]

        else: pts = None
        return pts

    # ----------------------------------------------------------------------
    # Langevin Sampled Skeletons
    #
    # Used by ld_sampler.py
    #
