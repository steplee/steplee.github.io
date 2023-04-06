import numpy as np
import time, sys
import sys, cv2
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import OpenGL.GL.shaders

class SingletonApp:
    _instance = None

    def __init__(self, wh, name='Viz'):
        SingletonApp._instance = self
        self.wh = wh
        self.window = None

        self.last_x, self.last_y = 0,0
        self.left_down, self.right_down = False, False
        self.scroll_down = False
        self.left_dx, self.left_dy = 0,0
        self.right_dx, self.right_dy = 0,0
        self.scroll_dx, self.scroll_dy = 0,0
        self.name = name
        self.pickedPointClipSpace = None

    def do_init(self):
        raise NotImplementedError('must implement')

    def render(self):
        raise NotImplementedError('must implement')

    def idle(self, rs):
        self.render(rs)

    def keyboard(self, *args):
        #sys.exit()
        pass

    def mouse(self, but, st, x,y):
        if but == GLUT_LEFT_BUTTON and (st == GLUT_DOWN):
            if not self.left_down: self.pick(x,y)
            self.last_x, self.last_y = x, y
            self.left_down = True
        else:
            self.pickedPointClipSpace = None
        if but == GLUT_LEFT_BUTTON and (st == GLUT_UP):
            self.left_down = False
        if but == GLUT_RIGHT_BUTTON and (st == GLUT_DOWN):
            self.last_x, self.last_y = x, y
            self.right_down = True
        if but == GLUT_RIGHT_BUTTON and (st == GLUT_UP):
            self.right_down = False
        if but == 3 and (st == GLUT_DOWN):
            self.scroll_dy = self.scroll_dy * .7 + .9 * (-1) * 1e-1
        if but == 4 and (st == GLUT_DOWN):
            self.scroll_dy = self.scroll_dy * .7 + .9 * (1) * 1e-1
    def motion(self, x, y):
        if self.left_down:
            self.left_dx = self.left_dx * .5 + .5 * (x-self.last_x) * 1e-1
            self.left_dy = self.left_dy * .5 + .5 * (y-self.last_y) * 1e-1
        if self.right_down:
            self.right_dx = self.right_dx * .5 + .5 * (x-self.last_x) * 1e-1
            self.right_dy = self.right_dy * .5 + .5 * (y-self.last_y) * 1e-1

        self.last_x, self.last_y = x,y

    def reshape(self, w,h):
        glViewport(0, 0, w, h)
        self.wh = w,h

    def _render(*args):
        glutSetWindow(SingletonApp._instance.window)
        SingletonApp._instance.render(*args)
    def _idle(*args):
        glutSetWindow(SingletonApp._instance.window)
        SingletonApp._instance.idle(*args)
    def _keyboard(*args): SingletonApp._instance.keyboard(*args)
    def _mouse(*args):
        SingletonApp._instance.mouse(*args)
    def _motion(*args): SingletonApp._instance.motion(*args)
    def _reshape(*args): SingletonApp._instance.reshape(*args)

    def init(self, init_glut=False):
        if init_glut:
            glutInit(sys.argv)
            glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
            #glutSetOption(GLUT_MULTISAMPLE, 4)
            #glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GL_MULTISAMPLE)
            #glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);

        glutInitWindowSize(*self.wh)
        self.reshape(*self.wh)
        self.window = glutCreateWindow(self.name)
        #glutSetWindow(self.window)
        glEnable(GL_MULTISAMPLE);

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glAlphaFunc(GL_GREATER, 0)
        glEnable(GL_ALPHA_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.do_init()
        glutReshapeFunc(SingletonApp._reshape)
        glutDisplayFunc(SingletonApp._render)
        glutIdleFunc(SingletonApp._idle)
        glutMouseFunc(SingletonApp._mouse)
        glutMotionFunc(SingletonApp._motion)
        glutMotionFunc(SingletonApp._motion)
        glutKeyboardFunc(SingletonApp._keyboard)

    def run_glut_loop(self):
        glutMainLoop()

    def pick(self, x,y):
        y = self.wh[1] - y - 1
        z = float(glReadPixels(x,y, 1,1, GL_DEPTH_COMPONENT, GL_FLOAT).squeeze())
        x = 2 * x / self.wh[0] - 1
        #y = -(2 * y / self.wh[1] - 1)
        y = (2 * y / self.wh[1] - 1)
        self.pickedPointClipSpace = np.array((x,y,1)) * z

class SurfaceRenderer(SingletonApp):
    def __init__(self, wh):
        super().__init__(wh, 'SurfaceRenderer')
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


    def do_init(self):
        pass

    def render(self):
        glViewport(0, 0, *self.wh)
        glMatrixMode(GL_PROJECTION)
        n = .001
        v = .5
        u = (v*self.wh[0]) / self.wh[1]
        glLoadIdentity()
        glFrustum(-u*n,u*n,-v*n,v*n,.002,100)

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(self.view.T.reshape(-1))

        glClearColor(0,0,0,1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBegin(GL_LINES)
        glColor4f(1,0,0,1); glVertex3f(1,0,0); glVertex3f(0,0,0);
        glColor4f(0,1,0,1); glVertex3f(0,1,0); glVertex3f(0,0,0);
        glColor4f(0,0,1,1); glVertex3f(0,0,1); glVertex3f(0,0,0);
        glEnd()

    def startFrame(self):
        self.q_pressed = False
        self.n_pressed = False
        glutMainLoopEvent()
        self.startTime = time.time()

        add_md = 0
        if self.mprev[0] > 0 and self.mprev2[0] > 0: add_md = (self.mprev - self.mprev2)
        self.md = self.md * .9 + .001 * add_md

        self.velTrans = self.velTrans * .94 + .01 * self.accTrans
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
        if key == 'f': self.accTrans[1] = 1

    def motion(self, x, y):
        self.mprev2[:] = self.mprev
        self.mprev[:] = x,y


