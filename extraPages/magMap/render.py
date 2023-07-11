import numpy as np, time
from .renderBase import *

__frustum_pts = np.array((
    -1,-1,0,
     1,-1,0,
     1, 1,0,
    -1, 1,0,
    -1,-1,1,
     1,-1,1,
     1, 1,1,
    -1, 1,1), dtype=np.float32)
__frustum_inds = np.array((
    0,1, 1,2, 2,3, 3,0,
    4+0,4+1, 4+1,4+2, 4+2,4+3, 4+3,4+0,
    0,4, 1,5, 2,6, 3,7),dtype=np.uint8)

def draw_frustum(VI, f,wh):
    uv = (wh/f) * .5
    # VI = np.eye(4, dtype=np.float32)
    # VI[:3,:3] = R
    # VI[:3, 3] = t
    n,f = .1, 1.
    # n,f = .01, np.sin(time.time())*.9 + 1.1
    if 0:
        # PI = np.array((
            # uv[0],0,0,0,
            # 0,uv[1],0,0,
            # 0,0,(f-n),n,
            # 0,0,0,1),dtype=np.float32).reshape(4,4)
        PI_0 = np.array((
            .5*f*uv[0],0,0,0,
            0,.5*f*uv[1],0,0,
            0,0,f,0,
            0,0,0,1),dtype=np.float32).reshape(4,4)
        PI_1 = np.array((
            1,0,0,0,
            0,1,0,0,
            0,0,1,n,
            0,0,0,1),dtype=np.float32).reshape(4,4)
        PI_2 = np.array((
            1,0,0,0,
            0,1,0,0,
            0,0,1,-f*n,
            0,0,(f-n),0),dtype=np.float32).reshape(4,4)
        PI = PI_2@PI_1@PI_0
    else:
        PI = np.linalg.inv(np.array((
            1./uv[0], 0, 0, 0,
            0, 1./uv[1], 0, 0,
            0, 0, 1*(f+n)/(f-n), -2*f*n/(f-n),
            0, 0, 1, 0),dtype=np.float32).reshape(4,4)) # FIXME: don't invert
    M = VI @ PI
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glMultMatrixf(M.T)

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3,GL_FLOAT,0,__frustum_pts)
    glDrawElements(GL_LINES,len(__frustum_inds),GL_UNSIGNED_BYTE,__frustum_inds)
    glDisableClientState(GL_VERTEX_ARRAY)

    glPopMatrix()

def draw_points(pts, inds=None, colors=None):
    assert pts.dtype == np.float32
    assert inds is None or inds.dtype == np.uint32
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3,GL_FLOAT,0,pts)
    if colors is not None:
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4,GL_FLOAT,0,colors)
    if inds is None:
        glDrawArrays(GL_POINTS,0,len(pts))
    else:
        glDrawElements(GL_POINTS,len(inds),GL_UNSIGNED_INT,inds)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

def draw_lines_to_point(opticalCenter, worldPts, colors=None):
    N = worldPts.shape[0]
    a = np.hstack((
        opticalCenter[np.newaxis].repeat(N,0),
        worldPts))
    print(a)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3,GL_FLOAT,0,a)
    if colors is not None:
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4,GL_FLOAT,0,np.hstack((colors,colors)))
    glDrawArrays(GL_LINES,0,N*2)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

if __name__ == '__main__':

    app = SurfaceRenderer((1024,1024))
    app.init(True)

    for i in range(999):
        app.startFrame()
        app.render()
        wh = np.array((512,512),dtype=np.float32)
        f = (wh*.5) / np.tan(np.deg2rad(53)*.5)
        # z = np.sin(time.time()) * .5 + 1
        z=-1
        t = np.array((.5,.5,z),dtype=np.float32)
        R = np.eye(3)
        VI = np.eye(4,dtype=np.float32)
        VI[:3,:3] = R
        VI[:3,3] = t
        draw_frustum(VI, f,wh)

        if i==0:
            pts = np.random.randn(10,3).astype(np.float32)
        inds = np.arange(len(pts),dtype=np.uint32)
        colors = np.random.randn(10,4).astype(np.float32)
        colors[:,3] = 1
        draw_points(pts,inds,colors=colors)

        glColor4f(1,1,1,.1)
        draw_lines_to_point(t,pts)

        app.endFrame()
