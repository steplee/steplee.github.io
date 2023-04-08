import torch
from renderBase import *
from OpenGL.GL.shaders import compileProgram, compileShader

def compile_shader(vsrc, fsrc):
    vs = compileShader(vsrc, GL_VERTEX_SHADER)
    fs = compileShader(fsrc, GL_FRAGMENT_SHADER)
    return compileProgram(vs,fs)

#
# Allows batch rendering of cubes of same size. Can be used to render an octree in just a few draw calls.
#
# The trick is in how the instancing is setup. You can use instancing to avoid copying data
# It took me a few tries to figure out, but basically you want to use N instances and an attribute divisor of 1,
# where N is the batch size.
# You render 24 indices per isntance, which are actually 2 inds per the 12 edges. The position offset is handled in
# the shader. The cube center (or corner) is the attribute, it is instanced 24x to each vertex shader.
#
class GridEntity():
    vsrc = '''#version 440

    in layout(location=0) vec4 a_pos_alpha;

    uniform layout(location=0) mat4 u_mvp;
    uniform layout(location=1) float u_size;
    uniform layout(location=2) vec4 u_color;

    out vec4 v_color;

    void main() {
            vec3 pos = a_pos_alpha.xyz;
            float alpha = a_pos_alpha.w;

            uint lineInds[] = uint[](
                0,1, 1,2, 2,3, 3,0,
                4,5, 5,6, 6,7, 7,4,
                0,4, 1,5, 2,6, 3,7
            );

            vec3 offsets[] = vec3[](
                vec3(-1.,-1.,-1.),
                vec3( 1.,-1.,-1.),
                vec3( 1., 1.,-1.),
                vec3(-1., 1.,-1.),
                vec3(-1.,-1., 1.),
                vec3( 1.,-1., 1.),
                vec3( 1., 1., 1.),
                vec3(-1., 1., 1.)
            );

            //pos = pos + offsets[gl_InstanceID];
            //pos = pos + offsets[lineInds[gl_InstanceID]];
            //pos = pos + .5 * u_size * offsets[lineInds[2*gl_InstanceID+gl_VertexID]];
            //pos = pos + .5 * u_size * offsets[lineInds[gl_InstanceID]];
            pos = pos + .5 * u_size * offsets[lineInds[gl_VertexID]];
            //pos = pos + .5 * u_size * (offsets[lineInds[gl_VertexID]] + 1.);

            vec4 p1 = u_mvp * vec4(pos, 1.);
            gl_Position = p1;

            // Lighting.
                    float light = .8;

            // Idk.
                    /*
                    int nidx = (gl_VertexID % 2 == 0) ? gl_VertexID + 1 : gl_VertexID - 1;
                    vec3 npos = pos + .5 * u_size * offsets[lineInds[nidx]];
                    vec4 p2 = u_mvp * vec4(npos, 1.);
                    vec3 lineDir = normalize(p1.xyz/p1.w - p2.xyz/p2.w);
                    // float light = 1.-pow(abs(normalize(lineDir).z), .5);
                    light = light * clamp(9200.*lineDir.z,0.,1.) + .2;
                    */

            // Camera acts as focus: only show lines near center of screen.
                    //light = light * smoothstep(0., 2., 1./length(p1.xy/p1.z));
                    light = light * clamp(smoothstep(0.0, 12.5, 7./length(p1.xy/p1.z)), .1, 1.);


            v_color = u_color*vec4(vec3(1.),alpha) * light;
    }

    '''
    fsrc = '''#version 440
    in vec4 v_color;
    out vec4 o_color;
    void main() {
            o_color = v_color;
    }
    '''

    def __init__(self):
        self.prog = compile_shader(GridEntity.vsrc, GridEntity.fsrc)

    def render(self, coordAlphas, size, alpha, mvp, coordAlphasVbo=None,N=None):
        if N is None: N = len(coordAlphas)
        glUseProgram(self.prog)
        glUniformMatrix4fv(0, 1, True, mvp)
        glUniform1f(1, size)
        glUniform4f(2, 1,1,1,alpha)
        glEnableVertexAttribArray(0)

        if coordAlphasVbo is None:
            glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, coordAlphas)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, coordAlphasVbo)
            glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glVertexAttribDivisor(0, 1)

        glDrawArraysInstanced(GL_LINES, 0, 24, N)
        glDisableVertexAttribArray(0)
        glUseProgram(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


class GridRenderer(SurfaceRenderer):
    def __init__(self, wh):
        super().__init__(wh)

        self.octree_verts = []
        self.octree_values = []

    def do_init(self):
        self.gridEnt = GridEntity()
        self.octree_vbos = glGenBuffers(20)
        self.octree_sizes = [0,]*20
        self.curLvl = 0

    def renderGrids(intCoords, sizes):
        pass

    def render(self):
        super().render()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        V = glGetFloatv(GL_MODELVIEW_MATRIX).T
        P = glGetFloatv(GL_PROJECTION_MATRIX).T
        mvp = P @ V
        '''
        coordAlphas = np.array((
            .5,.5,.1, 1.,
            .0,.0,-.5, .4,
            ),dtype=np.float32).reshape(-1,4)
        self.gridEnt.render(coordAlphas, .2, 1., mvp)
        '''

        self.render_octree(mvp)

    def set_octree(self, octree):
        # self.octree_verts = []
        self.octree_values = []
        for lvl in range(octree.numLvls()):
            st = octree.getLvl(lvl)
            coords,vals = st.indices(), st.values()
            self.octree_values.append(v.cpu().numpy())

            # Do verts.
            L = 1 << lvl
            coords = coords.t().float().div(float(L)).cpu().numpy() + (.5/L)
            if lvl == 8: print(verts)
            verts = np.hstack((coords, np.ones_like(coords[:,:1]))).astype(np.float32)
            verts = np.copy(verts,'C')
            self.octree_sizes[lvl] = len(verts)
            glBindBuffer(GL_ARRAY_BUFFER, self.octree_vbos[lvl])
            glBufferData(GL_ARRAY_BUFFER, verts.size*verts.itemsize, verts, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_octree(self, mvp):

        glDisable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        numLvls = len(self.octree_values)
        minLvl,maxLvl = 0, numLvls
        lvl = self.curLvl
        minLvl,maxLvl = lvl, lvl+1
        for lvl in range(minLvl,maxLvl):
            # verts,vals = self.octree_verts[lvl], self.octree_values[lvl]
            vbo,vals = self.octree_vbos[lvl], self.octree_values[lvl]
            N = self.octree_sizes[lvl]
            L = (1 << lvl)
            W = .95 / L
            # self.gridEnt.render(verts, W, 1., mvp)
            self.gridEnt.render(None, W, .4, mvp, coordAlphasVbo=vbo,N=N)

    def keyboard(self, key, x, y):
        super().keyboard(key,x,y)
        self.curLvl
        key = (key).decode()
        if key == 'k': self.curLvl = max(self.curLvl-1,0)
        if key == 'l': self.curLvl = min(self.curLvl+1,len(self.octree_values)-1)




if __name__ == '__main__':

    from tree import Tree
    torch.manual_seed(0)
    maxLvl = 12
    k = torch.randn(5_000_000, 3)
    print(k)
    v = torch.randn(k.size(0))
    ot = Tree()
    ot.set(k,v, maxLvl=maxLvl, do_average=True)

    r = GridRenderer((1024,)*2)
    r.init(True)
    r.set_octree(ot)


    while not r.q_pressed:
        r.startFrame()
        r.render()
        r.endFrame()
