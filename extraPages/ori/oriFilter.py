import numpy as np, os, sys
import dominate, dominate.tags as D
from dominate.util import raw
import sympy, time, json
from sympy import symbols, Matrix, cos as cos_, sin as sin_, diff, simplify, pprint, count_ops, cse, Abs
from sympy.utilities.lambdify import lambdify
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt

'''

This will qualitatively and quantitavely evaluate different orientation state
representations for an EKF.

Simulated data is generated (gyro for time evolution, magnometer + accel as
measurements), and the filters can be run on them.
There's two measurements:
    one from acceloremeter (when low pass filtered, predicts down),
    one from magnometer (predicts north)
These are dotted with the appropriate row of the orientation matrix to
give two scalar residuals for the EKF.

Estimates from filters are evaluated via statistics and a WebGL viz.

First I will start with the roll-pitch-yaw representation.
I am using SymPy to automatically do the math (there's some nasty jacobians)

Next I will use two models based on quaternions.
    1) Additive (i.e. compose by coeff add: q <- q + dwxyz)
    2) Multiplicative (i.e. multiply q <- q * [1 dx dy dz])
Then I will use the quat+rotVec Lie theory based framework.

I expect to find that the quat+rotVec SO3 rep works best.
But I want to build up to it!

'''

########################################################################################
# Jacobians & MathUtils
########################################################################################

# Mine is a Y-X-Z system:
#    1) yaw   around y
#    2) pitch around x
#    3) roll  around z
# This is the same as the more standard ZYX, but Z and Y are swapped and Z flipped.
# This is more intuitive for the image I have in my head:
# The front of the vehicle is Z+ and at id points north
# The right-side           is X+ and at id points east
# down                     is Y+ and at id points down into Earth
r,p,y = sympy.symbols('r p y')
x,y,z = sympy.symbols('x y z')
R = Matrix([[cos_(r), -sin_(r), 0], [sin_(r), cos_(r), 0], [0, 0, 1]])
P = Matrix([[1, 0, 0], [0, cos_(p), -sin_(p)], [0, sin_(p), cos_(p)]])
Y = Matrix([[cos_(y), 0, -sin_(y)], [0, 1, 0], [sin_(y), 0, cos_(y)]])
RPY = R*P*Y

def skew(v):
    return np.array((
          0,   -v[2], v[1],
         v[2], 0,     -v[0],
        -v[1], v[0], 0)).reshape(3,3)

# Note: this breaks down at theta=0
def rodrigues(k):
    theta = np.clip(np.linalg.norm(k), 1e-9, 9999)
    K = skew(k/theta)
    return np.eye(3) + np.sin(theta) * K + (1-np.cos(theta)) * K@K

def eval_expr(expr, **kwargs):
    v = expr.subs(kwargs)
    v = np.array(v.tolist())
    v = v.astype(np.float32)
    return v

def rpy_to_R(rpy):
    return eval_expr(RPY, r=rpy[0], p=rpy[1], y=rpy[2])

def skew_sympy(v):
    return sympy.Matrix([
        [0,   -v[2], v[1]],
        [v[2], 0,     -v[0]],
        [-v[1], v[0], 0]])

def log_R(R):
    t = np.arccos((np.trace(R) - 1) * .5)
    d = np.linalg.norm((R[2,1]-R[1,2], R[0,1]-R[1,0], R[0,2]-R[2,0]))
    return np.array((R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1])) * t / d

# print(' - cycle 000', log_R(rodrigues(np.array((1e-9,0,0)))))
# print(' - cycle 001', log_R(rodrigues(np.array((1e-9,0,1)))))
# print(' - cycle 011', log_R(rodrigues(np.array((1e-9,1,1))))) sys.exit(0)

# Note: this breaks down at theta=0
def rodrigues_sympy(k):
    theta = k.norm()
    K = skew_sympy(k/theta)
    return sympy.eye(3) + sympy.sin(theta) * K + (1-sympy.cos(theta)) * K*K

# Deriving is easy, but done by hand:
#      For the two atans: write out the RPY matrix in trig form,
#      match pairs of elements so that after dividing one factor
#      disappears and the other reduces to sin/cos (i.e. tangent).
#      For the arcsin one, just use the lone sin term
# Note: this is highly dependent on below RPY formulation
def matrix_to_rpy_sympy(R):
    from sympy import asin, atan2, Matrix
    return Matrix([atan2(-R[0,1], R[1,1]), asin(R[2,1]), atan2(R[2,0],R[2,2])])
def matrix_to_rpy(R):
    return np.array((np.arctan2(-R[0,1], R[1,1]), np.arcsin(R[2,1]), np.arctan2(R[2,0], R[2,2])))

# Test via cycle
# print('cycle rpy [0 0 .5]:', matrix_to_rpy_sympy(rpy_to_R([0,0,.5])))
# print('cycle rpy [0 .1 .5]:', matrix_to_rpy_sympy(rpy_to_R([0,0.1,.5])))
# print('cycle rpy [-.1 .1 .5]:', matrix_to_rpy_sympy(rpy_to_R([-.1,0.1,.5])))
# print('cycle rpy [-.1 1.1 2.5]:', matrix_to_rpy_sympy(rpy_to_R([-.1,1.1,2.5]))); sys.exit(0)

# Return Jacobian matrices for time and measurmenet steps
# The time expr must be given k1-k3, the gyro, and kn2, its norm
# The observe expr must be given m1-m3 and g1-g3, the mag+grav observations
# Of course, both most also be given r, p, & y
def gen_rpy_code(approximate=True):
    print(' - Rpy Matrix:\n')
    pprint(RPY)

    k1,k2,k3 = sympy.symbols('k1 k2 k3')
    k = Matrix([k1,k2,k3])

    # jacobians for time step
    if 1:
        Rs = Matrix(RPY)
        Rk = rodrigues_sympy(k)
        Rp = (Rs * Rk)
        # fRp = Rp @ Matrix([1,1,1])
        fRp = matrix_to_rpy_sympy(Rp)
        Jplus = Matrix(diff(fRp, Matrix([r,p,y]))[:,0,:,0])

        '''
        intr,expr = cse(Jplus)
        for intr_ in intr: print(intr_)
        pprint(expr)
        print(count_ops([intr,expr],True))
        '''

        # Substitute the common expr for |k|
        Jplus = Jplus.subs(Abs(k1)**2+Abs(k2)**2+Abs(k3)**2, symbols('kn2'))
        Jplus = Jplus.subs((k2)**2+(k1)**2+(k1)**2, symbols('kn2'))
        if approximate:
            # NOTE: This makes an *APPROXIMATION*
            #       Assume that all k are small, and therefore k**2 is zero, and so are any ka*kb pair
            for k in k1,k2,k3: Jplus = Jplus.subs(k**2, 0)
            for k in k1,k2,k3: Jplus = Jplus.subs(Abs(k)**2, 0)
            for a,b in (k1,k2),(k1,k3),(k2,k3): Jplus = Jplus.subs(a*b, 0)
        # print(Jplus)
        k = 0,0,.01
        kn2 = np.array(k)@k
        # print(Jplus.subs(dict(r=0,p=0,y=0, kn2=kn2,k1=k[0], k2=k[1], k3=k[2])))
        # print(Jplus.subs(dict(r=0,p=0,y=.1, kn2=kn2,k1=k[0], k2=k[1], k3=k[2])))

    # Jacobians for measurements
    # Assume we observe (grav mag), which are the (Y Z) axes
    if 1:
        # print(' - Rpy Matrix:'); pprint(RPY)
        g = Matrix([symbols('g1 g2 g3')])
        m = Matrix([symbols('m1 m2 m3')])
        # z = Matrix([g,m])
        Rs = Matrix(RPY)
        fRp_g = Rs[:,1].T @ g.T
        fRp_m = Rs[:,2].T @ m.T
        # fRp_g = Rs[1,:] @ g.T
        # fRp_m = Rs[2,:] @ m.T
        Jobs_g = Matrix(diff(fRp_g, Matrix([r,p,y])))
        Jobs_m = Matrix(diff(fRp_m, Matrix([r,p,y])))
        # print('Jobs_grav'); pprint(Jobs_g)
        # print('Jobs_mag'); pprint(Jobs_m)
        Jobs = Matrix([[Jobs_g, Jobs_m]])
        return Jplus, Jobs


# gen_rpy_code(); sys.exit(0)

########################################################################################
# Simulation Data
########################################################################################

GRAVITY = 9.92

# Return tensor of [N,9] measurments (acc, gyr, mag)
def generateSimData_1(N=100, sigmas=None):
    if sigmas is None: sigmas = np.ones(9)

    # Note: The R matrices should be f64 even if all else f32
    out = np.zeros((N,13),dtype=np.float32)
    trueRs = np.zeros((N,3,3),dtype=np.float64)
    trueRs[0] = np.eye(3)
    acc,gyr,mag = [np.zeros(3,dtype=np.float64) for _ in range(3)]
    rvec = np.zeros(3,dtype=np.float32) + 1e-9
    # speed = .1 / N
    speed = 2 / N
    for i,t in enumerate(np.linspace(0,1, N)):
        if t < .01:
            new_gyr = 0,0,0
        elif t < .1:
            new_gyr = (0,speed,0) #Messed up, this should be yaw but is pitch
        elif t < .3:
            new_gyr = (speed*.2,0,0)
        elif t < .5:
            new_gyr = (0,-speed,0)
        elif t < .7:
            new_gyr = (-speed*.2,0,0)
        else:
            new_gyr = (0,0,0)
        trueRs[i] = trueRs[i-1 if i-1 >= 0 else 0] @ rodrigues(new_gyr)
        R = rodrigues((rvec+new_gyr)*.5)
        rvec = rvec + new_gyr
        # rvec = log_R(R)
        # print(rvec)
        gyr[:] = new_gyr
        mag[:] = R.T @ (0,0,1)
        acc[:] = R.T @ (0,0,GRAVITY)
        out[i] = t, *acc, *gyr, *mag, *rvec

    return out, trueRs

########################################################################################
# Filters
########################################################################################

class BaseFilter():
    def __init__(self, name):
        self.name = name
        self.estimates = []
        self.covs = []
    def step(self, vals):
        raise NotImplementedError()
    def getRotAndCov(self, i):
        return rpy_to_R(self.estimates[i]), self.covs[i]

class RpyFilter(BaseFilter):
    def __init__(self, approx):
        super().__init__('Rpy' + ('Approx' if approx else ''))
        self.lastTime = 0
        self.Jplus, self.Jobs = gen_rpy_code(approx)
        self.Jplus_ = lambdify(symbols('r p y k1 k2 k3 kn2'), self.Jplus, 'numpy', cse=True)
        self.Jobs_ = lambdify(symbols('r p y g1 g2 g3 m1 m2 m3'), self.Jobs, 'numpy', cse=True)
        import inspect
        # print(' - source:\n', (inspect.getsource(self.Jplus_))
        self.rpy = np.array([1e-9,1e-9,1e-9],dtype=np.float32)
        self.cov = np.eye(3,dtype=np.float32)

    def step(self, vals, truth):
        t, acc, gyr, mag, rvec = vals[0], vals[1:4], vals[4:7], vals[7:10], vals[10:]
        dt = self.lastTime - t

        # ---------------------------------
        # Time Step: spin according to gyro
        # ---------------------------------
        if 0:
            # Slow interpreted sympy version
            F = eval_expr(self.Jplus,
                    r=self.rpy[0], p=self.rpy[1], y=self.rpy[2],
                    k1=gyr[0], k2=gyr[1], k3=gyr[2], kn2=gyr@gyr).T
        else:
            # Faster sympy->numpy version
            F = self.Jplus_(self.rpy[0], self.rpy[1], self.rpy[2], gyr[0], gyr[1], gyr[2], gyr@gyr+1e-9).T


        print('gyr:', gyr)
        print('F:\n',F)
        self.rpy = matrix_to_rpy(rpy_to_R(self.rpy) @ rodrigues(gyr)) % (2. * np.pi)
        # self.rpy = (self.rpy + F @ gyr) % (2 * np.pi)
        Q = np.eye(3) * 1e-5 # TODO:
        self.cov = F @ self.cov @ F.T + Q

        # ---------------------------------
        # Time Step: observe mag/grav
        # ---------------------------------
        if 1:
            # obs_zplus_res =
            # obs_zplus_res =
            R = np.eye(2) * 1e2 # TODO:
            u_grav = acc / np.linalg.norm(acc)
            u_mag = mag / np.linalg.norm(mag)
            H = self.Jobs_(*self.rpy, *u_grav, *u_mag).T
            y = H @ self.rpy
            S = H @ self.cov @ H.T + R
            K = self.cov @ H.T @ np.linalg.inv(S)
            # dx = (K @ -y) % (2 * np.pi)
            dx = (K @ y)
            print('H:\n',H)
            print('innov:',y)
            print('K:\n',K)
            print('dx:',dx)
            self.rpy = (self.rpy + dx) % (2 * np.pi)
            self.cov = self.cov - K @ H @ self.cov

        print('cov:\n', self.cov)
        print('rpy:', self.rpy)
        print('true R:\n', truth)
        print('pred R:\n', rpy_to_R(self.rpy))
        # if np.linalg.norm(gyr) > 1e-6: input()


        self.estimates.append(np.copy(self.rpy))
        self.covs.append(np.copy(self.cov))

        self.lastTime = t


########################################################################################
# Generate Web Page
########################################################################################

def writeWebPage(valss, truth, filterOutputs):
    outDir = os.getcwd()
    outHtml = 'ori.html'
    outJson = 'simAndFilterData.json'
    if 'extraPages' not in outDir: outDir = os.path.join(outDir, 'extraPages')
    if 'ori' not in outDir[-5:]: outDir = os.path.join(outDir, 'ori')
    outPathHtml = os.path.join(outDir,outHtml)
    outPathJson = os.path.join(outDir,outJson)

    # Generate json data and save to file
    jobj = dict(simData=valss.reshape(-1).tolist(), simDataShape=valss.shape)
    for i,(name,fo) in enumerate(filterOutputs.items()):
        pass
    with open(outPathJson, 'w') as fp:
        json.dump(jobj, fp)


    with open(outPathHtml, 'w') as fp:
        with dominate.document(title='ori') as doc:
            doc.body.add(D.script(type='text/javascript', src='runOri.bundled.js'))

            fp.write(doc.render())
    print(' - wrote',outPathHtml)
    print(' - wrote',outPathJson)

def run_filter(filt, valss, truth):
    for i, vals in enumerate(valss):
        print(i, '/', len(valss))
        filt.step(vals, truth[i])
        # if i > 20: sys.exit(0)
        # time.sleep(1)

    errs = []
    sigmas = []
    plt.title('Err/Sig')
    for i,vals in enumerate(valss):
        time, acc, gyr, mag, rvec = vals[0], vals[1:4], vals[4:7], vals[7:10], vals[10:]
        estRot,estCov = filt.getRotAndCov(i)
        estVec = log_R(estRot)
        sig = np.sqrt(np.cbrt(np.linalg.eig(estCov)[0].prod()))
        err = np.linalg.norm(estVec - rvec)
        if i % 10 == 0: print('err',err, 'from',estVec, rvec,'sigma',sig)
        errs.append(err)
        sigmas.append(sig)
    plt.plot(errs, color='r', label='err')
    plt.plot(sigmas, color='y', label='sig')
    plt.legend()
    plt.savefig('err_'+filt.name+'.jpg')
    plt.clf()



if __name__ == '__main__':

    valss,truth = generateSimData_1()

    filterOutputs = {}
    filt = RpyFilter(True)
    filterOutputs[filt.name] = run_filter(filt, valss, truth)
    # filt = RpyFilter(True)
    # filterOutputs[filt.name] = run_filter(filt, valss, truth)

    # writeWebPage(valss, truth, filterOutputs)
